import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing import assert_almost_equal

from .metrics import gradientSimilarity, HistogramMatching
from model.SymNet import sym_neg_Jdet_loss, magnitude_loss, smoothloss


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name if name is not None else re.sub('([a-z0-9])([A-Z])', r'\1_\2',
                                                          re.sub('(.)([A-Z][a-z]+)', r'\1_\2',
                                                                 self.__class__.__name__)).lower()

    @property
    def __name__(self):
        return self._name


###########################################################################################
# Gen Loss
class GenLoss(BaseObject):
    def __init__(self, fore_fn, back_fn=None, single=False, fore_eps=None, back_eps=None,
                 gt_seg=False, reduction='exclude', back_weight=1):
        super(GenLoss, self).__init__()
        from params import img_size
        self.fore_fn = fore_fn
        self.back_fn = fore_fn if back_fn is None else back_fn
        self.fore_eps = np.prod(img_size['size']) // 32 if fore_eps is None else fore_eps
        self.back_eps = np.prod(img_size['size']) // 32 if back_eps is None else back_eps
        self.reduction = reduction
        self.back_weight = back_weight
        print(f'{self.__name__}: {self.fore_fn} {self.fore_eps} {self.back_fn} {self.back_eps} {self.reduction}')
        self.single = single
        self.gt = nn.BCELoss() if gt_seg else None
        # if div:
        #     self.div = lambda gt, pred, m: (
        #             m * (gt - (pred.sum((2, 3, 4), keepdim=True) / m.sum((2, 3, 4), keepdim=True))) ** 2
        #     ).sum((1, 2, 3, 4)).mean()
        # else:
        #     self.div = None

    def neg_coeff_constraint_loss(self, x, mask, pred_foreground, pred_background):
        # computes 2-C(F,B), which is equivalent to -C(F,B) as a loss term
        # we have 2-C(F,B) = H(F|B)/H(F) + H(B|F)/H(B), where:
        # H(F|B) = H(X*M | X*(1-M)) = ||M * (X - phi(X*(1-M), 1-M))||_1 = ||M * (X - pred_foreground)||_1
        # H(B|F) = H(X*(1-M) | X*M) = ||(1-M) * (X - phi(X*M, M))||_1 = ||(1-M) * (X - pred_background)||_1
        # H(F) = ||M||
        # H(B) = ||1-M||
        def sim(y_true, y_pred, mask, distance, reduction='exclude', eps=None):
            if reduction == 'exclude':
                H_foreground_given_background = (mask * (1 - distance(y_true, y_pred))).mean(1).sum((1, 2, 3))
                H_foreground = mask.sum((1, 2, 3, 4)) + eps
                C = H_foreground_given_background / H_foreground
            elif reduction == 'sum':
                C = (mask * (1 - distance(y_true, y_pred))).sum(dim=(1, 2, 3, 4))
            else:
                raise ValueError
            return C.mean()

        C_foreground_given_background = -sim(x, pred_foreground, mask, self.fore_fn,
                                             reduction=self.reduction,
                                             eps=self.fore_eps)
        C_background_given_foreground = self.back_weight * sim(x, pred_background, 1 - mask, self.back_fn,
                                                               reduction=self.reduction,
                                                               eps=self.back_eps) if not self.single else 0

        return C_foreground_given_background + C_background_given_foreground

    def forward(self, pred, inp):
        mask, inpainted_b, inpainted_f, *_ = pred
        moving, _, seg, *_ = inp
        if torch.isnan(mask).any().item():
            print('pred mask nan')
            raise ValueError
        if self.gt is not None:
            return self.gt(mask, (seg > .5).float())
        return self.neg_coeff_constraint_loss(moving, mask, inpainted_f, inpainted_b)


class GenNCCMapLoss(GenLoss):
    def __init__(self, single=False, div=False):
        super(GenNCCMapLoss, self).__init__(fore_fn=lambda x, y: 1 - torch.abs(x - y), single=single, div=div)
        self._name = 'gen_loss'

    def forward(self, pred, inp):
        mask, ncc_map, inpainted_b, inpainted_f = pred
        moving, *_ = inp
        if torch.isnan(mask).any().item():
            print('pred mask nan')
            raise ValueError

        if self.div is not None:
            return self.neg_coeff_constraint_loss(ncc_map, mask, inpainted_f, inpainted_b) + \
                .001 * self.div(ncc_map, inpainted_f, mask)
        return self.neg_coeff_constraint_loss(ncc_map, mask, inpainted_f, inpainted_b)


###########################################################################################
# InpLoss
class InpLoss2(BaseObject):
    def __init__(self, ncc_fn, weight=(1, 1, 1), excludes_size_influence=False, single=False,
                 r2atlas=False, merge=False, histmatch=False, rescale=False):
        super(InpLoss2, self).__init__()
        self._name = 'inp_loss'
        self.excludes_size = excludes_size_influence
        self.single = single
        self.r2a = r2atlas
        self.ncc = lambda x, y: - weight[0] * ncc_fn(x, y) if weight[0] != 0 else 0
        self.histmatch = HistogramMatching(differentiable=False, rescale=rescale) if histmatch else None
        if excludes_size_influence:
            mse_fn = lambda m, x, y: weight[1] * (m * nn.MSELoss(reduction='none')(x, y)).sum() / (m.sum() + 1e-5) \
                if weight[1] != 0 else 0

            gradient_fn = lambda m, x, y: -weight[2] * (
                    m * gradientSimilarity(x, y, win=7, reduction='none')).sum() / (m.sum() + 1e-5) \
                if weight[2] != 0 else 0
        else:
            mse_fn = lambda x, y: weight[1] * nn.MSELoss(reduction='mean')(x, y) \
                if weight[1] != 0 else 0

            gradient_fn = lambda x, y: -weight[2] * gradientSimilarity(x, y, win=7) \
                if weight[2] != 0 else 0
        self.fn = [mse_fn, gradient_fn]
        self.merge = merge
        print(f'{self.__name__}: {weight} Histo: [{histmatch} {rescale}]')

    def base(self, moving, mask, y2x, rec):
        # ncc domain transfer
        normal = moving * (1 - mask) + y2x * mask
        if self.merge:
            dif = self.ncc(normal, rec)
        else:
            dif = self.ncc(moving, rec)
        if self.excludes_size:
            for fn in self.fn:
                dif += fn(mask, y2x, rec) + fn(1 - mask, moving, rec)
        else:
            for fn in self.fn:
                dif += fn(normal, rec)
        return dif

    def forward(self, pred, inp):
        mask, y2x, inp_b, inp_f = pred
        moving, fixed, seg_gt, *_ = inp
        if self.r2a:
            if self.histmatch:
                fixed = self.histmatch(fixed, y2x)
            return self.base(y2x, mask, fixed, inp_f) if self.single \
                else self.base(y2x, mask, fixed, inp_f) + self.base(y2x, 1 - mask, fixed, inp_b)
        if self.histmatch:
            y2x = self.histmatch(y2x, moving)
        return self.base(moving, mask, y2x, inp_f) if self.single \
            else self.base(moving, mask, y2x, inp_f) + self.base(moving, 1 - mask, y2x, inp_b)


class InpLoss(BaseObject):
    def __init__(self, ncc_fn=None, sim_fn=None, single=False, exclude_size=False):
        super(InpLoss, self).__init__()
        self.times = 0
        self.single = single
        self.ncc = ncc_fn
        self.grad = lambda x, y: -gradientSimilarity(x, y, 7)
        self.sim = sim_fn
        self.exclude = exclude_size
        print(f'{self.__name__}: ncc {ncc_fn} mix {sim_fn}')

    def base(self, x, mask, y2x, inp):
        if self.ncc:
            loss = 1 - self.ncc(x, inp)
        else:
            loss = 0
        if self.sim:
            if not self.exclude:
                reg_inp_f_normal = x * (1 - mask) + y2x * mask
                loss += self.sim(reg_inp_f_normal, inp, reduction='mean')
            else:
                loss += (mask * self.sim(y2x, inp, reduction='none')).sum() / (mask.sum() + 1e-5)
                loss += ((1 - mask) * self.sim(x, inp, reduction='none')).sum() / ((1 - mask).sum() + 1e-5)
        return loss

    def forward(self, pred, inp):
        pass


class GIR5InpLoss(InpLoss):
    def __init__(self, ncc_fn, single=False, mix=False):
        super(GIR5InpLoss, self).__init__(ncc_fn, single, mix)

    def forward(self, pred, inp):
        mask2y, x2y, inp_b, inp_f = pred
        _, fixed, seg_gt, *_ = inp
        if self.single:
            return self.base(x2y, mask2y, fixed, inp_f)
        return self.base(x2y, mask2y, fixed, inp_f) + self.base(x2y, 1 - mask2y, fixed, inp_b)


class InpLossP(InpLoss):
    def __init__(self, ncc_fn, single=False, exclude_size=False):
        super(InpLossP, self).__init__(ncc_fn=ncc_fn, single=single, sim_fn=None, exclude_size=exclude_size)
        self._name = 'inp_loss'

    def forward(self, pred, inp):
        mask, y2x, inp_b, inp_f = pred
        moving, fixed, seg_gt, *_ = inp
        if self.single == 'fore':
            return self.base(moving, mask, y2x, inp_f)
        return self.base(moving, mask, y2x, inp_f) + self.base(moving, 1 - mask, y2x, inp_b)


class InpLossNCCMap(BaseObject):
    def __init__(self, sim_fn=F.l1_loss, single=False):
        super(InpLossNCCMap, self).__init__(name='inp_loss')
        self.sim_fn = sim_fn
        self.single = single

    def forward(self, pred, inp):
        mask, ncc_map, inpainted_b, inpainted_f = pred
        moving, fixed, seg_gt, *_ = inp
        if self.single:
            return self.sim_fn(ncc_map, inpainted_f, reduction='mean')
        else:
            return self.sim_fn(ncc_map, inpainted_f, reduction='mean') \
                + self.sim_fn(ncc_map, inpainted_b, reduction='mean')


class InpLossGIR6(InpLoss2):
    def __init__(self, ncc_fn, weight, excludes_size_influence, single=False):
        super().__init__(ncc_fn, weight=weight, excludes_size_influence=excludes_size_influence,
                         single=single, r2atlas=True, merge=False)


class RecLoss(BaseObject):
    def __init__(self, win, weight=(1, 1, 1), excludes_size_influence=False, single=False,
                 r2atlas=False, merge=False):
        super(RecLoss, self).__init__()
        from .metrics import lncc
        self._name = 'rec_loss'
        self.excludes_size = excludes_size_influence
        self.single = single
        self.r2a = r2atlas
        self.ncc = lambda x, y: - weight[0] * lncc(x, y, win=win, reduction='mean', threshold=0.1) \
            if weight[0] != 0 else 0
        if excludes_size_influence:
            mse_fn = lambda m, x, y: weight[1] * (m * nn.MSELoss(reduction='none')(x, y)).sum() / (m.sum() + 1e-5) \
                if weight[1] != 0 else 0

            gradient_fn = lambda m, x, y: -weight[2] * (
                    m * gradientSimilarity(x, y, win=7, reduction='none')).sum() / (m.sum() + 1e-5) \
                if weight[2] != 0 else 0
        else:
            mse_fn = lambda x, y: weight[1] * nn.MSELoss(reduction='mean')(x, y) \
                if weight[1] != 0 else 0

            gradient_fn = lambda x, y: -weight[2] * gradientSimilarity(x, y, win=7) \
                if weight[2] != 0 else 0
        self.fn = [mse_fn, gradient_fn]
        self.merge = merge

    def base(self, moving, mask, y2x, rec):
        # ncc domain transfer
        normal = moving * (1 - mask) + y2x * mask
        if self.merge:
            dif = self.ncc(normal, rec)
        else:
            dif = self.ncc(moving, rec)
        if self.excludes_size:
            for fn in self.fn:
                dif += fn(mask, y2x, rec) + fn(1 - mask, moving, rec)
        else:
            for fn in self.fn:
                dif += fn(normal, rec)
        return dif

    def forward(self, pred, inp):
        mask, y2x, inp_b, inp_f = pred
        moving, fixed, seg_gt, *_ = inp
        if self.r2a:
            return self.base(y2x, mask, fixed, inp_f) if self.single \
                else self.base(y2x, mask, fixed, inp_f) + self.base(y2x, 1 - mask, fixed, inp_b)
        return self.base(moving, mask, y2x, inp_f) if self.single \
            else self.base(moving, mask, y2x, inp_f) + self.base(moving, 1 - mask, y2x, inp_b)


class ComRecLoss(RecLoss):
    def __init__(self, win, weight=(1, 1, 1), excludes_size_influence=False):
        super(ComRecLoss, self).__init__(win, weight, excludes_size_influence)

    def forward(self, pred, inp):
        mask, reg, _, inp_f = pred
        moving, fixed, seg_gt, *_ = inp
        return self.base(moving, mask, reg[-1], inp_f)


###########################################################################################
# Registration Loss
class VoxRegLoss(BaseObject):
    def __init__(self, win, eps=1e-5, bidir=True, int_downsize=2, diff=False):
        super(VoxRegLoss, self).__init__()
        from .metrics import lncc
        from model.VoxNet import KLLoss
        from params import img_size
        self.kl = KLLoss(img_sz=img_size[''], bidirectional=bidir)
        self._name = 'reg_loss'
        self.ncc = lambda I, J, mask=None: -lncc(Ii=I, Ji=J, win=win, threshold=eps, reduction='mean', fast=False)
        self.bidir = bidir
        self.diff = diff
        self.smooth = Grad('l2', loss_mult=int_downsize)

    def forward(self, pred, inp):
        x, y, *_ = inp
        if self.diff:
            [pos_flow, neg_flow, flow_mean, log_sigma], x2y, y2x, *_ = pred
            return self.kl(x, y, x2y, y2x, flow_mean, log_sigma)
        else:
            [pos_flow, neg_flow, preint_flow], x2y, y2x, *_ = pred
        if self.bidir:
            loss = .5 * self.ncc(y, x2y) + .5 * self.ncc(x, y2x)
        else:
            loss = self.ncc(y, x2y)
        return loss + self.smooth(preint_flow)


class SymRegLoss(BaseObject):
    def __init__(self, win, magnitude=0.1, eps=1e-5, local_ori=100, smooth=3,
                 sym=True, y2x_rec=False, multi=False, mask=False):
        """
        :param win:
        :param magnitude:
        :param eps:
        :param local_ori:
        :param smooth:
        :param sym:
        :param y2x_rec: Compute the NCC similarity between Y2X and inp(X)
        :param multi: Multi-scale NCC
        :param mask: cost function masking (default false)
        """
        super(SymRegLoss, self).__init__()
        self.win = win
        self.eps = eps
        self.win_list = [win] * 3

        self.mag = magnitude_loss
        self.Jdet = sym_neg_Jdet_loss
        self.smoothloss = smoothloss
        self.magnitude = magnitude
        self.local_ori = local_ori
        self.smooth = smooth
        self.sym = sym
        self.y2x_rec = y2x_rec
        self.mask = mask

        from utils.metrics import lncc, MultiNCC
        if multi:
            self.ncc = MultiNCC(kernel_size=[7, 5, 3],
                                kernel_weight=[1, 1 / 2, 1 / 4],
                                eps=eps,
                                fast=False,
                                reduction='mean')
        else:
            self.ncc = lambda I, J, mask=None: lncc(Ii=I, Ji=J, win=win, threshold=eps, reduction='mean', fast=False)
        print(f'{self.__name__} [win: {win}, eps: {eps}, sym: {sym}]')

    def forward(self, pred, inp):
        [F_xy, F_yx, F_X_Y_half, F_Y_X_half, F_X_Y, F_Y_X], X_Y_half, Y_X_half, X_Y, Y_X, grid, rec, *mask = pred
        X, Y, *_ = inp
        if not self.mask:
            mask = None
        else:
            mask = mask[0]
        if self.y2x_rec:
            X = rec
        loss3 = -self.ncc(Y, X_Y, mask) - self.ncc(X, Y_X, mask)
        loss4 = self.Jdet(F_X_Y.permute(0, 2, 3, 4, 1), grid) + self.Jdet(
            F_Y_X.permute(0, 2, 3, 4, 1), grid)
        loss5 = self.smoothloss(F_xy) + self.smoothloss(F_yx)
        if self.sym:
            loss1 = -self.ncc(X_Y_half, Y_X_half, None)
            loss2 = self.mag(F_X_Y_half, F_Y_X_half)
            loss = loss1 + self.magnitude * loss2 + loss3 + self.local_ori * loss4 + self.smooth * loss5
        else:
            loss = loss3 + self.local_ori * loss4 + self.smooth * loss5
        return loss


##########################################################################################
# Others
class Grad(BaseObject):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, flow):
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0

        if self.loss_mult is not None and self.loss_mult != 1:
            grad *= self.loss_mult
        return grad


class TripleLoss(BaseObject):
    def __init__(self, ncc_fn, margin=1.):
        super(TripleLoss, self).__init__()
        self.loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - ncc_fn(x, y),
            margin=margin
        )

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


class Contrast(torch.nn.Module):
    def __init__(self, device, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(device)
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class SwinLoss(BaseObject):
    def __init__(self, device, batch_size=1):
        super(SwinLoss, self).__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.contrast_loss = Contrast(device, batch_size)
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0
        print(self.__name__)

    def __call__(self, pre, inp):
        output_rot1, output_rot2, output_contrastive, target_contrastive, output_recons1, output_recons2 = pre
        x1, x2, rot1, rot2, *_ = inp

        output_rot = torch.cat([output_rot1, output_rot2], dim=0)
        target_rot = torch.cat([rot1, rot2], dim=0)
        output_recons = torch.cat([output_recons1, output_recons2], dim=0)
        target_recons = torch.cat([x1, x2], dim=0)

        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss
        return total_loss


if __name__ == "__main__":
    pass
