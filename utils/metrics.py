import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
from model.SymNet import generate_grid


class BaseObject(nn.Module):

    def __init__(self, tag=None, name=None, device='cpu'):
        super().__init__()
        self._name = name
        self.tag = tag
        self.device = torch.device(device)
        if self._name is None:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', self.__class__.__name__)
            self._name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @property
    def __name__(self):
        return self._name

    def get_pred(self, pred):
        return pred[self.tag]


class NegJacobian(BaseObject):
    __name__ = 'vox_negj'

    def __init__(self):
        super(NegJacobian, self).__init__(tag='reg')
        self.losses = neg_Jdet_count

    def forward(self, pred, inp):
        [pos_flow, neg_flow, preint_flow, field], x2y, y2x = self.get_pred(pred)
        score = self.losses(field).to('cpu')
        return score


def dice(pr, gt, eps=1e-7):
    """Calculate dice-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
    Returns:
        float: F score
    """
    intersections = (pr * gt).sum(dim=[2, 3, 4])
    cardinalities = (pr + gt).sum(dim=[2, 3, 4])

    score = ((2 * intersections.sum(dim=1) + eps) / (cardinalities.sum(dim=1) + eps)).mean()
    return score


def dice_np(pr, gt, eps=1e-7):
    """Calculate dice-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
    Returns:
        float: F score
    """
    intersections = (pr * gt).sum()
    cardinalities = (pr + gt).sum()

    score = ((2 * intersections.sum() + eps) / (cardinalities.sum() + eps)).mean()
    return score


def iou(pr, gt, eps=1e-7):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
    Returns:
        float: F score
    """
    tp = (gt * pr).sum()
    fp = pr.sum() - tp
    fn = gt.sum() - tp

    score = (tp + eps) / (tp + fn + fp + eps)
    return score


def dice_sets(array1, array2, labels=None):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    array1 = np.round(np.array(array1).squeeze())
    array2 = np.round(np.array(array2).squeeze())
    if labels is None:
        labels = np.intersect1d(array1, array2)
    dicem = {}
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem.update({int(label): top / bottom})
    return dicem


def landmark_mae(moving_landmark, fixed_landmark, flows):
    moving_landmark = torch.tensor(moving_landmark.squeeze())
    fixed_landmark = torch.tensor(fixed_landmark.squeeze())
    flows = torch.tensor(flows.squeeze())
    errors = []
    for i in fixed_landmark[fixed_landmark > 0]:
        f_index = torch.nonzero(fixed_landmark == i)[0]
        m_index = torch.nonzero(moving_landmark == i)[0]

        f_index = flows[tuple(f_index.cpu().numpy())][[2, 1, 0]].round()
        errors.append(torch.abs(m_index - f_index).sum())
    return torch.tensor(errors).mean()


def landmark_mae_df(moving_landmark, fixed_landmark, flows):
    moving_landmark = torch.tensor(moving_landmark).float()
    fixed_landmark = torch.tensor(fixed_landmark).float()
    errors = []
    grid = torch.from_numpy(generate_grid(moving_landmark.shape[2:])).float().to(flows.device) \
           + flows.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]].float()
    for i in range(3):
        grid[..., i] = 2 * grid[..., i] / (moving_landmark.shape[2:][-i - 1] - 1) - 1

    warped_landmark = torch.nn.functional.grid_sample(moving_landmark, grid, mode='nearest', align_corners=True, padding_mode='border')
    for i in fixed_landmark[fixed_landmark > 0]:
        if i in warped_landmark:
            f_index = torch.nonzero(fixed_landmark == i)[0, 2:].numpy()
            w_index = torch.nonzero(warped_landmark == i)[0, 2:].numpy()
            errors.append(w_index - f_index)
    if len(errors) == 0:
        return torch.tensor(0.0)
    return np.linalg.norm(errors, axis=1).mean()


def landmark_mae_np(moving_landmark, fixed_landmark, flows):
    moving_landmark = torch.tensor(moving_landmark.squeeze())
    fixed_landmark = torch.tensor(fixed_landmark.squeeze())
    flows = torch.tensor(flows.squeeze())
    f_indexes, m_indexes = [], []

    for i in fixed_landmark[fixed_landmark > 0]:
        if i in moving_landmark:
            f_indexes.append(torch.nonzero(fixed_landmark == i)[0].cpu().numpy())
            m_indexes.append(torch.nonzero(moving_landmark == i)[0].cpu().numpy())
    if len(f_indexes) == 0:
        return torch.tensor(0.0)

    f_indexes = np.array(f_indexes)
    m_indexes = np.array(m_indexes)
    # TRE
    flows = flows.cpu().numpy()
    fixed_keypoints = f_indexes
    moving_keypoints = m_indexes

    moving_disp_x = map_coordinates(flows[0], moving_keypoints.transpose())
    moving_disp_y = map_coordinates(flows[1], moving_keypoints.transpose())
    moving_disp_z = map_coordinates(flows[2], moving_keypoints.transpose())
    lms_moving_disp = np.array((moving_disp_x, moving_disp_y, moving_disp_z)).transpose()

    warped_moving_keypoint = moving_keypoints - lms_moving_disp
    tre_score = np.linalg.norm((warped_moving_keypoint - fixed_keypoints), axis=1).mean()
    return torch.tensor(tre_score)


class LandMarkMAE(BaseObject):
    def __init__(self, mode='x2y', model='sym'):
        '''
        Args:
            mode: mm (moving landmark) | ff (fixed landmark) | frevf (f -> m -> f) | mrevm (m->f->m)
        '''
        super(LandMarkMAE, self).__init__(tag='reg')
        self._name = f'land_mae_{mode}'
        self.mode = mode
        self.model = model

    def forward(self, pred, inp, *_):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        pred = self.get_pred(pred)
        *_, moving_landmark, fixed_landmark = inp
        shape = moving_landmark.shape[2:]

        if self.model == 'vox':
            [fxy, fyx, *_], X_Y, Y_X, grid, *_ = pred
            flows = {'xy': fxy[:, [2, 1, 0], ...], 'yx': fyx[:, [2, 1, 0], ...]}
        elif self.model == 'DIFVox':
            [fxy, fyx, *_], X_Y, Y_X, grid, *_ = pred
            fxy = fxy[:, [2, 1, 0], ...]
            # for i in range(3):
            #     fxy[:, i, ...] = (fxy[:, i, ...] + 1) * (shape[i] - 1) / 2
            flows = {'xy': fxy, 'yx': fyx[:, [2, 1, 0], ...]}
        elif self.model == 'sym':
            [*_, fxy, fyx], _, _, X_Y, Y_X, grid, *_ = pred
            flows = {'xy': fxy[:, [2, 1, 0], ...], 'yx': fyx[:, [2, 1, 0], ...]}
        elif self.model == 'DIRAC':
            [F_X_Y, F_Y_X], X_Y, Y_X, *_ = pred
            if F_X_Y.shape[2:] != shape:
                F_X_Y = F.interpolate(F_X_Y, size=shape, mode='trilinear', align_corners=True)
            F_X_Y = F_X_Y[:, [2, 1, 0], ...]
            for i in range(3):
                F_X_Y[:, i, ...] = F_X_Y[:, i, ...] * (shape[i] - 1) / 2
            flows = {'xy': F_X_Y, 'yx': F_Y_X}
        elif self.model == 'NCRNet':
            v1, phi1, deformed1, segmentation, *_ = pred[0]
            flows = {'xy': phi1[:, [2, 1, 0], ...]}
        else:
            raise NotImplementedError
        # flows = flows['xy'].permute(0, 2, 3, 4, 1) + grid
        return landmark_mae_np(moving_landmark, fixed_landmark, flows['xy'])
        # if self.mode == 'x2y':
        #     flows = flows['xy'].permute(0, 2, 3, 4, 1) + grid
        #     mae = landmark_mae(moving_landmark, fixed_landmark, flows)
        # else:
        #     flows = flows['yx'].permute(0, 2, 3, 4, 1) + grid
        #     mae = landmark_mae(fixed_landmark, moving_landmark, flows)
        # return mae


class Dice(BaseObject):
    def __init__(self, region='all', device='cpu', fn='dice'):
        """
        :param region: (all, core) lesion region
        :param device:
        """
        super(Dice, self).__init__(tag='gen')
        self.device = torch.device(device)
        self.region = region
        self.dice = dice if fn == 'dice' else iou
        self._name = f'{fn}_{region}'

    def get_mask(self, gt):
        if self.region == 'all':
            return (gt >= 1).float()
        elif self.region == 'core':
            return (gt == 1).float() + (gt == 4).float()
        elif self.region == 'back':
            return (gt < 1).float()

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        mask, seg_gt = self.interface(pred, inp)
        mask = (mask > 0.5).float().to(self.device)
        # if self.region_growth:
        #     mask = self.region_growth(images=inp[0], pred_seg=mask)
        seg_gt = self.get_mask(seg_gt).to(self.device)
        return max(self.dice(seg_gt, mask),
                   self.dice(seg_gt, 1 - mask))

    def interface(self, pred, inp):
        mask, *_ = pred['gen']
        moving, fixed, seg_gt, *_ = inp
        return mask, seg_gt


class Dice5(Dice):
    def __init__(self, region, device='cpu'):
        super(Dice5, self).__init__(region, device)

    def interface(self, pred, inp):
        mask, *_ = pred['gen']
        _, _, seg_gt, *_ = inp
        return mask, seg_gt


class Sim(BaseObject):
    def __init__(self, tag, sim=None, mode='', device='cpu'):
        super(Sim, self).__init__(tag=tag, device=device)
        self.mode = mode
        self._name = f'{self.tag}_{self.mode}'
        self.sim = sim


class SimReg(Sim):
    def __init__(self, win, mode='m2f|f', device='cpu'):
        '''
        Args:
            mode: ff | mm | bidir
        '''
        super(SimReg, self).__init__(tag='reg', sim=NCC(win=win), mode=mode, device=device)

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        _, x2y, y2x = self.get_pred(pred)
        moving, fixed, seg_gt, *_ = inp
        if self.mode == 'm2f|f':
            return self.sim(x2y.to(self.device), fixed.to(self.device)).cpu()
        elif self.mode == 'f2m|m':
            return self.sim(y2x.to(self.device), moving.to(self.device)).cpu()
        else:
            raise ValueError


class SimInp(Sim):
    def __init__(self, sim, mode='back', device='cpu', mix=False):
        super(SimInp, self).__init__(tag='inp', sim=sim, mode=mode, device=device)
        self.mix = mix

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        mask2y, x2y, reg_inp_b, reg_inp_f = [i.to(self.device) if i is not None else None for i in self.get_pred(pred)]
        moving, fixed, seg_gt, *_ = [i.to(self.device) for i in inp]

        if self.mix:
            reg_inp_b_normal = x2y * (1 - mask2y) + fixed * mask2y
            reg_inp_f_normal = x2y * mask2y + fixed * (1 - mask2y)
        else:
            reg_inp_f_normal = reg_inp_b_normal = x2y
        if self.mode == 'back':
            return self.sim(reg_inp_f_normal.to(self.device), reg_inp_b.to(self.device)).cpu()
        elif self.mode == 'fore':
            return self.sim(reg_inp_b_normal.to(self.device), reg_inp_f.to(self.device)).cpu()
        else:
            raise ValueError


class SimInp5(Sim):
    def __init__(self, sim, mode='back', device='cpu', tag='inp'):
        super(SimInp5, self).__init__(tag=tag, sim=sim, mode=mode, device=device)

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        _, x2y, inp_b, inp_f = [i.to(self.device) if isinstance(i, torch.Tensor) else None for i in
                                self.get_pred(pred)]
        _, _, seg_gt, *_ = [i.to(self.device) for i in inp]
        if self.mode == 'back':
            return self.sim(x2y, inp_b)
        elif self.mode == 'fore':
            return self.sim(x2y, inp_f)
        else:
            raise ValueError


class SimInpDF(Sim):
    def __init__(self, sim, mode='back', device='cpu', tag='inp'):
        super(SimInpDF, self).__init__(tag=tag, sim=sim, mode=mode, device=device)

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        _, pos_flow, inpainted_b, inpainted_f = [i.to(self.device) if isinstance(i, torch.Tensor) else None for i in
                                                 self.get_pred(pred)]
        if self.mode == 'back':
            return self.sim(pos_flow, inpainted_b)
        elif self.mode == 'fore':
            return self.sim(pos_flow, inpainted_f)
        else:
            raise ValueError


class PreSimInp(Sim):
    def __init__(self, sim, mode='back', device='cpu'):
        super(PreSimInp, self).__init__(tag='inp', sim=sim, mode=mode, device=device)

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        reg_inp_b, reg_inp_f = [i.to(self.device) for i in self.get_pred(pred)]
        moving, fixed, seg_gt, *_ = [i.to(self.device) for i in inp]
        if self.mode == 'back':
            return self.sim(reg_inp_b.to(self.device), moving.to(self.device)).cpu()
        elif self.mode == 'fore':
            return self.sim(moving.to(self.device), reg_inp_f.to(self.device)).cpu()
        else:
            raise ValueError


class SwinSim(Sim):
    def __init__(self, device):
        super(SwinSim, self).__init__(tag='swin', mode='sim', sim=lambda x, y: torch.nn.L1Loss()(x, y), device=device)

    def forward(self, pred, inp):
        output_recons1, output_recons2 = self.get_pred(pred)[-2:]
        x1_augment, x2_augment, *_ = inp[-2:]
        output_recons = torch.cat([output_recons1, output_recons2], dim=0).to(self.device)
        target_recons = torch.cat([x1_augment, x2_augment], dim=0).to(self.device)
        return self.sim(output_recons, target_recons)


#
#
# class CoeConstraint(BaseObject):
#     def __init__(self, device='cpu', win=21, single=False, loss='mae', threshold=1):
#         '''
#         Args:
#             mode: ff | mm | bidir
#         '''
#         super(CoeConstraint, self).__init__(tag='gen', device=device)
#         self.single = single
#         self._name = f'CoeCons_single' if single else 'CoeCons'
#         self.sim = lambda x2y, mask2y, reg_inp_f, reg_inp_b: neg_coeff_constraint(
#             x2y, mask2y, reg_inp_f, reg_inp_b,
#             ncc_fn=NCC(win=win, threshold=threshold, reduction='none'),
#             single=single,
#         )
#
#     def forward(self, pred, inp):
#         if self.get_pred(pred) is None:
#             return torch.tensor(0)
#         mask, inpainted_b, inpainted_f = self.get_pred(pred)
#         mask[mask > 0.5] = 1
#         mask[mask < 0.5] = 0
#         moving, fixed, seg_gt, *_ = inp
#         if self.single:
#             return self.sim(moving.to(self.device), mask.to(self.device), inpainted_f.to(self.device),
#                             None).to('cpu')
#         else:
#             return self.sim(moving.to(self.device), mask.to(self.device), inpainted_f.to(self.device),
#                             inpainted_b.to(self.device)).to('cpu')


class Magnitude(BaseObject):
    def __init__(self, device='cpu', mode='back'):
        super(Magnitude, self).__init__(tag='inp', device=device)
        self.mag = magnitude_loss
        self.mode = mode
        self._name = f'mag_{mode}'

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        mask2y, x2y, reg_inp_b, reg_inp_f = [i.to(self.device) for i in self.get_pred(pred)]
        if self.mode == 'back':
            return self.mag(x2y[mask2y < 0.5], reg_inp_b[mask2y < 0.5])
        elif self.mode == 'fore':
            return self.mag(x2y[mask2y > 0.5], reg_inp_f[mask2y > 0.5])
        else:
            raise ValueError


class SymNCC(BaseObject):
    def __init__(self, win=5, eps=1e-5, device='cpu', mode='x2y', grad=False):
        super(SymNCC, self).__init__(tag='reg', device=device)
        if grad:
            self.ncc = lambda x, y: lncc(x, y, win=win, fast=True, threshold=eps) + \
                                    gradientSimilarity(x, y, win=win)
        else:
            self.ncc = lambda x, y: lncc(x, y, win=win, fast=True, threshold=eps, mask=None)
        self.mode = mode
        self._name = f'ncc_{mode}'

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        _, _, _, X_Y, Y_X, *_ = self.get_pred(pred)
        # if len(mask) == 0:
        #     mask = None
        # else:
        #     mask = mask[-1]
        # if mask is not None:
        #     mask = mask.to(self.device)
        X, Y, *_ = inp
        if self.mode == 'x2y':
            return self.ncc(X_Y.to(self.device), Y.to(self.device)).cpu()
        elif self.mode == 'y2x':
            return self.ncc(Y_X.to(self.device), X.to(self.device)).cpu()
        else:
            raise ValueError


class VoxNCC(BaseObject):
    def __init__(self, win=5, eps=1e-5, device='cpu', mode='x2y'):
        super(VoxNCC, self).__init__(tag='reg', device=device)
        self.ncc = lambda x, y: lncc(x, y, win=win, fast=False, threshold=eps, mask=None)
        self.mode = mode
        self._name = f'ncc_{mode}'

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        _, X_Y, Y_X, *_ = self.get_pred(pred)
        X, Y, *_ = inp
        if X_Y.shape[2:] != X.shape[2:]:
            X = F.interpolate(X, size=X_Y.shape[2:], mode='trilinear')
            Y = F.interpolate(Y, size=X_Y.shape[2:], mode='trilinear')

        if self.mode == 'x2y':
            return self.ncc(X_Y.to(self.device), Y.to(self.device))
        elif self.mode == 'y2x':
            return self.ncc(Y_X.to(self.device), X.to(self.device))
        else:
            raise ValueError


class ComSymNCC(BaseObject):
    def __init__(self, win, mode='x2y', device='cpu'):
        super(ComSymNCC, self).__init__(tag='reg', device=device)
        self.symncc = SymNCC(win=win, mode=mode)

    def forward(self, pred, inp):
        x, y, *_ = inp
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        reg1, reg2, mask, masky = self.get_pred(pred)
        _, _, _, X_Y1, Y_X1, _ = reg1
        _, _, _, X_Y2, Y_X2, _ = reg2
        return self.symncc({self.tag: [_, _, _, X_Y1 + X_Y2, Y_X1 + Y_X2, _]}, inp)


class ComSimInp(Sim):
    def __init__(self, sim, mode='back', device='cpu', tag='rec'):
        super(ComSimInp, self).__init__(tag=tag, sim=sim, mode=mode, device=device)

    def forward(self, pred, inp):
        if self.get_pred(pred) is None:
            return torch.tensor(0)
        _, x2y, inp_b, inp_f = [i.to(self.device) if isinstance(i, torch.Tensor) else i for i in
                                self.get_pred(pred)]
        x2y = x2y[-1].to(self.device)
        _, _, seg_gt = [i.to(self.device) for i in inp]
        if self.mode == 'back':
            return self.sim(x2y, inp_b)
        elif self.mode == 'fore':
            return self.sim(x2y, inp_f)
        else:
            raise ValueError


def window_sum(I, win_size):
    if win_size == 1:
        return I
    half_win = int(win_size / 2)
    pad = [half_win + win_size % 2, half_win] * 3
    I_padded = F.pad(I, pad=pad, mode='constant', value=0).double()  # [x+pad, y+pad, z+pad]
    # Run the cumulative sum across all 3 dimensions
    I_cs_x = torch.cumsum(I_padded, dim=2)
    I_cs_xy = torch.cumsum(I_cs_x, dim=3)
    I_cs_xyz = torch.cumsum(I_cs_xy, dim=4)
    x, y, z = I.shape[2:]
    # Use subtraction trick to calculate the window sum
    I_win = I_cs_xyz[:, :, win_size:, win_size:, win_size:] \
            - I_cs_xyz[:, :, win_size:, win_size:, :z] \
            - I_cs_xyz[:, :, win_size:, :y, win_size:] \
            - I_cs_xyz[:, :, :x, win_size:, win_size:] \
            + I_cs_xyz[:, :, win_size:, :y, :z] \
            + I_cs_xyz[:, :, :x, win_size:, :z] \
            + I_cs_xyz[:, :, :x, :y, win_size:] \
            - I_cs_xyz[:, :, :x, :y, :z]
    return I_win.float()


def gradient(I, add=False, same=False, abs=False):
    def scale(x):
        x = torch.abs(x)
        return (x - x.min()) / (x.max() - x.min())

    if same:
        I = F.pad(I, pad=[1, 0] * 3, mode='constant', value=0)
    dx = I[:, :, 1:, 1:, 1:] - I[:, :, :-1, 1:, 1:]
    dy = I[:, :, 1:, 1:, 1:] - I[:, :, 1:, :-1, 1:]
    dz = I[:, :, 1:, 1:, 1:] - I[:, :, 1:, 1:, :-1:]
    if abs:
        dx = scale(dx)
        dy = scale(dy)
        dz = scale(dz)
    if add:
        return dx + dy + dz
    return dx, dy, dz


def gradientSimilarity(I, J, win=5, eps=1e-2, reduction='mean'):
    def det(dx, dy, dz):
        return dx * dx + dy * dy + dz * dz + eps

    dxI, dyI, dzI = gradient(I, same=reduction != 'mean')
    dxJ, dyJ, dzJ = gradient(J, same=reduction != 'mean')

    dy_I = window_sum(dyI, win)
    dx_I = window_sum(dxI, win)
    dz_I = window_sum(dzI, win)
    dy_J = window_sum(dyJ, win)
    dx_J = window_sum(dxJ, win)
    dz_J = window_sum(dzJ, win)
    del dxI, dyI, dzI, dxJ, dyJ, dzJ

    cross = torch.abs(dx_I * dx_J + dy_I * dy_J + dz_I * dz_J) + eps
    norm = torch.sqrt(det(dx_I, dy_I, dz_I) * det(dx_J, dy_J, dz_J))

    if reduction != 'mean':
        return cross / norm
    return torch.mean(cross / norm)


def ncc2d(Ii, Ji, win=7, reduction='mean', threshold=0.01):
    wins = [win] * 2
    win_size = np.prod(wins)
    channel = Ii.shape[1]
    # compute filters
    sum_filt = torch.ones([1, channel, *wins], device=Ii.device, requires_grad=False)
    pad_no = math.floor(win / 2)
    stride = (1, 1)
    padding = (pad_no, pad_no)
    conv_fn = lambda x: F.conv2d(x, sum_filt, stride=stride, padding=padding)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii)
    J_sum = conv_fn(Ji)
    I2_sum = conv_fn(I2)
    J2_sum = conv_fn(J2)
    IJ_sum = conv_fn(IJ)
    del Ii, Ji, I2, J2, IJ
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = (cross * cross + threshold) / (I_var * J_var + threshold)

    if reduction == 'original':
        return cross, I_var, J_var

    if reduction == 'mean':
        return torch.mean(cc)
    else:
        return cc


def ncc(Ii, Ji, reduction='mean', threshold=0.1, mask=None):
    if mask is not None:
        Ii = Ii * mask
        Ji = Ji * mask

    cross = torch.square(torch.sum(Ii * Ji, dim=(1, 2, 3, 4)))
    I_var = torch.sum(Ii ** 2, dim=(1, 2, 3, 4))
    J_var = torch.sum(Ji ** 2, dim=(1, 2, 3, 4))
    cc = (cross + threshold) / (I_var * J_var + threshold)

    if reduction == 'original':
        return cross, I_var, J_var

    if reduction == 'mean':
        return torch.mean(cc)
    else:
        return cc


def lncc(Ii, Ji, win=7, reduction='mean', threshold=0.1, fast=True, mask=None, clip=None, sigma=0, kernel_size=None,
         grad=False):
    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    if sigma is not None and sigma != 0:
        Ii = gauss_smooth(Ii, sigma=sigma, kernel_size=kernel_size)
        Ji = gauss_smooth(Ji, sigma=sigma, kernel_size=kernel_size)
    if grad:
        Ii = gradient(Ii, add=True, same=True, abs=True)
        Ji = gradient(Ji, add=True, same=True, abs=True)

    wins = [win] * 3
    win_size = np.prod(wins)
    if not fast:
        # compute filters
        sum_filt = torch.ones([1, 1, *wins], device=Ii.device, requires_grad=False)
        pad_no = math.floor(win / 2)
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)
        conv_fn = lambda x: F.conv3d(x, sum_filt, stride=stride, padding=padding)
    else:
        conv_fn = lambda x: window_sum(x, win)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii)
    J_sum = conv_fn(Ji)
    I2_sum = conv_fn(I2)
    J2_sum = conv_fn(J2)
    IJ_sum = conv_fn(IJ)
    del Ii, Ji, I2, J2, IJ
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = (cross * cross + threshold) / (I_var * J_var + threshold)
    if mask is not None:
        cc = cc * (1 - mask)
    if clip:
        cc = torch.clip(cc, 0, clip) / clip
    if reduction == 'original':
        return cross, I_var, J_var

    if reduction == 'mean':
        return torch.mean(cc)
    else:
        return cc


class GradientSim(BaseObject):
    def __init__(self, win=7, reduction='mean', threshold=0.01, smooth=False):
        super(GradientSim, self).__init__()
        self.win = win
        self.reduction = reduction
        self.threshold = threshold
        self.grad = lambda x, y: gradientSimilarity(x, y, win=win, reduction=reduction, eps=threshold)
        self.smooth = None if not smooth else lambda x: gauss_smooth(x, sigma=2, kernel_size=7)

    def __call__(self, Ii, Ji):
        if self.smooth is not None:
            Ii = self.smooth(Ii)
            Ji = self.smooth(Ji)
        return self.grad(Ii, Ji)

    def __str__(self):
        return f'{self.__name__}: [win: {self.win}, reduction: {self.reduction}, threshold: {self.threshold}]'


class NCC(BaseObject):
    def __init__(self, win=7, reduction='mean', threshold=0.1, smooth=False, mask=False, local=True, clip=None):
        super(NCC, self).__init__()
        self.win = win
        self.reduction = reduction
        self.threshold = threshold
        self.clip = clip
        if local:
            if mask:
                self.ncc = lambda x, y, m: lncc(x, y, win=win, reduction=reduction, threshold=threshold, mask=m,
                                                clip=clip)
            else:
                self.ncc = lambda x, y, m: lncc(x, y, win=win, reduction=reduction, threshold=threshold, mask=None,
                                                clip=clip)
        else:
            if mask:
                self.ncc = lambda x, y, m: ncc(x, y, reduction=reduction, threshold=threshold, mask=m)
            else:
                self.ncc = lambda x, y, m: ncc(x, y, reduction=reduction, threshold=threshold, mask=None)

        self.smooth = None if not smooth else lambda x: gauss_smooth(x, sigma=2, kernel_size=7)

    def __call__(self, Ii, Ji, mask=None):
        if self.smooth is not None:
            Ii = self.smooth(Ii)
            Ji = self.smooth(Ji)
        return self.ncc(Ii, Ji, mask)

    def __str__(self):
        return f'{self.__name__}: [' \
               f'win: {self.win}, ' \
               f'reduction: {self.reduction},' \
               f' threshold: {self.threshold} ' \
               f'clip: {self.clip}' \
               f']'


class MultiNCC(BaseObject):
    def __init__(self, kernel_size=None, kernel_weight=None, scale=0, clip=0, reduction='mean', eps=1e-5, fast=False):
        super(MultiNCC, self).__init__()
        if kernel_size is None:
            kernel_size = [11, 15, 21]
        if kernel_weight is None:
            kernel_weight = [1 / len(kernel_size)] * len(kernel_size)

        ensure_tuple = lambda x: x if isinstance(x, (list, tuple)) else [x] * len(kernel_size)
        self.kernel_size = ensure_tuple(kernel_size)
        self.kernel_weight = ensure_tuple(kernel_weight)
        self.scale = ensure_tuple(scale)
        self.clip = ensure_tuple(clip)
        self.reduction = reduction
        self.eps = eps
        self.fast = fast

    def __call__(self, Ii, Ji, mask=None):
        """
        multi-kernel means lncc with different window sizes

        Loss = w1*lncc_win1 + w2*lncc_win2 ... + wn*lncc_winn, where /sum(wi) =1
        """
        multi_ncc = 0.
        for w, s, k, c in zip(self.kernel_weight, self.scale, self.kernel_size, self.clip):
            multi_ncc += w * lncc(Ii, Ji, win=k, reduction=self.reduction, threshold=self.eps,
                                  clip=c, sigma=s, fast=self.fast, mask=mask)
            Ii = nn.functional.avg_pool3d(Ii, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            Ji = nn.functional.avg_pool3d(Ji, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            if mask is not None:
                mask = nn.functional.avg_pool3d(mask, kernel_size=3, stride=2, padding=1,
                                                count_include_pad=False)
        return multi_ncc


def __str__(self):
    return f'{self.__name__}: [win: {self.kernel_size}, weight: {self.kernel_weight}, ' \
           f'scale: {self.scale} clip: {self.clip}]'


# class MultiScaleNCC(MultiKernelNCC):
#     def __init__(self, win=21, scale=(0, 1, 2, 4, 8), reduction='mean', threshold=0.1, clip=None):
#         if isinstance(win, int):
#             win = [win] * len(scale)
#         super(MultiScaleNCC, self).__init__(kernel_size=win, reduction=reduction, threshold=threshold, clip=clip)
#         self.scale = scale
#
#     def __call__(self, Ii, Ji, mask=None):
#         multi_loss = 0.
#         for s, fn in zip(self.scale, self.ncc):
#             multi_loss += fn(gauss_smooth(Ii, sigma=s), gauss_smooth(Ji, sigma=s), mask)
#         return multi_loss / len(self.scale)


def gauss_kernel1d(sigma, kernel_size, c):
    squared_dists = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size) ** 2
    gaussian_kernel = torch.exp(-0.5 * squared_dists / sigma ** 2)
    gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).view(1, 1, kernel_size).repeat(c, 1, 1)
    return gaussian_kernel


def gauss_smooth(x, sigma, kernel_size=None):
    if sigma == 0 or kernel_size == 0:
        return x
    else:
        if kernel_size is None:
            kernel_size = int(sigma * 6) + 1
        c = x.shape[1]
        padding = kernel_size // 2
        kernel = gauss_kernel1d(sigma, kernel_size, c).to(x.device)
        x = F.conv3d(x, kernel.unsqueeze(3).unsqueeze(4), padding=(padding, 0, 0), groups=c)
        x = F.conv3d(x, kernel.unsqueeze(2).unsqueeze(4), padding=(0, padding, 0), groups=c)
        x = F.conv3d(x, kernel.unsqueeze(2).unsqueeze(3), padding=(0, 0, padding), groups=c)
        del kernel
        return x


def smooth_l1(x, y):
    return nn.SmoothL1Loss(reduction='none')(x, y)


def neg_Jdet_count(field):
    neg_Jdet = -1 * JacboianDet(field)
    neg_Jdet[neg_Jdet < 0] = 0
    neg_Jdet[neg_Jdet > 0] = 1
    return neg_Jdet.sum()


def JacboianDet(field):
    J = field
    dx = J[:, :, 1:, :-1, :-1] - J[:, :, :-1, :-1, :-1]
    dy = J[:, :, :-1, 1:, :-1] - J[:, :, :-1, :-1, :-1]
    dz = J[:, :, :-1, :-1, 1:] - J[:, :, :-1, :-1, :-1]

    Jdet0 = dx[:, 0, :, :, :] * (dy[:, 1, :, :, :] * dz[:, 2, :, :, :] - dy[:, 2, :, :, :] * dz[:, 1, :, :, :])
    Jdet1 = dx[:, 1, :, :, :] * (dy[:, 0, :, :, :] * dz[:, 2, :, :, :] - dy[:, 2, :, :, :] * dz[:, 0, :, :, :])
    Jdet2 = dx[:, 2, :, :, :] * (dy[:, 0, :, :, :] * dz[:, 1, :, :, :] - dy[:, 1, :, :, :] * dz[:, 0, :, :, :])

    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def neg_Jdet_loss(field):
    neg_Jdet = -1.0 * JacboianDet(field)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def magnitude_loss(flow_1, flow_2, slices=None, reduction='mean'):
    if slices is not None:
        num_ele = torch.numel(flow_1[slices])
    else:
        num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))
    if reduction == 'mean':
        diff = (torch.abs(flow_1_mag - flow_2_mag)) / (num_ele + 1)
    else:
        diff = torch.abs(flow_1_mag - flow_2_mag)
    return diff


def gram_matrix(x):
    (b, c, h, w, d) = x.size()
    features = x.view(b, c, w * h * d)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w * d)
    return gram


def accuracy(pr, gt, threshold=0.5):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = (pr > threshold).float()
    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


class GANMetrics(BaseObject):
    def __init__(self, gen=True, w=(5, 100)):
        super(GANMetrics, self).__init__()
        self.gen = gen
        self._name = 'grad_mae' if gen else 'disc_acc'
        self.tag = 'gen' if gen else 'inp'
        self.w = w
        self.grad = lambda x, y: gradientSimilarity(x, y, 1)
        self.acc = lambda x, y: accuracy(x, y)

    def forward(self, pre, inp):
        moving, y, *_ = inp
        if self.gen:
            gen, pred_label = self.get_pred(pre)
            return self.w[0] * self.grad(gen, y) - self.w[1] * torch.abs(gen - moving).mean()
        else:
            _, _, pred_label, gt_label = self.get_pred(pre)
            return self.acc(pred_label, gt_label)


class SegGANMetrics(BaseObject):
    def __init__(self, gen=True):
        super(SegGANMetrics, self).__init__()
        self.gen = gen
        self._name = 'grad_mae' if gen else 'disc_acc'
        self.tag = 'gen' if gen else 'inp'
        self.acc = lambda x, y: accuracy(x, y)

    def forward(self, pre, inp):
        moving, y, seg, *_ = inp
        if self.gen:
            gen, pred_label = self.get_pred(pre)
            gen = (gen > 0.5).float()
            seg = (seg > 0.5).float()
            return dice(gen, seg)
        else:
            _, _, pred_label, gt_label = self.get_pred(pre)
            return self.acc(pred_label, gt_label)


class MeanRegError(BaseObject):
    def __init__(self, pos='in', device='cpu'):
        super(MeanRegError, self).__init__(tag='reg', device=device)
        self.pos = pos
        self._name = f'reg_error_{pos}'

    def forward(self, pred, inp):
        pass


class HistogramMatching(nn.Module):
    def __init__(self, differentiable=False, rescale=False):
        super(HistogramMatching, self).__init__()
        self.differentiable = differentiable
        self.rescale = rescale

    def forward(self, dst, ref):
        # B C
        B, C, *_ = dst.size()
        # assertion
        assert dst.device == ref.device
        # [B*C 256]
        hist_dst = self.cal_hist(dst)
        hist_ref = self.cal_hist(ref)
        # [B*C 256]
        tables = self.cal_trans_batch(hist_dst, hist_ref)
        # [B C H W]
        rst = dst.clone()
        for b in range(B):
            for c in range(C):
                rst[b, c] = tables[b * c, (dst[b, c] * 255).long()]
        # [B C H W]
        if self.rescale:
            rst = (rst - rst.min()) / (rst.max() - rst.min())
        else:
            rst /= 255.
        return rst

    def cal_hist(self, img):
        B, C, *_ = img.size()
        # [B*C 256]
        if self.differentiable:
            hists = self.soft_histc_batch(img * 255, bins=256, min=0, max=256, sigma=3 * 25)
        else:
            hists = torch.stack(
                [torch.histc(img[b, c] * 255, bins=256, min=0, max=255) for b in range(B) for c in range(C)])
        hists = hists.float()
        hists = F.normalize(hists, p=1)
        # BC 256
        bc, n = hists.size()
        # [B*C 256 256]
        triu = torch.ones(bc, n, n, device=hists.device).triu()
        # [B*C 256]
        hists = torch.bmm(hists[:, None, :], triu)[:, 0, :]
        return hists

    def soft_histc_batch(self, x, bins=256, min=0, max=256, sigma=3 * 25):
        # B C H W
        B, C, *_ = x.size()
        # [B*C H*W]
        x = x.view(B * C, -1)
        # 1
        delta = float(max - min) / float(bins)
        # [256]
        centers = float(min) + delta * (torch.arange(bins, device=x.device, dtype=torch.bfloat16) + 0.5)
        # [B*C 1 H*W]
        x = torch.unsqueeze(x, 1)
        # [1 256 1]
        centers = centers[None, :, None]
        # [B*C 256 H*W]
        x = x - centers
        # [B*C 256 H*W]
        x = x.type(torch.bfloat16)
        # [B*C 256 H*W]
        x = torch.sigmoid(sigma * (x + delta / 2)) - torch.sigmoid(sigma * (x - delta / 2))
        # [B*C 256]
        x = x.sum(dim=2)
        # [B*C 256]
        x = x.type(torch.float32)
        # torch.cuda.empty_cache()
        return x

    def cal_trans_batch(self, hist_dst, hist_ref):
        hist_dst = hist_dst[:, None, :].repeat(1, 256, 1)
        hist_ref = hist_ref[:, :, None].repeat(1, 1, 256)
        table = hist_dst - hist_ref
        table = torch.where(table >= 0, 1., 0.)
        table = torch.sum(table, dim=1) - 1
        table = torch.clamp(table, min=0, max=255)
        return table
