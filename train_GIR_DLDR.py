# -*- encoding: utf-8 -*-
"""
@File    :   train_GIRNet_MIxInpMSE.py
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/6 9:12 PM   liu.yang      1.0         None
"""
import os
import torch.optim.lr_scheduler

from monai.utils import set_determinism
from utils.metrics import NCC, MultiNCC, LandMarkMAE, SimInp5
from utils.loss_functions import *
from params import *
from utils.train_base import train_fold, MultiDict
import warnings

warnings.filterwarnings("ignore")
from model.GIRNet import GIRNetPseudo
from DIRAC.Code.Functions import generate_grid_unit


class DLDR(nn.Module):
    def __init__(self, imgshape=(128, 160, 112)):
        super().__init__()
        self.imgshape = imgshape
        imgshape_4 = (imgshape[0] // 4, imgshape[1] // 4, imgshape[2] // 4)
        imgshape_2 = (imgshape[0] // 2, imgshape[1] // 2, imgshape[2] // 2)
        from DIRAC.Code.bratsreg_model_stage import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, \
            Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3
        model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(2, 3, 6, is_train=True,
                                                                       imgshape=imgshape_4,
                                                                       range_flow=.4,
                                                                       num_block=5)
        model_lvl2 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(2, 3, 6, is_train=True,
                                                                       imgshape=imgshape_2,
                                                                       range_flow=.4, model_lvl1=model_lvl1,
                                                                       num_block=5)

        self.model = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(2, 3, 6, is_train=True,
                                                                       imgshape=imgshape,
                                                                       range_flow=.4, model_lvl2=model_lvl2,
                                                                       num_block=5)
        self.transform = self.model.transform
        self.grid_0 = generate_grid_unit((160, 192, 144))
        self.grid_0 = torch.from_numpy(np.reshape(self.grid_0, (1,) + self.grid_0.shape)).float()

    def grid(self):
        return self.model.grid_1

    def forward(self, X_o, Y_o, train=True, Moving=None, Fixed=None):
        shape = X_o.shape[2:]
        if Moving is not None:
            X_o = Moving
        if Fixed is not None:
            Y_o = Fixed
        if self.imgshape != shape:
            X = F.interpolate(X_o, size=self.imgshape, mode='trilinear')
            Y = F.interpolate(Y_o, size=self.imgshape, mode='trilinear')
        else:
            X = X_o
            Y = Y_o
        if train == False:
            reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
            F_X_Y = self.model(X, Y, reg_code, is_train=False)
            F_Y_X = self.model(Y, X, reg_code, is_train=False)
            if self.imgshape != shape:
                F_X_Y = F.interpolate(F_X_Y, size=shape, mode='trilinear')
                F_Y_X = F.interpolate(F_Y_X, size=shape, mode='trilinear')
            X_Y = self.transform(X_o, F_X_Y.permute(0, 2, 3, 4, 1), self.grid_0.to(X.device))
            Y_X = self.transform(Y_o, F_Y_X.permute(0, 2, 3, 4, 1), self.grid_0.to(X.device))
            for i in range(3):
                F_X_Y[:, i, ...] = F_X_Y[:, i, ...] * (shape[-i - 1] - 1) / 2
                F_Y_X[:, i, ...] = F_Y_X[:, i, ...] * (shape[-i - 1] - 1) / 2
            return [F_X_Y, F_Y_X], X_Y, Y_X
        else:
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)
            F_X_Y, X_Y, F_xy, F_xy_lvl1, F_xy_lvl2, _ = self.model(X, Y, reg_code)
            F_Y_X, Y_X, F_yx, F_yx_lvl1, F_yx_lvl2, _ = self.model(Y, X, reg_code)

        return [F_X_Y, F_Y_X], X_Y, Y_X, reg_code


class GIDLDR(GIRNetPseudo):
    def __init__(self, input_shape,
                 nb_features,
                 int_steps,
                 single,
                 norm,
                 guide_by_rec,
                 gt_seg,
                 ntuc,
                 conv_num,
                 add_flow_gen,
                 fore_inp,
                 back_inp):
        super().__init__(
            input_shape=input_shape,
            nb_features=nb_features,
            int_steps=int_steps,
            single=single,
            norm=norm,
            guide_by_rec=guide_by_rec,
            gt_seg=gt_seg,
            ntuc=ntuc,
            conv_num=conv_num,
            add_flow_gen=add_flow_gen,
            fore_inp=fore_inp,
            back_inp=back_inp
        )
        self.model_id = 'GIDLDR'
        self.symnet = DLDR()


class RegLoss(BaseObject):
    def __init__(self, occ=.01, inv_con=.01, eps=1e-5):
        super(RegLoss, self).__init__()
        from DIRAC.Code.bratsreg_model_stage import multi_resolution_NCC, smoothloss
        self.loss_similarity = multi_resolution_NCC(win=7, scale=3, eps=eps)
        self.loss_smooth = smoothloss
        self.occ = occ
        self.inv_con = inv_con

    def forward(self, pred, inp):
        # 3 level deep supervision NCC
        X, Y = inp[0], inp[1]
        [F_X_Y, F_Y_X], X_Y, Y_X, reg_code, mask = pred
        if X_Y.shape[2:] != X.shape[2:]:
            X = F.interpolate(X, size=X_Y.shape[2:], mode='trilinear')
            Y = F.interpolate(Y, size=X_Y.shape[2:], mode='trilinear')
        loss_multiNCC = self.loss_similarity(X_Y, Y) + self.loss_similarity(Y_X, X)

        # reg2 - use velocity
        _, _, x, y, z = F_X_Y.shape
        norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
        norm_vector[0, 0, 0, 0, 0] = z
        norm_vector[0, 1, 0, 0, 0] = y
        norm_vector[0, 2, 0, 0, 0] = x
        loss_regulation = self.loss_smooth(F_X_Y * norm_vector) + self.loss_smooth(F_Y_X * norm_vector)
        loss = (1. - reg_code) * loss_multiNCC + reg_code * loss_regulation
        return loss


def get_model_opti(args):
    if 'Landmark' in args.name:
        args.metrics = [
            LandMarkMAE(model='DIRAC'),
            SimInp5(sim=args.ncc_fn_mean, mode='fore', device=torch.device(args.device)),
        ]
        args.metrics_tags_map = {
            args.metrics[0].__name__: 'reg',
            args.metrics[1].__name__: 'inp',
        }
        args.modes = {
            'reg': 'min',
            'gen': 'max',
            'inp': 'min'
        }
    else:
        from utils.metrics import Dice, VoxNCC
        args.metrics = [
            VoxNCC(win=7),
            Dice(device=args.device),
            SimInp5(sim=args.ncc_fn_mean, mode='fore', device=torch.device(args.device)),
        ]
        args.metrics_tags_map = {
            args.metrics[0].__name__: 'reg',
            args.metrics[1].__name__: 'gen',
            args.metrics[2].__name__: 'inp',
        }
        args.modes = {
            'reg': 'max',
            'gen': 'max',
            'inp': 'min'
        }

    model = GIDLDR(
        input_shape=img_size[args.task],
        nb_features=net_config[args.task],
        int_steps=args.int_steps,
        single=args.single,
        norm=args.norm,
        guide_by_rec=args.guide_by_rec,
        gt_seg=args.gt_seg,
        ntuc=args.ntuc,
        conv_num=args.conv_num,
        add_flow_gen=args.add_flow_gen,
        fore_inp=args.fore_inp,
        back_inp=args.back_inp
    )
    losses = [
        GenLoss(fore_fn=args.fore_fn, back_fn=args.back_fn, fore_eps=args.iem_eps, back_eps=args.iem_eps,
                single=args.single, gt_seg=args.gt_seg, reduction=args.reduction, back_weight=1),
        InpLoss2(ncc_fn=args.ncc_fn_mean, weight=args.weight, excludes_size_influence=args.exclude,
                 merge=False, single=args.single, histmatch=args.histmatch, rescale=False),
        RegLoss(eps=args.ncc_eps)
    ]
    opt = MultiDict({
        'gen': torch.optim.AdamW(params=model.mask_generator.parameters(), lr=args.lr, weight_decay=0.01),
        'inp': torch.optim.AdamW(params=model.inpainter.parameters(), lr=args.lr, weight_decay=0.01),
        'reg': torch.optim.AdamW(params=model.symnet.parameters(), lr=args.lr, weight_decay=0.01)
    })
    return model, opt, losses


def main(fold=0):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    set_determinism(42)
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    args.dir_name = 'GIDLDR'
    args.data_dir = 'data_json/train_BraTSPseudoHistoNLin'
    # tags that control training
    # reg == registration   gen == mask generator   inp == in-painter
    args.train_tags = {'reg': True, 'gen': True, 'inp': True}
    args.epochs = {'reg': args.epoch, 'inp': args.epoch, 'gen': args.epoch}
    args.k = {'gen': 1, 'inp': 1, 'reg': 1}
    # pretrain one model before training
    args.pretrain = {'reg': 0}
    # Train {gen 3, inp 1} in one iteration
    args.loop_flag = False
    args.norm = False
    args.conv_num = 1
    args.guide_by_rec = True
    args.gt_seg = False
    args.exclude = True
    args.ntuc = False
    args.add_flow_gen = False
    args.histmatch = False

    args.ncc_fn_mean = NCC(win=args.win, threshold=args.threshold, reduction='mean')
    args.fore_fn = NCC(win=args.win, threshold=args.threshold, reduction='none', clip=args.clip)
    args.fore_inp = True
    args.back_fn = NCC(win=args.win, threshold=args.threshold, reduction='none', clip=args.clip)
    args.back_inp = True
    args.reduction = 'exclude'

    args.y2x_rec = False
    args.sym = True

    if 'BraTS' in args.name:
        if 'PseudoHisto' in args.name:
            args.data_dir = 'data_json/train_BraTSPseudoHistoNLin'
        else:
            args.data_dir = 'data_json/train_BraTSNLin'
    elif 'Stroke' in args.name:
        if 'Pseudo' in args.name:
            args.data_dir = 'data_json/train_StrokePseudoNLin'
        elif '100' in args.name:
            args.data_dir = 'data_json/train_StrokeNLin100'
        else:
            args.data_dir = 'data_json/train_StrokeNLin140'

    if 'Landmark' in args.name:
        if 'RAS' in args.name:
            args.data_dir = 'data_json/train_BraTSRegLandmarkRAS'
            args.load_data_keys = 'landmark_ras'
            args.task = 'landmark_ras'
        else:
            args.data_dir = 'data_json/train_BraTSRegLandmarkNLin'
            args.load_data_keys = 'landmark'
            args.task = 'landmark'

        args.guide_by_rec = True
        args.gt_seg = False
        args.best_tag = 'reg'

    if 'Stroke' == args.task or 'NLin' == args.task:
        args.weight = (1, 100, 0)

    if 'normal' in args.mark or 'NoSeg' in args.mark:
        args.train_tags = {'reg': True, 'gen': False, 'inp': False}
        args.load_data_keys = 'normal'
        args.guide_by_rec = False
        args.gt_seg = False

    if 'BCESeg' in args.mark:
        args.weight = (0, 1, 0)
        args.gt_seg = True

    if 'mask' in args.mark:
        args.mask = True
        args.guide_by_rec = True
    else:
        args.mask = False

    if 'noexclude' in args.mark:
        args.exclude = False

    if 'exclude212' in args.mark:
        args.k = {'gen': 2, 'inp': 1, 'reg': 2}

    if 'exclude211' in args.mark:
        args.k = {'gen': 2, 'inp': 1, 'reg': 1}

    if 'exclude221' in args.mark:
        args.k = {'gen': 2, 'inp': 2, 'reg': 1}

    if 'Lesion' in args.mark:
        args.train_tags = {'reg': True, 'gen': True, 'inp': True}
        args.guide_by_rec = False
        args.gt_seg = False

    if 'ntuc' in args.mark:
        args.ntuc = True

    if 'norm' in args.mark:
        args.norm = True
        # args.fore_eps = 100
        # args.back_eps = 100

    if 'add_flow' in args.mark:
        args.add_flow_gen = True

    if 'backncc' in args.mark:
        args.back_inp = False
        args.back_fn = NCC(win=args.win, threshold=args.threshold, reduction='none', clip=args.clip)
    if 'RegMulti' in args.mark:
        args.multi = True
    else:
        args.multi = False
    if 'SegMulti' in args.mark:
        args.kernel_size = (9, 15, 21)
        args.kernel_weight = (.3, .4, .3)
        args.scale = 0
        args.clip = None
        args.fore_fn = MultiNCC(kernel_weight=args.kernel_weight, kernel_size=args.kernel_size, scale=args.scale,
                                reduction='none', clip=args.clip)
        args.back_fn = MultiNCC(kernel_weight=args.kernel_weight, kernel_size=args.kernel_size, scale=args.scale,
                                reduction='none', clip=args.clip)
    if 'hist' in args.mark:
        args.histmatch = True
        args.weight = (1, 100, 0)

    if 'reductionsum' in args.mark:
        args.reduction = 'sum'

    if 'y2xrec' in args.mark:
        args.y2x_rec = True
    if 'nonsym' in args.mark:
        args.sym = False

    print(args)
    print(all_params(args.task))

    model, opti, args.losses = get_model_opti(args)
    train_fold(args, fold, model, opti)


if __name__ == "__main__":
    args = get_args()
    setproctitle.setproctitle(
        '/home/dzx/.conda/envs/zzp_env/bin/python3.8')
    # enabling cudnn determinism appears to speed up training by a lot
    for i in range(args.cross_validation_start, args.cross_validation_end):
        main(i)
