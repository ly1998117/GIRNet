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
from utils.metrics import *
from utils.loss_functions import *
from params import *
from utils.train_base import train_fold, MultiDict
import warnings

warnings.filterwarnings("ignore")


def get_model_opti(args):
    if 'landmark' in args.load_data_keys:
        args.metrics = [
            LandMarkMAE(),
            SimInp5(sim=args.ncc_fn_mean, mode='fore', device=torch.device(args.device)),
            SymNCC(win=7),
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
        args.metrics = [
            Dice('all', device=torch.device(args.device)),
            SimInp5(sim=args.ncc_fn_mean, mode='fore', device=torch.device(args.device)),
            SymNCC(win=7),
        ]
        args.metrics_tags_map = {
            args.metrics[0].__name__: 'gen',
            args.metrics[-2].__name__: 'inp',
            args.metrics[-1].__name__: 'reg',
        }
        args.modes = {
            'reg': 'max',
            'gen': 'max',
            'inp': 'min'
        }
    from model.GIRNet import GIRNetPseudo
    model = GIRNetPseudo(
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
        SymRegLoss(win=7, eps=args.ncc_eps, sym=args.sym, y2x_rec=args.y2x_rec, smooth=args.smooth,
                   multi=args.multi, mask=args.mask, local_ori=1)
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

    args.dir_name = 'GIRNet5'
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
    # enabling cudnn determinism appears to speed up training by a lot
    for i in range(args.cross_validation_start, args.cross_validation_end):
        main(i)
