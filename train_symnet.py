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
from utils.metrics import LandMarkMAE, SymNCC
from utils.loss_functions import SymRegLoss
from params import *
from utils.train_base import train_fold, MultiDict
import warnings

warnings.filterwarnings("ignore")


def get_model_opti(args):
    if 'landmark' in args.load_data_keys:
        args.metrics = [
            LandMarkMAE(model='sym'),
            SymNCC(win=7),
        ]
        args.metrics_tags_map = {
            args.metrics[0].__name__: 'reg',
        }
        args.modes = {'reg': 'min'}
    else:
        args.metrics = [
            SymNCC(win=7)
        ]
        args.metrics_tags_map = {
            args.metrics[0].__name__: 'reg',
        }
        args.modes = {'reg': 'max'}
    from model.SymNet import SymnetSingle
    model = SymnetSingle(
        img_shape=img_size[args.task],
        gt=args.gt,
        HM=args.histo_matching,
    )
    losses = SymRegLoss(win=7, eps=1e-5, sym=True, y2x_rec=False, smooth=args.smooth)
    opt = MultiDict({'reg': torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.01)})
    return model, opt, losses


def main(fold=0):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    set_determinism(42)
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    args.dir_name = 'Symnet'
    args.data_dir = 'data_json/train_BraTSPseudoHistoNLin'
    args.train_tags = {'reg': True}
    args.epochs = {'reg': args.epoch}
    args.k = {'reg': 1}
    # Train {gen 3, inp 1} in one iteration
    args.loop_flag = False
    args.best_tag = 'reg'
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

    if 'normal' in args.mark or 'NoSeg' in args.mark:
        args.load_data_keys = 'normal'

    if 'GT' in args.mark:
        args.gt = True
    else:
        args.gt = False

    if "HistoMatching" in args.mark:
        args.histo_matching = True
    else:
        args.histo_matching = False

    print(args)
    print(all_params(args.task))

    model, opti, args.losses = get_model_opti(args)
    train_fold(args, fold, model, opti)


if __name__ == "__main__":
    args = get_args()
    # enabling cudnn determinism appears to speed up training by a lot
    for i in range(args.cross_validation_start, args.cross_validation_end):
        main(i)
