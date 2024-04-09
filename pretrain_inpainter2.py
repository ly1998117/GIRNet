# -*- encoding: utf-8 -*-
"""
@File    :   IEM.py
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/21 4:48 PM   liu.yang      1.0         None
"""
import argparse, time, torch
import glob
import os

from monai.utils import set_determinism

from utils.dataloader import get_loader
from model.MaskGenerator import RandMaskG
from visualize.slices_plot import plot_img3d
from model.Inpainter import Inpainter2, Inpainter
from model.SymNet import SymNet


def dice(pr, gt, eps=1e-7):
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + fp + eps)
    return score


def inp_loss(x, atlas, mask, pred_foreground, pred_background):
    # ncc_loss = - ncc(x, pred_foreground, win=win) - ncc(x, pred_background, win=win)

    mse_loss = torch.nn.MSELoss()(x * (1 - mask) + atlas * mask, pred_foreground) + \
               torch.nn.MSELoss()(x * mask + atlas * (1 - mask), pred_background)
    return mse_loss


def inp_loss_single(x, y2x, mask, pred_foreground):
    mse_loss = torch.nn.MSELoss()(x * (1 - mask) + y2x * mask, pred_foreground)
    return mse_loss


def epoch(train, dataloader, e):
    tag = 'train' if train else 'valid'
    inpainter.train()
    sym.eval()
    for batch_idx, data in enumerate(dataloader):
        print(f"Epoch {e}/{args.epoch}   Batch {batch_idx + 1}/{len(dataloader)}")
        f_name = data[f'moving_t1_meta_dict']['filename_or_obj'][0].split('/')[-2]
        x, atlas = data['moving_t1'].to(args.device), data['atlas'].to(args.device)
        with torch.no_grad():
            y = sym(atlas, x)[-2]
            atlas = atlas.cpu()

        with torch.no_grad():
            mask = (maskg(x) > 0.5).float()
        if train:
            pred_foreground = inpainter(x * (1 - mask), mask, y)
            loss = inp_loss_single(x, y, mask, pred_foreground)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pred_background = inpainter(x * mask, 1 - mask, y)
            loss = inp_loss_single(x, y, 1 - mask, pred_background)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # else:
            #     with torch.no_grad():
            #         inpainter.eval()
            #         pred_foreground = inpainter(background, (mask > 0.5).float(), y)
            #         pred_background = inpainter(foreground, (mask < 0.5).float(), y)
            #         loss = None

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    x = x.cpu()
                    y = y.cpu()
                    mask = mask.cpu()
                    pred_background = pred_background.cpu()
                    pred_foreground = pred_foreground.cpu()
                    images = [x, atlas, y, mask, pred_foreground, pred_background]
                    title = ['moving', 'atlas', 'y2x', 'mask', 'pred_foreground',
                             'pred_background']
                    plot_img3d(images, title, rows=5, width=50, batch_id=e, iter=batch_idx,
                               name=os.path.join(dir_name, tag + f_name), fontsize=35)


def train(resume=False):
    start_time = time.time()
    if resume:
        sym.load_state_dict(
            torch.load('/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/checkpoint/reg.pth', map_location='cpu')[
                'state_dict'], strict=False)
        try:
            name = sorted(glob.glob(f'plotIEM/checkpoint/{args.name}/*'))[-1]
            inpainter.load_state_dict(torch.load(name, map_location='cpu'))
            print(name)
        except Exception:
            pass
    inpainter.to(device=args.device)
    sym.to(device=args.device)
    for e in range(args.epoch):
        epoch(True, loader[0], e)
        # epoch(False, valid_data, e, maskg, inpainter, optim1, optim2)
        os.makedirs(f'plotIEM/checkpoint/{args.name}', exist_ok=True)

        torch.save(inpainter.state_dict(), f'plotIEM/checkpoint/{args.name}/{e}.pth')

    end_time = time.time()
    print("IEM finished in {:.1f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inpainting Error Maximization')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--plot_step', type=int, default=1)
    parser.add_argument('--device', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--name', type=str, default='inp2_num2')
    args = parser.parse_args()
    print(args.device)
    set_determinism(41)
    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    dir_name = args.name
    loader = get_loader(task='NLin', load_data_keys=('moving_t1', 'atlas', 'moving_seg'),
                        data_dir='data_json/train_OASISNLin', k_fold=0,
                        batch_size=1, num_worker=0)
    # pretrain(False)
    maskg = RandMaskG(mask_scale=(0.1, 0.8))
    inpainter = Inpainter2(shape=[160, 192, 144],
                           input_channel=[2, 1],
                           nb_features=[[16, 32, 64, 64],
                                        [64, 64, 32, 32, 32, 16, 16]],
                           norm=False,
                           conv_num=2)

    sym = SymNet(img_shape=[160, 192, 144])
    optim = torch.optim.Adam(inpainter.parameters(), lr=0.0001)

    train(True)
