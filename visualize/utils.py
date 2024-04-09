# -*- encoding: utf-8 -*-
"""
@File    :   utils.py   
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/14 9:26 PM   liu.yang      1.0         None
"""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import seaborn as sns
from .slices_plot import plot_slices
from model.SymNet import generate_grid
from utils.metrics import gradientSimilarity
from utils.utils import make_dirs


class Plot:
    def __init__(self, model=6, ncc_fn=None, dir_path='result', slices_n=8, save_img=True, epoch=0, dpi=300,
                 im_format='png'):
        self.model = model
        self.dir_path = dir_path
        self.slices_n = slices_n
        self.times = epoch
        make_dirs(dir_path)
        self.logs = []
        self.ncc = ncc_fn
        self.save_img = save_img
        self.dpi = dpi
        self.format = im_format

    def get_pred(self, inp, pred):
        keys = [k for k, v in pred.items() if v is not None]
        images = []
        ncc_map = HB = HF = mask = mask2y = inpainted_b = inpainted_f = reg_inp_f = reg_inp_b = pos_flow = x2y = y2x = field = flow = None

        if self.model == 0:
            output_rot1, output_rot2, output_contrastive, target_contrastive, output_recons1, output_recons2 = pred[
                'swin']
            x1, x2, _, _, x1_augment, x2_augment = inp
            images = [x1, x2, x1_augment, x2_augment, output_recons1, output_recons2]
            titles = ['x1', 'x2', 'x1_augment', 'x2_augment', 'output_recons1', 'output_recons2']

        if self.model == 'DIRAC':
            [pos_flow, _], x2y, y2x, \
                norm_diff_fw, norm_diff_bw, occ_xy, occ_yx, mask_xy, mask_yx, reg_code = pred['reg']
            pos_flow = pos_flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
            for i in range(3):
                pos_flow[..., i] = (pos_flow[..., i]) * x2y.shape[2:][i] / 2
            grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()[..., [2, 1, 0]]
            field = pos_flow + grid
            images.extend([inp[0], inp[1], x2y, y2x,
                           pos_flow, field, norm_diff_fw, mask_xy, mask_yx])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving',
                      'florgb', 'contour', 'norm_diff_fw_rgb', 'mask_xy', 'mask_yx']

        if self.model == 'NCRNet':
            v1, pos_flow, x2y, y2x, _, mask = pred['reg'][0]
            pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
            grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()
            field = (pos_flow + grid)[..., [2, 1, 0]]
            images.extend([inp[0], inp[1], x2y, y2x, pos_flow, field, inp[-1], mask])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour', 'seg_gt', 'mask']

        if self.model == 'DLDR':
            [pos_flow, _], x2y, y2x, *_ = pred['reg']
            pos_flow = pos_flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
            for i in range(3):
                pos_flow[..., i] = (pos_flow[..., i]) * x2y.shape[2:][i] / 2
            grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()[..., [2, 1, 0]]
            field = pos_flow + grid
            images.extend([inp[0], inp[1], x2y, y2x, pos_flow, field])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour']

        if self.model == 'GIDLDR':
            if 'inp' in keys:
                mask, y2x, inpainted_b, inpainted_f = pred['inp']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

            if 'rec' in keys:
                mask, x2y, inpainted_b, inpainted_f = pred['rec']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

            if 'gen' in keys:
                mask, inpainted_b, inpainted_f, x2y, *_ = pred['gen']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
            inp2y = None
            if 'reg' in keys:
                [pos_flow, _], inp2y, y2x, *_ = pred['reg']
                pos_flow = pos_flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
                for i in range(3):
                    pos_flow[..., i] = (pos_flow[..., i]) * inp2y.shape[2:][i] / 2
                grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()[..., [2, 1, 0]]
                field = pos_flow + grid
            images.extend(
                [inp[0], inp[1], x2y, inp2y, y2x, pos_flow, field, inp[-1],
                 mask, inpainted_b, inpainted_f])
            titles = [f'moving', f'atlas', 'moving2fixed', 'inp2fixed', 'fixed2moving', 'field', 'contour', 'seg_gt',
                      'seg_pred', 'inpainted_b', 'inpainted_f']

        if self.model == 'DIFVox':
            [pos_flow, *_], x2y, y2x, *_ = pred['reg']
            pos_flow = pos_flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
            # for i in range(3):
            #     pos_flow[..., i] = (pos_flow[..., i]) * x2y.shape[2:][i] / 2
            grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()[..., [2, 1, 0]]
            field = pos_flow + grid
            images.extend([inp[0], inp[1], x2y, y2x, pos_flow, field])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour']

        if self.model == 1:
            # Voxelmorph
            if 'reg' in keys:
                [pos_flow, *_], x2y, y2x, *_ = pred['reg']
                pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
            grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()
            field = (pos_flow + grid)[..., [2, 1, 0]]
            images.extend([inp[0], inp[1], x2y, y2x, pos_flow, field])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour']

        if self.model == 2:
            if 'gen' in keys:
                mask, ncc_map, inpainted_b, inpainted_f = pred['gen']
            if 'inp' in keys:
                mask, ncc_map, inpainted_b, inpainted_f = pred['inp']
            if 'reg' in keys:
                try:
                    [pos_flow, _], x2y, y2x = pred['reg']
                    pos_flow = pos_flow * 100
                except ValueError:
                    [_, _, _, _, pos_flow, _], _, _, x2y, y2x, *_ = pred['reg']
                del _
            if pos_flow is not None:
                pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
            if inpainted_b is not None:
                inpainted_b = inpainted_b.permute(0, 2, 3, 4, 1)
            if inpainted_f is not None:
                inpainted_f = inpainted_f.permute(0, 2, 3, 4, 1)

            images.extend(
                [inp[0], inp[1], x2y, y2x, mask, ncc_map, pos_flow, inpainted_f, inpainted_b])
            titles = ['x', 'y', 'x2y', 'y2x', 'mask', 'ncc map', 'pos_flow', 'inp_f', 'inp_b']

        if self.model == 3:
            gen = gradmap = None
            if 'gen' in keys:
                gen, pred_label = pred['gen']
            if 'inp' in keys:
                gen, y2x, pred_label, Y_disc_label = pred['inp']
            if 'reg' in keys:
                try:
                    [pos_flow, _], x2y, y2x = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1) * 100
                except ValueError:
                    [_, _, _, _, pos_flow, _], _, _, x2y, y2x, _ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
                del _
            if gen is not None:
                gradmap = 1 - gradientSimilarity(gen, inp[1], win=1, reduction='none')
            images.extend(
                [inp[0], inp[1], gen, x2y, y2x, pos_flow, gradmap])
            titles = [f'x', f'y', 'gen', 'x2g', 'g2x', 'florgb', 'gradmap']

        if self.model == 4:
            if 'gen' in keys:
                mask, _, x2y, y2x, inpainted_b, inpainted_f, _ = pred['gen']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                if self.ncc:
                    HF = 1 - self.ncc(inp[0], inpainted_f)
                    if inpainted_b is not None:
                        HB = 1 - self.ncc(inp[0], inpainted_b)

            if 'inp' in keys:
                mask, x2y, inpainted_b, inpainted_f = pred['inp']
                if self.ncc:
                    HF = 1 - self.ncc(inp[0], inpainted_f)
                    if inpainted_b is not None:
                        HB = 1 - self.ncc(inp[0], inpainted_b)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

            if 'reg' in keys:
                try:
                    [pos_flow, _], x2y, y2x = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1) * 100
                except ValueError:
                    [_, _, _, _, pos_flow, _], _, _, x2y, y2x, _ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
                grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()

                field = (pos_flow + grid)[..., [2, 1, 0]]
                flow = pos_flow[0, ::4, ::4]
                del _

            images.extend(
                [inp[0], inp[1], x2y, y2x, pos_flow, field, flow, inp[-1],
                 mask, inpainted_b, inpainted_f, HF, HB])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour', 'flow', 'seg_gt',
                      'seg_pred', 'inpainted_b', 'inpainted_f', 'HF', 'HB']

        if self.model == 5:
            if 'inp' in keys:
                mask, y2x, inpainted_b, inpainted_f = pred['inp']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

            if 'rec' in keys:
                mask, x2y, inpainted_b, inpainted_f = pred['rec']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0

            if 'gen' in keys:
                mask, inpainted_b, inpainted_f, x2y, *_ = pred['gen']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
            inp2y = None
            if 'reg' in keys:
                try:
                    [pos_flow, _], inp2y, y2x, *_ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1) * 100
                except ValueError:
                    [_, _, _, _, pos_flow, _], _, _, inp2y, y2x, *_, = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
                grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()
                field = (pos_flow + grid)[..., [2, 1, 0]]
                del _
            images.extend(
                [inp[0], inp[1], x2y, inp2y, y2x, pos_flow, field, inp[-1],
                 mask, inpainted_b, inpainted_f])
            titles = [f'moving', f'atlas', 'moving2fixed', 'inp2fixed', 'fixed2moving', 'field', 'contour', 'seg_gt',
                      'seg_pred', 'inpainted_b', 'inpainted_f']

        if self.model == 6:
            if 'gen' in keys:
                mask, inpainted_b, inpainted_f = pred['gen']
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
            if 'inp' in keys:
                mask2y, x2y, reg_inp_b, reg_inp_f = pred['inp']
                mask2y[mask2y > 0.5] = 1
                mask2y[mask2y <= 0.5] = 0

            if 'reg' in keys:
                try:
                    [pos_flow, _], x2y, y2x, *_ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1) * 100
                except ValueError:
                    [_, _, _, _, pos_flow, _], _, _, x2y, y2x, *_ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
                grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()
                field = (pos_flow + grid)[..., [2, 1, 0]]
                del _

            if 'rec' in keys:
                mask2y, x2y, _, reg_inp_f = pred['rec']
                mask2y[mask2y > 0.5] = 1
                mask2y[mask2y <= 0.5] = 0
            images.extend(
                [inp[0], inp[1], x2y, y2x, pos_flow, field, inp[-1],
                 mask, mask2y, inpainted_b, inpainted_f, reg_inp_b, reg_inp_f])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour', 'seg_gt',
                      'seg_pred', 'seg_pred2atlas', 'inpainted_b', 'inpainted_f', 'reg_inp_b', 'reg_inp_f']

        if self.model == 7:
            if 'gen' in keys:
                mask, mask2y, inpainted_b, inpainted_f = pred['gen']

            if 'inp' in keys:
                mask2y, x2y, inpainted_b, inpainted_f = pred['inp']

            if 'reg' in keys:
                try:
                    [pos_flow, _], x2y, y2x, *_ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1) * 100
                except ValueError:
                    [_, _, _, _, pos_flow, _], _, _, x2y, y2x, *_ = pred['reg']
                    pos_flow = pos_flow.permute(0, 2, 3, 4, 1)
                grid = torch.from_numpy(generate_grid(pos_flow.shape[1:-1])).unsqueeze(0).float()

                field = (pos_flow + grid)[..., [2, 1, 0]]
                del _

            if mask is not None:
                mask = (mask > .5).float()
            if mask2y is not None:
                mask2y = (mask2y > .5).float()

            images.extend(
                [inp[0], inp[1], x2y, y2x, pos_flow, field, inp[-1], mask, mask2y, inpainted_b, inpainted_f])
            titles = [f'moving', f'atlas', 'moving2fixed', 'fixed2moving', 'florgb', 'contour', 'seg_gt',
                      'seg_pred', 'seg_reg', 'inpainted_b', 'inpainted_f']

        while None in images:
            idx = images.index(None)
            images.pop(idx)
            titles.pop(idx)
        return images, titles, keys

    def __call__(self, logs: dict, f_name, inp, pred, norm):
        # update metrics logs
        logs = logs.copy()
        try:
            logs.update({
                'volume': (inp[-1] > .5).float().cpu().sum().item() /
                          (inp[0] > 1e-2).float().cpu().sum().item()
            })
        except Exception:
            pass
        self.logs.append([f_name, logs])
        if not self.save_img:
            return
        images, titles, keys = self.get_pred(inp, pred)
        f_name = '_'.join([*f_name.split('_'), *keys])
        self.plot(images, titles, f_name, logs)
        del images
        self.times += 1

    def __del__(self):
        if len(self.logs) == 0:
            return
        try:
            pids, logs = list(zip(*self.logs))
        except ValueError:
            print(f'Plot del error: \n{self.logs}')
            return
        from utils.utils import TXTLogs
        logger = TXTLogs(dir_path=self.dir_path, file_name='plot_info')
        for p, l in zip(pids, logs):
            logger.log(f'{p} : {l}')

        plot_dice_per_data(pids, data=logs, dir_path=self.dir_path)

    def plot(self, images, title, f_name, logs):
        if isinstance(images[0], torch.Tensor):
            images = [i.squeeze() for i in images]
            images = [(255 * (i - i.min()) / (i.max() - i.min())).cpu().numpy().astype(np.uint8)
                      if i.shape[-1] != 3 else i.cpu().numpy() for i in images]
        else:
            images = [(255 * (i - i.min()) / (i.max() - i.min())).astype(np.uint8)
                      if i.shape[-1] != 3 else i for i in images]

        rows, cols = self.slices_n, len(images)
        data = []
        titles = []
        for c in range(rows):
            for i in range(len(images)):
                if images[i].shape[-1] == 3:
                    i_z = images[i].shape[-2]
                    data.append(images[i][:, :, i_z // rows * c, :])
                else:
                    i_z = images[i].shape[-1]
                    data.append(images[i][:, :, i_z // rows * c])
                titles.append(title[i] + f'_{i_z // rows * c}')

        fig, _ = plot_slices(data, titles=titles, cmaps=['gray'], do_colorbars=True, show=False, width=30,
                             grid=(rows, cols), dpi=self.dpi, rotate=-90)
        ti = str(logs)[1:-1].replace(',', '\n')
        plt.suptitle(ti, size=20)
        plt.savefig(os.path.join(self.dir_path, 'epoch_' + f'{self.times}'.zfill(3) + f'{f_name}.{self.format}'),
                    transparent=True)
        plt.close(fig)
        del data, titles


def read_from_plot_txt(file_path):
    f = open(file_path, 'r')
    txt = f.read().splitlines()
    f.close()
    s = '{' + ','.join(["'" + t.split(' ')[0] + "'" + ''.join(t.split(' ')[1:]) for t in txt]).replace("'", '"') + '}'
    s = json.loads(s)
    return s


def plot_from_txt(file_path):
    def _decode(strs):
        strs = strs.removesuffix('\n')
        key = strs.split('{')[0].split(':')[0]
        value = json.loads(('{' + strs.split('{')[-1]).replace("'", '"'))
        return key, value

    f = open(file_path, 'r')
    txt = f.read().splitlines()
    f.close()
    txt = [t for t in txt if 'train:' in t or 'val:' in t]
    train_list = []
    val_list = []
    [train_list.append(v) if k == 'train' else val_list.append(v) for k, v in map(_decode, txt)]
    train_logs_df = pd.DataFrame(train_list)
    valid_logs_df = pd.DataFrame(val_list)
    path = os.path.split(file_path)[0]
    for i in train_logs_df.keys():
        plot_curve(path, train_logs_df, valid_logs_df, i)

    pass


def plot_curve(path, csv_logger):
    path = os.path.join(path, 'CURVE')
    make_dirs(path)
    sns.set_theme(style='whitegrid')
    dataframe = csv_logger.dataframe()
    metric_columns = dataframe.columns.drop(['epoch', 'stage'])
    for metric in metric_columns:
        plt.figure(figsize=(10, 7), dpi=200)
        sns.lineplot(data=dataframe, x='epoch', y=metric, hue='stage')
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel(f'{metric} Score', fontsize=15)
        plt.title(f'{metric} Score Plot', fontsize=15)
        plt.savefig(f'{path}/{metric}.png')
        plt.close('all')
    # plt.plot(train_logs_df.index.tolist(), train_logs_df[name].tolist(), 'g-', lw=1, label='Train')
    # plt.plot(valid_logs_df.index.tolist(), valid_logs_df[name].tolist(), 'r-', lw=1, label='Valid')
    # plt.legend(loc='best', fontsize=15)


def plot_dice_per_data(pids, data, dir_path):
    def log5(x):
        return np.log(x) / np.log(5)

    lost_key = []
    pids = list(pids)
    if isinstance(data, (list, tuple)):
        logs_d = {}
        keys = data[-1].keys()
        for i, d_l in enumerate(data):
            for k in keys:
                try:
                    v = d_l[k]
                except KeyError:
                    lost_key.append({pids[i]: k})
                    continue
                if k not in logs_d.keys():
                    logs_d[k] = [v]
                else:
                    logs_d[k].append(v)
        print(lost_key)
        data = logs_d

    if not os.path.exists(dir_path):
        make_dirs(dir_path)
    for key, value in data.items():
        if isinstance(key, str) and 'dice' in key and 'volume' in data.keys():
            y1 = np.array(value)
            y2 = np.array(data['volume'])
            fig, ax = plt.subplots(figsize=(min(26, int(12 * log5(len(value) / 20 + 4))), 6), dpi=200)
            x = np.arange(len(pids))
            width = 0.35
            rects1 = ax.bar(x - width / 2, y1, width, label='Dice')
            rects2 = ax.bar(x + width / 2, y2, width, label='Lesion Volume')

            ax.set_ylabel('Scores')
            ax.set_title(f'AVG Dice: {y1.mean()} \n AVG Vol: {y2.mean()}')
            ax.set_xticks(x, pids, rotation=60, fontsize=7)

            ax.legend()
            if len(value) < 150:
                ax.bar_label(rects1, fontsize=5, rotation=20)
                ax.bar_label(rects2, fontsize=5, rotation=20)
            fig.tight_layout()
            key = key + 'volume'
        else:
            y = np.array(value)
            plt.figure(figsize=(min(26, int(12 * log5(len(value) / 20 + 4))), 6), dpi=200)
            _pids = pids.copy()
            for l in lost_key:
                for k, v in l.items():
                    if v == key:
                        _pids.pop(_pids.index(k))
            rects = plt.bar(_pids, y, width=.35)
            plt.xticks(rotation=60, fontsize=7)
            if len(value) < 150:
                plt.bar_label(rects, fontsize=5, rotation=20)
            plt.title(f'AVG {key}: {y.mean()}')
        plt.savefig(os.path.join(dir_path, f'{key}.png'))
        plt.close('all')


def filter_names_for_boxplot(names, suppress_pattern, suppress_pattern_keep_first_as, replace_pattern_from=None,
                             replace_pattern_to=None):
    idx = []
    eff_names = []
    found_first = False
    for i, n in enumerate(names):
        if n.endswith(suppress_pattern):
            if not found_first:
                found_first = True
                idx.append(i)
                eff_names.append(suppress_pattern_keep_first_as)
        else:
            if n.endswith(replace_pattern_from):
                replaced_str = n[0:-len(replace_pattern_from)] + replace_pattern_to
                eff_names.append(replaced_str)
            else:
                eff_names.append(n)
            idx.append(i)

    return idx, eff_names


def plot_boxplot(compound_results_orig, compound_names_orig, title=None, semilogy=False, showfliers=True,
                 suppress_pattern=None, suppress_pattern_keep_first_as=None, figsize=(8, 6),
                 replace_pattern_from=None,
                 replace_pattern_to=None,
                 show_labels=True, fix_aspect=None,
                 path=None,
                 grouped=False
                 ):
    if suppress_pattern is not None:
        idx_to_keep, compound_names = filter_names_for_boxplot(names=compound_names_orig,
                                                               suppress_pattern=suppress_pattern,
                                                               suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                                                               replace_pattern_from=replace_pattern_from,
                                                               replace_pattern_to=replace_pattern_to,
                                                               )
        compound_results = [compound_results_orig[i] for i in idx_to_keep]
    else:
        compound_results = compound_results_orig
        compound_names = compound_names_orig

    # create a figure instance
    fig = plt.figure(1, figsize=figsize, dpi=200)

    # create an axes instance
    ax = fig.add_subplot(111)

    # set axis tick
    if semilogy:
        ax.semilogy()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_tick_params(left=True, direction='in', width=1)
    ax.yaxis.set_tick_params(right=True, direction='in', width=1)
    ax.xaxis.set_tick_params(top=False, direction='in', width=1)
    ax.xaxis.set_tick_params(bottom=False, direction='in', width=1)

    # create the boxplot
    if show_labels:
        bp = plt.boxplot(compound_results, vert=True, whis=1.5, meanline=True, widths=0.16, showfliers=showfliers,
                         showcaps=False, patch_artist=True, labels=compound_names)

        # rotate x labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
    else:
        empty_compound_names = [''] * len(compound_names)
        bp = plt.boxplot(compound_results, vert=True, whis=1.5, meanline=True, widths=0.16, showfliers=showfliers,
                         showcaps=False, patch_artist=True, labels=empty_compound_names)
    if title is not None:
        plt.title(title, fontsize=8)  # 标题，并设定字号大小
    # set properties of boxes, medians, whiskers, fliers
    plt.setp(bp['medians'], color='orange')
    plt.setp(bp['boxes'], color='blue')
    plt.setp(bp['whiskers'], linestyle='-', color='blue')
    plt.setp(bp['fliers'], marker='o', markersize=5, markeredgecolor='blue')

    # setup font
    # font = {'family': 'normal', 'weight': 'semibold', 'size': 10}
    font = {'family': 'sans-serif', 'size': 8}
    matplotlib.rc('font', **font)

    # set the line width of the figure
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    if fix_aspect is not None:
        yrange = ax.get_ylim()
        yext = yrange[1] - yrange[0]
        xext = float(len(compound_names))
        ax.set_aspect(fix_aspect * xext / yext)

    if path is not None:
        make_dirs(os.path.split(path)[0])
        plt.savefig(path)
    return fig


if __name__ == "__main__":
    plot_from_txt(
        '/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/result/GIRNet_gen_loss_gir5_inp_loss_BraTSNCC21_atlaspace/fold_0_t1/print_2022-08-21_23:28:19.txt'
    )
    pass
