# -*- encoding: utf-8 -*-
"""
@File    :   SymNet.py   
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/14 10:20 AM   liu.yang      1.0         None
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def generate_grid(imgshape):
    return np.mgrid[0:imgshape[0], 0:imgshape[1], 0:imgshape[2]].transpose(1, 2, 3, 0)[..., [2, 1, 0]].astype(float)


def normalize_grid(sample_grid):
    size_tensor = sample_grid.size()
    sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
            size_tensor[3] - 1) * 2
    sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
            size_tensor[2] - 1) * 2
    sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
            size_tensor[1] - 1) * 2
    return sample_grid


class SymUNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(SymUNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU(inplace=True))
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Softsign())
        return layer

    def forward(self, x, y):
        e0 = torch.cat((x, y), 1)
        e0 = self.eninput(e0)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)
        del e4
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)
        d1 = torch.cat((self.up2(d0), e2), 1)
        del e2, d0
        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)
        del e1, d1
        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        del e0, d2
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)
        f_yx = self.dc10(d3)
        del d3
        return f_xy, f_yx


def spatial_transform(x: torch.Tensor, flow, grid=None, ismask=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
        flow = torch.from_numpy(flow).float()
    else:
        x = x.float()
        flow = flow.float()

    while len(x.shape) != 5:
        x = x.unsqueeze(0)
    while len(flow.shape) != 5:
        flow = flow.unsqueeze(0)

    if x.shape[-1] == 1:
        x = x.permute(0, 4, 1, 2, 3)

    if flow.shape[1] == 3:
        flow = flow.permute(0, 2, 3, 4, 1)

    if grid is None:
        grid = generate_grid(x.shape[2:])
        grid = grid.reshape(1, *grid.shape)
        grid = torch.from_numpy(grid).float()
    else:
        grid = grid.float()
    grid = grid + flow
    size_tensor = grid.size()
    grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
            size_tensor[3] - 1) * 2
    grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
            size_tensor[2] - 1) * 2
    grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
            size_tensor[1] - 1) * 2
    if ismask:
        mask = torch.zeros_like(x)
        for label in x.round().int().unique():
            mask += label.item() * torch.nn.functional.grid_sample((x == label).float(), grid, mode='bilinear',
                                                                   align_corners=True).round()
        x = mask
    else:
        x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', align_corners=True)
    return x


class SpatialTransform(nn.Module):
    def __init__(self, range_flow=1):
        super(SpatialTransform, self).__init__()
        self.range_flow = range_flow

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow.permute(0, 2, 3, 4, 1) * self.range_flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', align_corners=True)

        return flow


class LandmarkTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self):
        super().__init__()

    def forward(self, target_landmark, flow, grid):
        # new locations
        if isinstance(target_landmark, torch.Tensor):
            target_landmark = target_landmark.squeeze().to(torch.int)

        if grid.device != flow.device:
            grid = grid.to(flow.device)

        new_locs = grid + flow.permute(0, 2, 3, 4, 1)
        source_target = []

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        for l in target_landmark:
            source_target.append(
                [new_locs[..., l[0], l[1], l[2], i].cpu().item() for i in range(3)])
        return torch.tensor(source_target).unsqueeze(0)


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7, range_flow=100):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step
        self.range_flow = range_flow

    def forward(self, velocity, sample_grid):
        flow = velocity / (2.0 ** self.time_step)
        size_tensor = sample_grid.size()
        # 0.5 flow
        for _ in range(self.time_step):
            grid = sample_grid + (flow.permute(0, 2, 3, 4, 1) * self.range_flow)
            grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
            grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
            grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
        return flow


class CompositionTransform(nn.Module):
    def __init__(self, range_flow=100):
        super(CompositionTransform, self).__init__()
        self.range_flow = range_flow

    def forward(self, flow_1, flow_2, sample_grid):
        size_tensor = sample_grid.size()
        grid = sample_grid + (flow_2.permute(0, 2, 3, 4, 1) * self.range_flow)
        grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        compos_flow = F.grid_sample(flow_1, grid, mode='bilinear', align_corners=True) + flow_2
        return compos_flow


import torch
from torch import Tensor
from typing import TypeVar, Mapping


class SymNetSvF:
    def __init__(self, time_step=7, range_flow=100, sym=True):
        self.diff_transform = DiffeomorphicTransform(time_step=time_step, range_flow=range_flow)
        self.com_transform = CompositionTransform(range_flow)
        self.transform = SpatialTransform(range_flow)
        self.range_flow = range_flow
        self.sym = sym

    def get_flow(self, F_xy, F_yx, grid):
        F_X_Y_half = self.diff_transform(F_xy, grid)
        F_Y_X_half = self.diff_transform(F_yx, grid)

        F_X_Y_half_inv = self.diff_transform(-F_xy, grid)
        F_Y_X_half_inv = self.diff_transform(-F_yx, grid)

        F_X_Y = self.com_transform(F_X_Y_half, F_Y_X_half_inv, grid)
        F_Y_X = self.com_transform(F_Y_X_half, F_X_Y_half_inv, grid)
        return F_xy, F_yx, F_X_Y_half, F_Y_X_half, F_X_Y, F_Y_X

    def __call__(self, X, Y, F_xy, F_yx, grid, train=False, Moving=None, Fixed=None):
        if self.sym:
            F_xy, F_yx, F_X_Y_half, F_Y_X_half, F_X_Y, F_Y_X = self.get_flow(F_xy, F_yx, grid)
        else:
            F_X_Y, F_Y_X = self.diff_transform(F_xy, grid), self.diff_transform(F_yx, grid)
        if Moving is not None:
            X = Moving
        if Fixed is not None:
            Y = Fixed
        X_Y = self.transform(X, F_X_Y, grid)
        Y_X = self.transform(Y, F_Y_X, grid)
        if train:
            if self.sym:
                X_Y_half = self.transform(X, F_X_Y_half, grid)
                Y_X_half = self.transform(Y, F_Y_X_half, grid)
                return [F_xy * self.range_flow,
                        F_yx * self.range_flow,
                        F_X_Y_half * self.range_flow,
                        F_Y_X_half * self.range_flow,
                        F_X_Y * self.range_flow,
                        F_Y_X * self.range_flow], X_Y_half, Y_X_half, X_Y, Y_X, grid, X
            else:
                return [F_xy * self.range_flow,
                        F_yx * self.range_flow,
                        None,
                        None,
                        F_X_Y * self.range_flow,
                        F_Y_X * self.range_flow], None, None, X_Y, Y_X, grid, X

        return [F_X_Y * self.range_flow, F_Y_X * self.range_flow], X_Y, Y_X


class SymNet(nn.Module):
    def __init__(self, img_shape, time_step=7, range_flow=100, sym=True):
        super(SymNet, self).__init__()
        self.unet = SymUNet(2, 3, 7)
        self.range_flow = range_flow
        self.svf = SymNetSvF(time_step=time_step, range_flow=range_flow, sym=sym)

        grid_np = generate_grid(img_shape)
        grid = torch.from_numpy(np.reshape(grid_np, (1,) + grid_np.shape)).float()
        self.register_buffer('grid', grid)

    def forward(self, X, Y, train=False, Moving=None, Fixed=None):
        F_xy, F_yx = self.unet(X, Y)
        return self.svf(X, Y, F_xy, F_yx, self.grid, train=train, Moving=Moving, Fixed=Fixed)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None and 'grid' not in name:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set and 'grid' not in name:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + '_extra_state'
        if getattr(self.__class__, "get_extra_state", nn.Module.get_extra_state) is not nn.Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrederedDict` is created and returned.
    T_destination = TypeVar('T_destination', bound=Mapping[str, Tensor])


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


def SymJacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def sym_neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * SymJacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag)) / num_ele

    return diff


class SymnetSingle(nn.Module):
    def __init__(self, img_shape, time_step=7, range_flow=100, sym=True, gt=False, HM=False):
        super().__init__()
        self.symnet = SymNet(img_shape, time_step=time_step, range_flow=range_flow, sym=sym)
        self.model_id = 5
        self.tags = ('reg',)
        self.gt = gt
        from utils.metrics import HistogramMatching
        self.HM = HistogramMatching() if HM else None

    def grid(self):
        return self.symnet.grid

    def forward(self, tag, inp, train=False):
        x, y, mask = inp[0], inp[1], inp[2]
        if self.gt:
            with torch.no_grad():
                mask = (mask > 0.5).float()
                _, _, y2x = self.symnet(x, y, train=False)
                if self.HM:
                    y2x = self.HM(y2x, x)
                moving = y2x * mask + x * (1 - mask)
        else:
            moving = x
        if tag == 'infer':
            return self.symnet(x, y, train=False)
        return self.symnet(x, y, Moving=moving, train=True)

    def __getitem__(self, item):
        return self
