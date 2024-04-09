import torch
import torch.nn as nn
from .helper import Encoder, Decoder


class Inpainter(nn.Module):
    def __init__(self,
                 shape,
                 input_channel=1,
                 nb_conv_per_level=1,
                 nb_features=(
                         [16, 32, 32, 32],
                         [32, 32, 32, 32, 32, 16, 16]
                 ),
                 max_pool_size=2,
                 norm=False,
                 output_channel=1,
                 conv_num=1
                 ):
        """
            nb_features = [
                        [16, 32, 32, 32],  # encoder
                        [32, 32, 32, 32, 32, 16, 16]  # decoder
                    ]
        """
        super(Inpainter, self).__init__()
        ndims = len(shape)
        self.input_channel = input_channel
        self.encoder = Encoder(ndims,
                               input_channel,
                               nb_conv_per_level,
                               nb_features,
                               max_pool_size,
                               norm=norm,
                               conv_num=conv_num)
        self.decoder = Decoder(ndims,
                               nb_conv_per_level,
                               nb_features,
                               max_pool_size,
                               False,
                               shape=shape,
                               norm=norm,
                               conv_num=conv_num)
        self.final = nn.Sequential(
            getattr(nn, 'Conv%dd' % ndims)(nb_features[1][-1], output_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, *x):
        x = self.encoder(torch.cat(x, dim=1))
        x = self.decoder(x, is_final_stage=True)
        x = self.final(x)
        return x


class Inpainter2(nn.Module):
    def __init__(self,
                 shape,
                 input_channel=(2, 1),
                 nb_conv_per_level=1,
                 nb_features=(
                         [16, 32, 32, 32],
                         [32, 32, 32, 32, 32, 16, 16]
                 ),
                 max_pool_size=2,
                 norm=False,
                 output_channel=1,
                 conv_num=1
                 ):
        """
            nb_features = [
                        [16, 32, 32, 32],  # encoder
                        [32, 32, 32, 32, 32, 16, 16]  # decoder
                    ]
        """
        super(Inpainter2, self).__init__()
        self.input_channel = input_channel
        if isinstance(shape, int):
            ndims = shape
            shape = None
        else:
            ndims = len(shape)
        self.encoder1 = Encoder(ndims,
                                input_channel[0],
                                nb_conv_per_level,
                                nb_features,
                                max_pool_size,
                                norm=norm,
                                conv_num=conv_num)
        self.encoder2 = Encoder(ndims,
                                input_channel[1],
                                nb_conv_per_level,
                                nb_features,
                                max_pool_size,
                                norm=norm,
                                conv_num=conv_num)
        self.decoder = Decoder(ndims,
                               nb_conv_per_level,
                               nb_features,
                               max_pool_size,
                               False,
                               2,
                               norm=norm,
                               shape=shape,
                               conv_num=conv_num)
        self.final = nn.Sequential(
            getattr(nn, 'Conv%dd' % ndims)(nb_features[1][-1], output_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, *x):
        if len(x) == 3:
            x, m, y = x
            x = torch.cat([x, m], dim=1)
        else:
            x, y = x
        x, x_his = self.encoder1(x)
        y, y_his = self.encoder2(y)
        his = [torch.cat([i, j], dim=1) for i, j in zip(x_his, y_his)]
        del x_his, y_his
        x = self.decoder([torch.cat([x, y], dim=1), his], is_final_stage=True)
        x = self.final(x)
        return x


class NaiveInpainter(nn.Module):
    def __init__(self):
        super(NaiveInpainter, self).__init__()

    def forward(self, *x):
        if len(x) == 3:
            x, mask, y = x
        else:
            x, y = x
            x, mask = torch.chunk(x, 2, dim=1)
        return x * (1 - mask) + mask * y


import torch.nn.functional as F


class GaussianInpainter(nn.Module):
    def __init__(self, sigma=5, kernel_size=7, reps=2, scale_factor=1):
        super(GaussianInpainter, self).__init__()
        self.reps = reps
        self.padding = kernel_size // 2
        self.downsample = nn.AvgPool3d(scale_factor) if scale_factor > 1 else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest') if scale_factor > 1 else nn.Identity()
        squared_dists = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size) ** 2
        gaussian_kernel = torch.exp(-0.5 * squared_dists / sigma ** 2)
        gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).view(1, 1, kernel_size).repeat(2, 1, 1)
        self.register_buffer('gaussian_kernel', gaussian_kernel)

    def gaussian_filter(self, x):
        # due to separability we can apply two 1d gaussian filters to get some speedup

        v = F.conv3d(x, self.gaussian_kernel.unsqueeze(3).unsqueeze(4), padding=(self.padding, 0, 0), groups=2)
        h = F.conv3d(v, self.gaussian_kernel.unsqueeze(2).unsqueeze(4), padding=(0, self.padding, 0), groups=2)
        h = F.conv3d(h, self.gaussian_kernel.unsqueeze(2).unsqueeze(3), padding=(0, 0, self.padding), groups=2)
        return h

    def forward(self, x, mask, *_):
        # to perform the same convolution on each channel of x and on the mask,
        # we concatenate x and m and perform a convolution with groups=num_channels=4
        u = torch.cat((x, mask), 1)
        epsilon = u.sum((2, 3, 4), keepdim=True) * 1e-8

        u = self.downsample(u)
        for _ in range(self.reps):
            u = self.gaussian_filter(u)
        u = self.upsample(u)
        u = u + epsilon
        filtered_x = u[:, :-1]
        filtered_m = u[:, -1:]
        return filtered_x / filtered_m
