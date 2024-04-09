import random

import numpy as np
import torch
import torch.nn as nn
from .helper import Encoder, Decoder
import torch.nn.functional as F


class MaskG(nn.Module):
    def __init__(self,
                 shape,
                 input_channel=1,
                 nb_conv_per_level=1,
                 nb_features=(
                         [16, 32, 32, 32],
                         [32, 32, 32, 32, 32, 16, 16]
                 ),
                 max_pool_size=2,
                 norm=True,
                 load_state_path=None,
                 conv_num=1
                 ):
        """
            nb_features = [
                        [16, 32, 32, 32],  # encoder
                        [32, 32, 32, 32, 32, 16, 16]  # decoder
                    ]
        """
        super(MaskG, self).__init__()
        if isinstance(shape, int):
            ndims = shape
            shape = None
        else:
            ndims = len(shape)
        self.encoder = Encoder(ndims,
                               input_channel,
                               nb_conv_per_level,
                               nb_features,
                               max_pool_size,
                               norm=norm,
                               conv_num=conv_num)

        if load_state_path is not None:
            check = torch.load(load_state_path, map_location='cpu')['state_dict']
            self.encoder.load_state_dict(check)
            print(f'load encoder: {load_state_path}')

        self.decoder = Decoder(ndims,
                               nb_conv_per_level,
                               nb_features,
                               max_pool_size,
                               False,
                               1,
                               norm=norm,
                               shape=shape,
                               conv_num=conv_num)

        self.final = nn.Sequential(
            getattr(nn, 'Conv%dd' % ndims)(nb_features[1][-1], 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, y], dim=1)
        x = self.encoder(x)
        x = self.decoder(x, is_final_stage=True)
        x = self.final(x)
        return x


class MaskG2(nn.Module):
    def __init__(self,
                 ndims,
                 input_channel=(1, 3),
                 nb_conv_per_level=1,
                 nb_features=(
                         [16, 32, 32, 32],
                         [32, 32, 32, 32, 32, 16, 16]
                 ),
                 max_pool_size=2,
                 random_mask=False,
                 norm=False
                 ):
        """
            nb_features = [
                        [16, 32, 32, 32],  # encoder
                        [32, 32, 32, 32, 32, 16, 16]  # decoder
                    ]
        """
        super(MaskG2, self).__init__()
        self.encoder1 = Encoder(ndims,
                                input_channel[0],
                                nb_conv_per_level,
                                nb_features,
                                max_pool_size,
                                norm=norm)
        self.encoder2 = Encoder(ndims,
                                input_channel[1],
                                nb_conv_per_level,
                                nb_features,
                                max_pool_size,
                                norm=norm)
        self.decoder = Decoder(ndims,
                               nb_conv_per_level,
                               nb_features,
                               max_pool_size,
                               False,
                               2,
                               norm=norm)
        self.final = nn.Sequential(
            getattr(nn, 'Conv%dd' % ndims)(nb_features[1][-1], 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.random_mask = RandMaskG() if random_mask else None

    def forward(self, x, y=None):
        if self.random_mask is not None:
            return self.random_mask(x)
        x, x_his = self.encoder1(x)
        y, y_his = self.encoder2(y)
        his = [torch.cat([i, j], dim=1) for i, j in zip(x_his, y_his)]
        del x_his, y_his
        x = self.decoder([torch.cat([x, y], dim=1), his], is_final_stage=True)
        x = self.final(x)
        return x


class Boundary(nn.Module):
    def __init__(self, kernel=None):
        # a pixel belongs to the mask's boundary if its value is 1 and
        # it has at least one 0-valued neighboring pixel (non-diagonal).
        # we use a conv kernel that yields out[i,j] = 4*M[i,j] - (M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]),
        # hence a out[i,j] >= 0.5 if and only if the pixel (i,j) belongs to the boundary of M.
        super(Boundary, self).__init__()

        if kernel is None:
            kernel = torch.Tensor([[[0., 0., 0.],
                                    [0., -1., 0.],
                                    [0., 0., 0.]],

                                   [[0., -1., 0.],
                                    [-1, 6., -1.],
                                    [-0, -1., 0.]],

                                   [[0., 0., 0.],
                                    [0., -1., 0.],
                                    [0., 0., 0.]],
                                   ]).view((1, 1, 3, 3, 3))
        if isinstance(kernel, int):
            k = kernel
            kernel = torch.zeros(1, 1, k, k, k)
            kernel[0, 0, k // 2, k // 2, k // 2] = 0
            kernel[0, 0, k // 2 - 1, k // 2, k // 2] = 1
            kernel[0, 0, k // 2, k // 2 - 1, k // 2] = 1
            kernel[0, 0, k // 2, k // 2, k // 2 - 1] = 1
            kernel[0, 0, k // 2 + 1, k // 2, k // 2] = 1
            kernel[0, 0, k // 2, k // 2 + 1, k // 2] = 1
            kernel[0, 0, k // 2, k // 2, k // 2 + 1] = 1

            kernel[0, 0, 0, k // 2, k // 2] = -1
            kernel[0, 0, k // 2, 0, k // 2] = -1
            kernel[0, 0, k // 2, k // 2, 0] = -1
            kernel[0, 0, -1, k // 2, k // 2] = -1
            kernel[0, 0, k // 2, -1, k // 2] = -1
            kernel[0, 0, k // 2, k // 2, -1] = -1
        self.pad = kernel.shape[-1] // 2
        self.register_buffer('kernel', kernel)

    def conv(self, x):
        x = F.pad(x, (self.pad,) * 6, mode='replicate')
        x = F.conv3d(x, self.kernel, padding=0, stride=1)
        return x

    def forward(self, m):
        fore_boundary = self.conv(m) >= 0.5
        back_boundary = self.conv(1 - m) >= 0.5
        return fore_boundary + back_boundary


class Mask(nn.Module):
    def __init__(self):
        # a pixel belongs to the mask's boundary if its value is 1 and
        # it has at least one 0-valued neighboring pixel (non-diagonal).
        # we use a conv kernel that yields out[i,j] = 4*M[i,j] - (M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]),
        # hence a out[i,j] >= 0.5 if and only if the pixel (i,j) belongs to the boundary of M.
        super(Mask, self).__init__()
        self.register_buffer('kernel',
                             torch.Tensor([[[0., 0., 0.],
                                            [0., -1., 0.],
                                            [0., 0., 0.]],

                                           [[0., -1., 0.],
                                            [-1, 6., -1.],
                                            [-0, -1., 0.]],

                                           [[0., 0., 0.],
                                            [0., -1., 0.],
                                            [0., 0., 0.]],
                                           ]).view((1, 1, 3, 3, 3)))

    def forward(self, m):
        fore_boundary = F.conv3d(m, self.kernel, padding=1) >= 0
        back_boundary = F.conv3d(1 - m, self.kernel, padding=1) >= 0.5
        return fore_boundary + back_boundary


class RandMaskG:
    def __init__(self, mask_scale=(0.3, 0.7), length=5):
        super(RandMaskG, self).__init__()
        self.length = length
        mask_scale = mask_scale if isinstance(mask_scale, (list, tuple)) else (mask_scale, mask_scale)
        self.mask_scale = np.linspace(*mask_scale, length)

    def _get_rand_size(self, x):
        idx = random.randint(0, self.length - 1)
        return int(x * self.mask_scale[idx])

    def _get_rand_center(self, mask_l, img_l):
        return random.randint(mask_l // 2, img_l - mask_l // 2)

    def get_roi(self, input_shape):
        size = [self._get_rand_size(i) for i in input_shape]
        center = [self._get_rand_center(m, i) for m, i in zip(size, input_shape)]
        start = [np.clip(c - s // 2, 0, i - 1) for c, s, i in zip(center, size, input_shape)]
        end = [np.clip(c + s // 2, 0, i - 1) for c, s, i in zip(center, size, input_shape)]
        return start, end

    def __call__(self, x):
        input_shape = x.shape[2:]
        x = torch.zeros_like(x)
        start, end = self.get_roi(input_shape)
        slices = [slice(None)] * 2
        slices.extend([slice(s, e) for s, e in zip(start, end)])
        x.data[slices].fill_(1.0)
        return x
