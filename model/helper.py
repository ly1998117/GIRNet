import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler


###############################################################################
# Framework Class
class Network(nn.Module):
    def __init__(self, tags, ):
        super().__init__()


###############################################################################
# Helper Functions
###############################################################################


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if isinstance(gpu_ids, (list, tuple)):
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    else:
        assert (torch.cuda.is_available())
        net.to(gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


###############################################################################
# Layers
###############################################################################


class Encoder(nn.Module):
    def __init__(self,
                 ndims,
                 input_channel,
                 nb_conv_per_level,
                 nb_features,
                 max_pool_size=2,
                 norm=False,
                 de_conv=False,
                 conv_num=1
                 ):
        """
            Parameters:
                nb_features: [
                            [16, 32, 32, 32],  # encoder
                            [32, 32, 32, 32, 32, 16, 16]  # decoder
                        ]
                input_channel: Number of input features.
                nb_conv_per_level: Number of convolutions per unet level. Default is 1.
                max_pool_size: max pooling kernel size.
        """
        super(Encoder, self).__init__()
        # extract any surplus (full resolution) decoder convolutions
        enc_nf, _ = nb_features
        nb_dec_convs = len(enc_nf)
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if not de_conv:
            if isinstance(max_pool_size, int):
                max_pool_size = [max_pool_size] * self.nb_levels
            MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
            self.pooling = [MaxPooling(s) for s in max_pool_size]
            stride = 1
        else:
            self.pooling = None
            stride = 2

        self.encoder_nfs = [input_channel]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(
                    ConvBlock(ndims, input_channel, nf, stride=stride, norm=norm, conv_num=conv_num))
                input_channel = nf
            self.encoder.append(convs)
            self.encoder_nfs.append(input_channel)

    def get_encoder_nfs(self):
        return self.encoder_nfs

    def forward(self, x, skip=True):
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            if skip:
                x_history.append(x)
            if self.pooling:
                x = self.pooling[level](x)
        if skip:
            return x, x_history
        else:
            return x


def clip(x, num, down=True):
    if down or x % num == 0:
        return x // num
    else:
        return x // num + 1


class Decoder(nn.Module):
    def __init__(self,
                 ndims,
                 nb_conv_per_level,
                 nb_features,
                 max_pool,
                 half_res=False,
                 encoder_num=1,
                 norm=False,
                 skip=True,
                 shape=None,
                 de_conv=False,
                 conv_num=1
                 ):
        super(Decoder, self).__init__()
        # extract any surplus (full resolution) decoder convolutions
        self.skip = skip

        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level)
        self.half_res = half_res

        if shape is not None:
            self.feature_map_shape = [[clip(s, 2 ** i, not de_conv) for s in shape] for i in range(nb_dec_convs + 1)]
            self.feature_map_shape.reverse()
        else:
            self.feature_map_shape = None
        if not de_conv:
            if isinstance(max_pool, int):
                max_pool = [max_pool] * (self.nb_levels + 1)
            self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]
        else:
            self.upsampling = None

        encoder_nfs = []
        for level in range(self.nb_levels):
            encoder_nfs.append(enc_nf[level * nb_conv_per_level + nb_conv_per_level - 1] * encoder_num)
        prev_nf = encoder_nfs[-1]
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf, norm=norm, trans=de_conv, conv_num=conv_num))
                prev_nf = nf
            self.decoder.append(convs)
            if (not half_res and self.skip) or (level < (self.nb_levels - 1) and self.skip):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf, norm=norm, conv_num=conv_num))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, inp, is_final_stage=True):
        if self.skip:
            x, x_history = inp
        else:
            x = inp
        del inp
        i = 1
        for level, convs in enumerate(self.decoder):
            if self.feature_map_shape and sum(x.shape[2:]) != sum(self.feature_map_shape[level]):
                pos = []
                for o, t in zip(x.shape[2:], self.feature_map_shape[level]):
                    pos.extend([0, t - o])
                pos.reverse()
                x = torch.nn.functional.pad(x, pos).contiguous()
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 1):
                if self.upsampling:
                    x = self.upsampling[level](x)
                    if self.feature_map_shape and sum(x.shape[2:]) != sum(self.feature_map_shape[level + 1]):
                        pos = []
                        for o, t in zip(x.shape[2:], self.feature_map_shape[level + 1]):
                            pos.extend([0, t - o])
                        pos.reverse()
                        x = torch.nn.functional.pad(x, pos).contiguous()
                if self.skip:
                    if is_final_stage:
                        x = torch.cat([x, x_history.pop()], dim=1)
                    else:
                        x = torch.cat([x, x_history[-i]], dim=1)
                i += 1

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, conv_num=1, stride=1, norm=False, trans=False,
                 activation=True):
        super().__init__()
        self.ndims = ndims
        self.norm = norm
        conv = []
        if conv_num > 1:
            conv.extend([
                self.get_conv(in_channels, in_channels, stride) for _ in range(conv_num - 1)
            ])

        if not trans:
            conv.append(getattr(nn, 'Conv%dd' % self.ndims)(in_channels, out_channels, 3, stride, 1))
        else:
            conv.append(getattr(nn, f'ConvTranspose{ndims}d')(in_channels, out_channels,
                                                              kernel_size=3,
                                                              stride=2,
                                                              padding=1,
                                                              output_padding=1))
        self.conv = nn.Sequential(*conv)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if activation else None

    def get_conv(self, in_channels, out_channels, stride):
        if self.norm and out_channels > 1:
            return nn.Sequential(
                getattr(nn, 'Conv%dd' % self.ndims)(in_channels, out_channels, 3, stride, 1),
                getattr(nn, f'BatchNorm{self.ndims}d')(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                getattr(nn, 'Conv%dd' % self.ndims)(in_channels, out_channels, 3, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        return out

# class ConvBlock(nn.Module):
#     """
#     Specific convolutional block followed by leakyrelu for unet.
#     """
#
#     def __init__(self, ndims, in_channels, out_channels, conv_num=1, stride=1, norm=False, trans=False,
#                  activation=True):
#         super().__init__()
#         Conv = getattr(nn, 'Conv%dd' % ndims)
#         conv = []
#         if conv_num > 1:
#             conv.extend([
#                 Conv(in_channels, in_channels, 3, stride, 1) for _ in range(conv_num - 1)
#             ])
#
#         if not trans:
#             conv.append(Conv(in_channels, out_channels, 3, stride, 1))
#
#         else:
#             conv.append(getattr(nn, f'ConvTranspose{ndims}d')(in_channels, out_channels,
#                                                               kernel_size=3,
#                                                               stride=2,
#                                                               padding=1,
#                                                               output_padding=1))
#         self.conv = nn.Sequential(*conv)
#         self.activation = nn.LeakyReLU(0.2, inplace=True) if activation else None
#         self.norm = getattr(nn, f'BatchNorm{ndims}d')(num_channels=out_channels) if norm else None
#
#     def forward(self, x):
#         out = self.conv(x)
#         if self.norm is not None:
#             out = self.norm(out)
#         if self.activation:
#             out = self.activation(out)
#         return out
