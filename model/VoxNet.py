import numpy as np
import torch
import torch.nn as nn

from .helper import Encoder, Decoder, F
from torch.distributions import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', unit=False):
        super().__init__()

        self.mode = mode
        self.unit = unit
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.meshgrid(vectors)
        grid = torch.stack(grid)[[2, 1, 0], ...]
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.padding_mode = 'zeros'
        if unit:
            for i in range(3):
                grid[:, i, ...] = 2 * grid[:, i, ...] / (grid.shape[2:][-i - 1] - 1) - 1
            self.padding_mode = 'border'
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def get_grid(self):
        return self.grid

    def forward(self, src, flow):
        new_locs = self.grid + flow
        if not self.unit:
            shape = flow.shape[2:]
            # need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * new_locs[:, i, ...] / (shape[-i - 1] - 1) - 1
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode=self.padding_mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, unit=False):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape, mode='bilinear', unit=unit)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class LandmarkTransformer(SpatialTransformer):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', direction='pos'):
        super().__init__(size, mode)
        self.mode = mode
        self.direction = direction

    def forward(self, target_landmark, flow):
        if self.direction != 'pos':
            flow = -flow
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        source_target = []

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            for l in target_landmark:
                source_target.append([new_locs[..., l[0], l[1], i]._to_cpu().item() for i in range(len(shape))])
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            for l in target_landmark:
                source_target.append(
                    [new_locs[..., l[0], l[1], l[2], i]._to_cpu().item() for i in range(len(shape))])
        return source_target


class VoxDecoder(nn.Module):
    def __init__(self,
                 inshape,
                 nb_conv_per_level,
                 nb_features,
                 max_pool,
                 half_res,
                 int_steps=7,
                 int_downsize=2,
                 ):
        super(VoxDecoder, self).__init__()
        ndims = len(inshape)
        self.decoder = Decoder(ndims=ndims,
                               nb_conv_per_level=nb_conv_per_level,
                               nb_features=nb_features,
                               max_pool=max_pool,
                               half_res=half_res,
                               encoder_num=1)
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.decoder.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if not half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

    def transformer(self):
        return self.integrate.transformer

    def forward(self, inp, is_final=True):

        x = self.decoder(inp, is_final)
        # transform into flow field
        pos_flow = self.flow(x)
        # resize flow for integration
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow)
        return pos_flow, neg_flow, preint_flow


class VoxNet(nn.Module):
    def __init__(self,
                 input_shape,
                 input_channel=2,
                 nb_conv_per_level=1,
                 nb_features=(
                         [16, 32, 32, 32],
                         [32, 32, 32, 32, 32, 16, 16]
                 ),
                 half_res=False,
                 int_steps=7,
                 int_downsize=1,
                 max_pool_size=2):
        super(VoxNet, self).__init__()
        self.tags = ('reg',)
        self.model_id = 1
        ndims = len(input_shape)
        self.encoder = Encoder(ndims=ndims,
                               input_channel=input_channel,
                               nb_conv_per_level=nb_conv_per_level,
                               nb_features=nb_features,
                               max_pool_size=max_pool_size)
        self.vox_decoder = VoxDecoder(inshape=input_shape,
                                      nb_conv_per_level=nb_conv_per_level,
                                      nb_features=nb_features,
                                      max_pool=max_pool_size,
                                      half_res=half_res,
                                      int_steps=int_steps,
                                      int_downsize=int_downsize)

        self.transformer = self.vox_decoder.transformer() if int_downsize == 1 else SpatialTransformer(size=input_shape)

    def grid(self):
        return self.transformer.get_grid()

    def forward(self, tag, inp, train=False):
        x, y = inp[0], inp[1]
        pos_flow, neg_flow, preint_flow = self.vox_decoder(self.encoder(torch.cat([x, y], dim=1)))
        x2y = self.transformer(x, pos_flow)
        y2x = self.transformer(y, neg_flow)
        if tag == 'infer':
            return [pos_flow, neg_flow], x2y, y2x
        return [pos_flow, neg_flow, preint_flow], x2y, y2x, self.grid()

    def __getitem__(self, item):
        return self


class DiffDecoder(nn.Module):
    def __init__(self,
                 inshape,
                 nb_conv_per_level,
                 nb_features,
                 max_pool,
                 int_steps=7,
                 int_downsize=2,
                 ):
        super(DiffDecoder, self).__init__()
        ndims = len(inshape)
        self.decoder = Decoder(ndims=ndims,
                               nb_conv_per_level=nb_conv_per_level,
                               nb_features=nb_features,
                               max_pool=max_pool,
                               half_res=True,
                               encoder_num=1)

        self.flow_mean = nn.Conv3d(self.decoder.final_nf, ndims, kernel_size=3, stride=1, padding=1, bias=True)
        self.flow_sigma = nn.Conv3d(self.decoder.final_nf, ndims, kernel_size=3, stride=1, padding=1, bias=True)
        self.flow_mean.weight.data.normal_(0., 1e-5)
        self.flow_sigma.weight.data.normal_(0., 1e-10)
        self.flow_sigma.bias.data = torch.Tensor([-10] * 3)

        # resize to full res
        self.fullsize = ResizeTransform(1 / int_downsize, ndims)

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps, unit=False) if int_steps > 0 else None

    def transformer(self):
        return self.integrate.transformer

    def forward(self, inp, is_final=True):
        x = self.decoder(inp, is_final)

        flow_mean = self.flow_mean(x)
        log_sigma = self.flow_sigma(x)
        noise = torch.randn(flow_mean.shape).to(flow_mean.device)
        flow = flow_mean + torch.exp(log_sigma / 2.0) * noise
        flow_pos = self.integrate(flow)
        flow_neg = self.integrate(-flow)
        # resize flow for integration
        flow_pos = self.fullsize(flow_pos)
        flow_neg = self.fullsize(flow_neg)
        return flow_pos, flow_neg, flow_mean, log_sigma


class DiffVoxelMorph(nn.Module):
    def __init__(self, input_shape,
                 input_channel=2,
                 nb_conv_per_level=1,
                 nb_features=(
                         [16, 32, 32, 32],
                         [32, 32, 32, 32, 32, 16]
                 ),
                 int_steps=7,
                 max_pool_size=2,
                 ):
        super().__init__()
        self.tags = ('reg',)
        self.model_id = 'DIFVox'
        ndims = len(input_shape)
        self.encoder = Encoder(ndims=ndims,
                               input_channel=input_channel,
                               nb_conv_per_level=nb_conv_per_level,
                               nb_features=nb_features,
                               max_pool_size=max_pool_size)
        self.vox_decoder = DiffDecoder(inshape=input_shape,
                                       nb_conv_per_level=nb_conv_per_level,
                                       nb_features=nb_features,
                                       max_pool=max_pool_size,
                                       int_steps=int_steps,
                                       int_downsize=2)
        self.transformer = SpatialTransformer(size=input_shape, unit=False)

    def grid(self):
        return self.transformer.get_grid()

    def forward(self, tag, inp, train=False):
        x, y = inp[0], inp[1]
        pos_flow, neg_flow, flow_mean, log_sigma = self.vox_decoder(self.encoder(torch.cat([x, y], dim=1)))
        x2y = self.transformer(x, pos_flow)
        y2x = self.transformer(y, neg_flow)
        if tag == 'infer':
            return [pos_flow, neg_flow], x2y, y2x
        return [pos_flow, neg_flow, flow_mean, log_sigma], x2y, y2x, self.grid()

    def __getitem__(self, item):
        return self


class KLLoss(nn.Module):
    def __init__(self, img_sz, bidirectional=True):
        super(KLLoss, self).__init__()

        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """
        self.bidirectional = bidirectional
        self.image_sigma = 0.02
        self.prior_lambda = 50
        self.ndims = ndims = len(img_sz)
        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)  # 3 3 3
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied
        # ith feature to ith feature
        self.filt = np.zeros([ndims, ndims] + [3] * ndims)  # 3 3 3 3  ##!!!!!!!! in out w h d
        for i in range(ndims):
            self.filt[i, i, ...] = filt_inner  ##!!!!!!!!
        self.filt = torch.from_numpy(self.filt).float()  # 3 3 3 3  ##!!!!!!!!
        self.D = self._degree_matrix([int(x / 2) for x in img_sz])
        # self.D = (self.D).cuda()  # 1, 96, 40,40 3'

    def _degree_matrix(self, vol_shape):
        # get shape stats
        sz = [self.ndims, *vol_shape]  # 96 96 40 3  ##!!!!!!!!

        # prepare conv kernel

        # prepare tf filter
        z = torch.ones([1] + sz)  # 1 96 96 40 3
        strides = [1] * (self.ndims)  ##!!!!!!!!
        return F.conv3d(z, self.filt, padding=1, stride=strides)  ##!!!!!!!!

    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i
        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        # _, _, x, y, z = y_pred.shape
        # norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=y_pred.dtype, device=y_pred.device)
        # norm_vector[0, 0, 0, 0, 0] = z
        # norm_vector[0, 1, 0, 0, 0] = y
        # norm_vector[0, 2, 0, 0, 0] = x
        # y_pred = y_pred * norm_vector

        dy = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
        dx = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
        dz = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]
        return .5 * (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0

    def forward(self, x, y, z2y, y2x, flow_mean, log_sigma):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        # flow_mean = self.res_flow_mean
        # log_sigma = self.res_log_sigma

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims,
        # which is a function of the data

        # sigma terms
        sigma_term = self.prior_lambda * self.D.to(flow_mean.device) * torch.exp(log_sigma) - log_sigma  ##!!!!!!!!
        sigma_term = torch.mean(sigma_term)  ##!!!!!!!!

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(flow_mean)  # this is the jacobi loss
        # combine terms
        if self.bidirectional:
            return self.recon_loss(z2y, y) + self.recon_loss(y2x, x) + \
                0.5 * self.ndims * (sigma_term + prec_term)
        return self.recon_loss(z2y, y) + \
            0.5 * self.ndims * (sigma_term + prec_term)  # ndims because we averaged over dimensions as well

    def recon_loss(self, y_pred, y_true):
        """ reconstruction loss """
        # y_pred = self.warped
        # y_true = self.target
        return 1. / (self.image_sigma ** 2) * torch.mean((y_true - y_pred) ** 2)
