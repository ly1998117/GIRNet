import torch
import torch.nn as nn
from .MaskGenerator import MaskG
from .Inpainter import Inpainter2, NaiveInpainter
from .helper import init_net
from .SymNet import SymNet


class BaseNet(nn.Module):
    def __init__(self,
                 device='cpu',
                 single=False,
                 guide_by_rec=True,
                 ntuc=False,
                 gt_seg=False,
                 pap=False,
                 conv_num=1,
                 model_id=5):
        super(BaseNet, self).__init__()
        self.device = device
        self.single = single
        self.model_id = model_id
        self.tags = ('inp', 'gen', 'reg')
        self.guide_by_rec = guide_by_rec
        self.ntuc = ntuc
        self.pap = pap
        self.gt_seg = gt_seg
        self.conv_num = conv_num
        print(f'Model PAP: {pap}')

    def trainG(self, x, y, seg, *params):
        pass

    def trainI(self, x, y, seg, *params):
        pass

    def trainR(self, x, y, seg, *params):
        pass

    def infer(self, x, y, seg, *params):
        with torch.no_grad():
            if self.pap:
                from .SymNet import CompositionTransform, SpatialTransform
                comform = CompositionTransform()
                spaform = SpatialTransform()
                a = params[-1]
                [pos1, neg1], x2a, a2x = self.symnet(x, a, train=False)
                [pos2, neg2], y2a, a2y = self.symnet(y, a, train=False)
                df = comform(pos1, neg2, self.symnet.grid), comform(pos2, neg1, self.symnet.grid)
                return df, spaform(x, df[0], self.symnet.grid), spaform(y, df[1], self.symnet.grid)
            return self.symnet(x, y, train=False)

    def __getitem__(self, tag):
        if tag == 'reg':
            return self.symnet
        elif tag == 'gen':
            return self.mask_generator
        elif tag == 'inp':
            return self.inpainter
        else:
            raise ValueError

    def normalize(self, flow):
        x = flow.clone()
        x = torch.abs(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def __delitem__(self, tag):
        if tag == 'reg':
            del self.symnet
        elif tag == 'gen':
            del self.mask_generator
        elif tag == 'inp':
            del self.inpainter
        else:
            raise ValueError

    def __setitem__(self, tag, model):
        if tag == 'reg':
            self.symnet = model
        elif tag == 'gen':
            self.mask_generator = model
        elif tag == 'inp':
            self.inpainter = model
        else:
            raise ValueError

    def test(self, x):
        self.set_requires_grad(self.mask_generator, True, device=self.device)
        self.set_requires_grad([self.symnet], False, device='cpu')
        self.set_requires_grad([self.inpainter], False, device='cpu')

        mask = self.mask_generator(x)
        return mask

    def to(self, device):
        super().to(device)
        self.device = device

    def set_requires_grad(self, nets, requires_grad=False, device='cpu'):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if nets is None:
            return
        if not isinstance(nets, list):
            nets = [nets]

        [net.to(device) for net in nets]

        for net in nets:
            if net is not None:
                if self.training and requires_grad:
                    net.train()
                else:
                    net.eval()
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, tag, inp, train=True):
        if tag == 'reg':
            return self.trainR(*inp)
        if tag == 'gen':
            return self.trainG(*inp)
        if tag == 'inp':
            return self.trainI(*inp)
        if tag == 'infer':
            return self.infer(*inp)


class GIRNetSimple(BaseNet):
    def __init__(self,
                 input_shape,
                 nb_conv_per_level=1,
                 nb_features=None,
                 range_flow=100,
                 int_steps=7,
                 max_pool_size=2,
                 device='cpu',
                 single=False,
                 norm=False,
                 load_state_path=None
                 ):
        super(GIRNetSimple, self).__init__()
        if nb_features is None:
            nb_features = {
                # unet architecture
                'mask': [
                    [16, 32, 64, 128],
                    [32, 32, 64, 128, 32, 16, 16]
                ]
            }
        self.mask_generator = MaskG(input_shape,
                                    1,
                                    nb_conv_per_level,
                                    nb_features['mask'],
                                    max_pool_size,
                                    norm=norm,
                                    load_state_path=load_state_path
                                    )
        self.inpainter = NaiveInpainter()
        self.symnet = SymNet(input_shape,
                             time_step=int_steps,
                             range_flow=range_flow,
                             )
        self.device = device
        self.single = single
        self.model_id = 5

    def trainG(self, x, y, seg, *params):
        self.set_requires_grad(self.mask_generator, True, device=self.device)
        mask = self.mask_generator(x)
        with torch.no_grad():
            self.set_requires_grad([self.symnet], False, device=self.device)
            [_, _], _, y2x = self.symnet(x, y)
            del _
            self.set_requires_grad([self.symnet], False, device='cpu')
            foreground = x * (mask > 0.5).float()
            background = x * (mask <= 0.5).float()
            inpainted_b = self.inpainter(foreground, (mask <= 0.5).float(), y2x) if not self.single else None
            inpainted_f = self.inpainter(background, (mask > 0.5).float(), y2x)
            del foreground, background, y2x
        return mask, inpainted_b, inpainted_f

    def trainI(self, x, y, seg, *params):
        self.set_requires_grad([self.mask_generator, self.symnet], False, device=self.device)
        with torch.no_grad():
            _, _, y2x = self.symnet(x, y)
            del _
            mask = (self.mask_generator(x) > 0.5).float()
            foreground = x * mask
            background = x * (1 - mask)
            self.set_requires_grad([self.mask_generator], False, device='cpu')
            # This is used for normalization, flow from single image.
            inpainted_b = self.inpainter(foreground, 1 - mask, y2x) if not self.single else None
            inpainted_f = self.inpainter(background, mask, y2x)
        return mask, y2x, inpainted_b, inpainted_f

    def trainR(self, x, y, seg, *params):
        self.set_requires_grad(self.mask_generator, False, device='cpu')
        self.set_requires_grad(self.symnet, True, device=self.device)
        return self.symnet(x, y, train=True)


class GIRNetPseudo(BaseNet):
    def __init__(self,
                 input_shape,
                 nb_conv_per_level=1,
                 nb_features=None,
                 range_flow=100,
                 int_steps=7,
                 max_pool_size=2,
                 device='cpu',
                 single=False,
                 norm=False,
                 load_state_path=None,
                 simple_inp=False,
                 guide_by_rec=True,
                 ntuc=False,
                 gt_seg=False,
                 pap=False,
                 conv_num=1,
                 add_flow_gen=False,
                 fore_inp=True,
                 back_inp=True
                 ):
        """
        :param input_shape:
        :param nb_conv_per_level:
        :param nb_features:
        :param range_flow:
        :param int_steps:
        :param max_pool_size:
        :param init_type:
        :param init_gain:
        :param device:
        :param single:
        :param norm:
        :param load_state_path: pretrained checkpoint path
        :param simple_inp: just copy paste rather that use model
        :param guide_by_rec: Train Registration by inpainted Image
        :param ntuc: Normal Tissue Unchanged, when guide_by_rec=True.
        :param gt_seg: use ground truth mask when train segmentation
        :param pap:
        :param conv_num:
        :param back_y2x: not inpaint back and output atlas2moving directly
        """
        super(GIRNetPseudo, self).__init__(
            device=device,
            single=single,
            guide_by_rec=guide_by_rec,
            ntuc=ntuc,
            gt_seg=gt_seg,
            pap=pap,
        )
        self.add_flow_mask = add_flow_gen
        if add_flow_gen:
            input_channel = 4
        else:
            input_channel = 1
        self.mask_generator = MaskG(shape=input_shape,
                                    input_channel=input_channel,
                                    nb_conv_per_level=nb_conv_per_level,
                                    nb_features=nb_features['mask'],
                                    max_pool_size=max_pool_size,
                                    norm=norm,
                                    load_state_path=load_state_path,
                                    conv_num=conv_num
                                    )
        if load_state_path is None:
            init_net(
                self.mask_generator,
                gpu_ids=device
            )

        self.inpainter = init_net(Inpainter2(shape=input_shape,
                                             input_channel=[2, 1],
                                             nb_conv_per_level=nb_conv_per_level,
                                             nb_features=nb_features['inp'],
                                             max_pool_size=max_pool_size,
                                             norm=norm,
                                             conv_num=conv_num),
                                  gpu_ids=device
                                  ) if not simple_inp else NaiveInpainter()

        self.symnet = SymNet(input_shape,
                             time_step=int_steps,
                             range_flow=range_flow,
                             )
        self.simple_inp = simple_inp
        self.fore_inp = fore_inp
        self.back_inp = back_inp

    def trainG(self, x, y, seg, *params):
        self.set_requires_grad(self.mask_generator, True, device=self.device)
        self.set_requires_grad(self.inpainter, False, device=self.device)
        with torch.no_grad():
            self.set_requires_grad([self.symnet], False, device=self.device)
            [pos, _], x2y, y2x = self.symnet(x, y, train=False)
            del _

        if self.add_flow_mask:
            pos = self.normalize(pos)
            mask = self.mask_generator(x, pos)
        else:
            mask = self.mask_generator(x)

        with torch.no_grad():
            foreground = x * (mask > 0.5).float()
            background = x * (mask <= 0.5).float()
            if self.simple_inp:
                inpainted_b = foreground + y2x * (mask <= 0.5).float()
                inpainted_f = background + y2x * (mask > 0.5).float()
            else:
                inpainted_b = None if self.single else self.inpainter(foreground, (mask <= 0.5).float(),
                                                                      y2x) if self.back_inp else y2x
                inpainted_f = self.inpainter(background, (mask > 0.5).float(), y2x) if self.fore_inp else y2x
                if self.ntuc:
                    inpainted_b = foreground + inpainted_b * (
                            mask <= 0.5).float() if not self.single and self.back_inp else inpainted_b
                    inpainted_f = background + inpainted_f * (mask > 0.5).float() if self.fore_inp else inpainted_f
            del foreground, background
        return mask, inpainted_b, inpainted_f, x2y

    def trainI(self, x, y, seg, *params):
        self.set_requires_grad([self.mask_generator, self.symnet], False, device=self.device)
        self.set_requires_grad(self.inpainter, True, device=self.device)
        with torch.no_grad():
            [pos, _], _, y2x = self.symnet(x, y, train=False)
            del _
            if self.add_flow_mask:
                pos = self.normalize(pos)
                mask = (self.mask_generator(x, pos) > 0.5).float()
            else:
                mask = (self.mask_generator(x) > 0.5).float()
            foreground = x * mask
            background = x * (1 - mask)

        # This is used for normalization, flow from single image.
        inpainted_b = self.inpainter(foreground, 1 - mask, y2x) if not self.single else None
        inpainted_f = self.inpainter(background, mask, y2x)
        return mask, y2x, inpainted_b, inpainted_f

    def trainR(self, x, y, seg, *params):
        self.set_requires_grad([self.mask_generator, self.inpainter], False, device=self.device)
        self.set_requires_grad(self.symnet, True, device=self.device)
        if self.guide_by_rec:
            with torch.no_grad():
                [pos, _], _, y2x = self.symnet(x, y, train=False)
                if self.gt_seg:
                    mask = (seg > .5).float()
                else:
                    if self.add_flow_mask:
                        pos = self.normalize(pos)
                        mask = (self.mask_generator(x, pos) > .5).float()
                    else:
                        mask = (self.mask_generator(x) > .5).float()
                background = x * (mask <= 0.5).float()
                if self.simple_inp:
                    moving = background + y2x * mask
                else:
                    moving = self.inpainter(background, mask, y2x)
                    if self.ntuc:
                        moving = background + moving * mask
                del y2x, background
        else:
            moving = x
        return *self.symnet(x, y, Moving=moving, train=True), mask

