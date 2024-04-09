from functools import reduce

import torch
import time
from skimage.segmentation import random_walker

try:
    from .utils import erode, dilate
except ImportError:
    from utils import erode, dilate

import numpy as np


class TypeAdapt:
    """
    Adapt the type of the input image and prediction to the type of the output image.
    """

    def __init__(self, expand=30, crop=True, out_range=None, threshold=0.5):
        self.out = None
        self.start_x = None
        self.start_y = None
        self.start_z = None
        self.end_z = None
        self.end_y = None
        self.end_x = None
        self.shape = None
        self.pytorch = None
        self.expand = expand
        self.crop = crop
        self.out_range = out_range
        self.threshold = threshold

    def input_reformat(self, img, pred):
        self.pytorch = isinstance(img, torch.Tensor)
        self.shape = img.shape
        img = img.squeeze()
        pred = pred.squeeze()
        if self.pytorch:
            img = img.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
        img = self.rescale_intensity(img)
        self.get_size_range(pred)
        if self.crop:
            self.out = pred
            img, pred = self.crop_expand_pos(img, pred)
        return img, pred

    def output_reformat(self, out):
        if self.crop:
            self.out[self.start_x: self.end_x, self.start_y: self.end_y, self.start_z: self.end_z] = out
        else:
            self.out = out

        self.out = self.out.reshape(self.shape)
        if self.pytorch:
            self.out = torch.from_numpy(self.out)
        return self.out

    def get_size_range(self, pred):
        # x = np.any((pred > self.threshold).astype(float), axis=(1, 2))
        # start_x, end_x = np.where(x)[0][[0, -1]]
        #
        # y = np.any((pred > self.threshold).astype(float), axis=(0, 1))
        # start_y, end_y = np.where(y)[0][[0, -1]]
        #
        # z = np.any((pred > self.threshold).astype(float), axis=(0, 2))
        # start_z, end_z = np.where(z)[0][[0, -1]]

        x, y, z = np.where(pred > self.threshold)
        start_x, end_x = x.min(), x.max()
        start_y, end_y = y.min(), y.max()
        start_z, end_z = z.min(), z.max()

        # 扩张
        self.start_x = max(0, start_x - self.expand)
        self.start_y = max(0, start_y - self.expand)
        self.start_z = max(0, start_z - self.expand)

        self.end_x = min(pred.shape[0], end_x + self.expand)
        self.end_y = min(pred.shape[1], end_y + self.expand)
        self.end_z = min(pred.shape[2], end_z + self.expand)

    def crop_expand_pos(self, img, pred):
        # 切割出预测结果部分，减少处理难度
        return img[self.start_x: self.end_x, self.start_y: self.end_y, self.start_z: self.end_z], \
            pred[self.start_x: self.end_x, self.start_y: self.end_y, self.start_z: self.end_z]

    def rescale_intensity(self, x):
        if self.out_range is None:
            return x
        return x * self.out_range[1] + self.out_range[0]

    def shrink(self, x, p=None):
        shape = x.shape
        volume = x.sum()

        x = torch.from_numpy(x).reshape(1, 1, *shape).float()
        while True:
            x = erode(x, 5)
            if x.sum() <= volume * p or x.min() != 0 or x.max() == 0:
                break
        x = x.cpu().squeeze().numpy().reshape(shape)
        return x

    def grid(self, mask):
        _grid = np.zeros_like(mask)
        _grid[..., ::4, ::4, ::4] = 1
        mask = _grid * mask
        return mask


class CRFs(TypeAdapt):
    def __init__(self, crop=True, expand=30, logids=False,
                 sdims1=(1, 1, 1), sdims2=(25, 25, 25), schan=(5,), num=20):
        super().__init__(expand=expand, crop=crop)
        self.num = num
        self.n_labels = 2
        self.sdims1 = sdims1
        self.sdims2 = sdims2
        self.schan = schan
        self.logids = logids

    def get_unary(self, logids):
        from pydensecrf.utils import unary_from_softmax

        if not self.logids:
            unary = np.zeros_like(logids, dtype=np.float32)
            unary[logids == 0] = 0.2
            unary[logids == 1] = 0.8
        else:
            unary = logids
        labels = np.stack((1 - unary, unary), axis=0)
        return unary_from_softmax(labels)

    def __call__(self, img, pred):
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
        img, logids = self.input_reformat(img, pred)
        img = img.astype(np.uint8)
        ###########################
        ###     设置CRF模型     ###
        ###########################
        # 得到一元势（负对数概率）
        d = dcrf.DenseCRF(np.prod(img.shape), self.n_labels)
        U = self.get_unary(logids)

        # 使用 densecrf 类
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=self.sdims1, shape=img.shape)
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=self.sdims2, schan=self.schan, img=img, chdim=-1)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        ####################################
        ###         做推理和计算         ###
        ####################################

        # 进行5次推理
        Q = d.inference(self.num)

        # 找出每个像素最可能的类
        MAP = np.argmax(np.array(Q), axis=0).reshape(img.shape)
        MAP = self.output_reformat(MAP)
        return MAP


class RandomWalk(TypeAdapt):
    def __init__(self, crop=False, expand=30, beta=130, fp=0.4, bp=0.8, kernel_size=5, threshold=.5):
        super().__init__(crop=crop, expand=expand, out_range=None, threshold=threshold)
        self.beta = beta
        self.fp = fp
        self.bp = bp
        self.kernel_size = kernel_size

    def __call__(self, img, pred):
        """
        optimize the mask given by model, erode both the foreground and background of mask,
        then given the final mask by processing the random walk
        :param img:
        :param mark:
        :param beta: Penalization coefficient for the random walker motion
        (the greater beta, the more difficult the diffusion).
        :return:
        """
        if (pred > self.threshold).astype(float).max() == 0:
            print('no mask')
            return pred

        img, pred_np = self.input_reformat(img, pred)
        x = time.time()

        fore = (pred_np > self.threshold).astype(np.float32)
        back = (pred_np <= self.threshold).astype(np.float32)
        if fore.max() == 0:
            import pdb
            pdb.set_trace()
            return

        fore = self.shrink(fore, p=self.fp)
        back = self.shrink(back, p=self.bp)
        mark = np.zeros_like(fore)
        mark[back == 1] = 1
        mark[fore == 1] = 2
        labels = random_walker(img, mark, beta=self.beta, mode='cg_j') - 1
        print(time.time() - x)
        return self.output_reformat(labels)


def region_growing2(batch_size, pb_ori, batch_index, num_steps, input_size):
    import skimage.morphology as mpg
    # initialize the expanded label matrix E
    E = torch.argmax(pb_ori, 1)
    pb_cur, _ = torch.max(torch.softmax(pb_ori, 1), 1)
    # 8-connectivity neighborhood
    delta_r = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    delta_c = np.array([1, 1, 1, 0, 0, -1, -1, -1])
    sample_rate = int(input_size[1] / 8)
    labels_down = np.zeros((batch_size, sample_rate, sample_rate)).astype('uint8')

    ind_expand = 0
    num_expand = 0
    # limit the max iteration of region growing for acceleration
    max_iter = int(10 * ((float(batch_index) / num_steps) ** 0.9))

    # morphological operations for removing the boundary of the expanded annotations
    seed = mpg.square(3)
    se_label = mpg.square(8)
    tau = 0.95
    for id_img in range(batch_size):
        label_cur = labels_down[id_img, :, :]
        label_new = label_cur.copy()
        cur_iter = 0
        is_grow = True
        while is_grow:
            cur_iter += 1
            label_inds = (label_cur < 255) * 1
            erosion = mpg.erosion(label_inds, seed)

            # only need to visit the 8-connectivity neighborhood of the boundary pixels in the annotation
            label_inds = label_inds - erosion
            rc_inds = np.where(label_inds > 0)
            update_count = 0

            # visit the 8-connectivity neighborhood
            for i in range(len(rc_inds[0])):
                y_cur = label_cur[rc_inds[0][i], rc_inds[1][i]]
                for j in range(len(delta_r)):
                    index_r = rc_inds[0][i] + delta_r[j]
                    index_c = rc_inds[1][i] + delta_c[j]

                    valid = (index_r >= 0) & (index_r < sample_rate) & (index_c >= 0) & (index_c < sample_rate)
                    if valid:
                        if (label_new[index_r, index_c] == 255):
                            y_neighbor = E[id_img, index_r, index_c]
                            p_neighbor = pb_cur[id_img, index_r, index_c]
                            if (y_neighbor == y_cur) & (p_neighbor > tau):
                                label_new[index_r, index_c] = y_cur
                                update_count += 1
            if update_count > 0:
                ind_expand += 1
                label_cur = label_new.copy()
                num_expand += update_count
                if cur_iter >= max_iter:
                    is_grow = False
                    labels_down[id_img, :, :] = label_cur
            else:
                is_grow = False
                labels_down[id_img, :, :] = label_cur
    # Expansion Loss
    label_up_interp = torch.nn.Upsample(size=(input_size[1], input_size[0]), mode='nearest')
    labels_downup = label_up_interp(
        torch.from_numpy(labels_down).float().unsqueeze(1)).squeeze().numpy().astype('uint8')
    # remove the boundary pixels in the expanded annotations to reduce noise
    for id_img in range(batch_size):
        labels_downup[id_img] = mpg.dilation(labels_downup[id_img], se_label)
    labels_downup = torch.from_numpy(labels_downup).float().cuda()
    return labels_downup


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import nibabel as nib

    img = nib.load(
        '/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSPseudoHisto/OAS1_0001_MR1-BraTS20_192/brain.nii.gz').get_fdata()
    seg = nib.load(
        '/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSPseudoHisto/OAS1_0001_MR1-BraTS20_192/seg.nii.gz').get_fdata()  # gray scale
    img = (img - img.min()) / (img.max() - img.min())
    seg = (seg > .5).astype(np.uint8)
    # out_img = RegionGrowth(upperThreshold=1,
    #                        lowerThreshold=0,
    #                        neighborMode="6n",
    #                        lowerMargin=2e-2,
    #                        seed_mode='grid',
    #                        update=True)(img,
    #                                     seg)
    out_img = CRFs(crop=True, expand=30, logids=False, sdims1=(1, 1, 1), sdims2=(10, 10, 10), schan=(5,))(img, seg)
    # out_img = RandomWalk(crop=True, expand=30, out_range=(-1, 1), beta=1000, fp=0.4, bp=0.8, kernel_size=5)(img, seg)
    plt.subplot(131)
    sli = 80
    plt.imshow(img.squeeze()[..., sli], cmap='gray')
    plt.subplot(132)
    plt.imshow(seg.squeeze()[..., sli], cmap='gray')
    plt.subplot(133)
    plt.imshow(out_img.squeeze()[..., sli], cmap='gray')
    plt.show()
