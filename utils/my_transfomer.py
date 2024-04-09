import csv
import os
import random
import numpy as np
import torch
from copy import deepcopy
from itertools import chain
from math import floor
from nibabel.affines import apply_affine
from monai.transforms import (
    RandFlipd,
    SpatialPadd,
    NormalizeIntensityd,
    RandomizableTransform,
    InvertibleTransform,
    CenterSpatialCropd,
    BorderPad, SpatialCrop,
    ToTensord,
    LoadImaged, ScaleIntensityd
)
from monai.utils import convert_to_dst_type, TransformBackends, TraceKeys, convert_data_type, Method, fall_back_tuple
from monai.transforms.transform import MapTransform, Transform
from enum import Enum
from monai.transforms.utility.array import ToNumpy
from .utils import read_landmark


###################################################################################################
class RandomAtlasPath(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys, dir_path='Atlas', seed=41, atlas_name='brain', activate=True):
        super(RandomAtlasPath, self).__init__(keys=keys, allow_missing_keys=False)
        random.seed(seed)
        self.atlas = atlas_name
        self.dir_path = dir_path
        self.patients = os.listdir(self.dir_path) if dir_path is not None and os.path.isdir(self.dir_path) else None
        self.activate = activate

    def get_path_dict(self, d):
        if self.patients is None:
            return {self.keys[0]: self.dir_path}
        idx = random.randint(0, len(self.patients) - 1)
        patient = self.patients[idx]
        if not os.path.isdir(os.path.join(self.dir_path, patient)):
            return self.get_path_dict(d)

        meta_dict = [k for k in d.keys() if 'meta_dict' in k]
        # ensure the atlas not same as the original
        for meta in meta_dict:
            patient_name = d[meta]['filename_or_obj'][0].split('/')[-2]
            if patient == patient_name:
                return self.get_path_dict(d)
        return {self.keys[0]: os.path.join(self.dir_path, patient, f'{self.atlas}.nii.gz')}

    def __call__(self, data):
        d = dict(data)
        if self.activate:
            d.update(self.get_path_dict(d))
        return d


class RandomSegPath(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: list, dir_path='Seg', seed=41, activate=True):
        super(RandomSegPath, self).__init__(keys=keys, allow_missing_keys=False)
        random.seed(seed)
        if activate:
            keys.append('lesion')
        self.dir_path = dir_path
        self.patients = os.listdir(self.dir_path)
        self.activate = activate
        print(f'RandomSegPath: {activate}')

    def get_seg_path_dict(self, d):
        idx = random.randint(0, len(self.patients) - 1)
        patient = self.patients[idx]
        if not os.path.isdir(os.path.join(self.dir_path, patient)):
            return self.get_seg_path_dict(d)
        paths = {}
        for child in os.listdir(os.path.join(self.dir_path, patient)):
            if 'seg' in child:
                paths.update({'moving_seg': os.path.join(self.dir_path, patient, child)})
            if 't1' in child:
                paths.update({'lesion': os.path.join(self.dir_path, patient, child)})
        return paths

    def __call__(self, data):
        d = dict(data)
        if self.activate:
            d.update(self.get_seg_path_dict(d))
        return d


class RandomPseudo(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: list, activate=False):
        super(RandomPseudo, self).__init__(keys=keys, allow_missing_keys=False)
        self.activate = activate
        if activate:
            keys.pop(keys.index('lesion'))
        print(f'RandomPseudo: {activate}')

    def create_pseudo(self, d):
        for k in self.keys:
            if 'seg' in k:
                seg = k
            elif 'moving' in k:
                moving = k
        d[moving] = d[moving] * (d[seg] <= 0.5).astype(float) + d['lesion'] * (d[seg] > 0.5).astype(float)
        for k in self.keys:
            if 'lesion' in k:
                del d[k]
        return d

    def __call__(self, data):
        d = dict(data)
        if self.activate:
            d = self.create_pseudo(d)
        return d


class ReadLandmarkAsVol(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    # backend = AddChannel.backend
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.dir = None

    def landmark2vol(self, path, shape):
        x = np.zeros(shape)
        indexs = read_landmark(path)
        for i, index in enumerate(indexs):
            x[index] = 1 + i * 10
        return x

    def __call__(self, data):
        d = dict(data)
        shape = d[self.keys[0]].shape
        for key in self.key_iterator(d):
            if 'landmark' not in key:
                continue
            d[f'{key}_meta_dict'] = d[f'{self.keys[0]}_meta_dict'].copy()
            d[f'{key}_meta_dict']['filename_or_obj'] = d[key]
            d[key] = self.landmark2vol(d[key], shape)

        return d


class MyLoadImaged(LoadImaged):
    def __init__(self, keys):
        super(MyLoadImaged, self).__init__(keys,
                                           reader=None,
                                           dtype=np.float32,
                                           meta_keys=None,
                                           overwriting=False,
                                           image_only=False,
                                           ensure_channel_first=False,
                                           allow_missing_keys=False)

    def __call__(self, data, reader=None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if 'landmarks' in key and 'csv' in d[key]:
                continue
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                if not isinstance(data, np.ndarray):
                    raise ValueError("loader must return a numpy array (because image_only=True was used).")
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d


class MyScaleIntensityd(ScaleIntensityd):
    def __init__(self, keys,
                 minv=0.0,
                 maxv=1.0, ):
        super(MyScaleIntensityd, self).__init__(keys=keys, minv=minv, maxv=maxv)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if 'seg' in key or 'landmark' in key:
                continue
            d[key] = self.scaler(d[key])
        return d


class Keys:
    def set_inv_keys(self, keys):
        self.inv_keys = keys


class MyNormalizeIntensityd(NormalizeIntensityd):
    def __init__(self,
                 keys,
                 subtrahend=None,
                 divisor=None,
                 nonzero=False,
                 channel_wise=False,
                 dtype=np.float32,
                 allow_missing_keys=False):
        super(MyNormalizeIntensityd, self).__init__(
            keys,
            subtrahend,
            divisor,
            nonzero,
            channel_wise,
            dtype,
            allow_missing_keys)
        self.nonzero = nonzero

    def get_slice(self, img):
        if self.nonzero:
            slices = img != 0
        else:
            if isinstance(img, np.ndarray):
                slices = np.ones_like(img, dtype=bool)
            else:
                slices = torch.ones_like(img, dtype=torch.bool)
        return slices

    @staticmethod
    def _mean(x):

        if isinstance(x, np.ndarray):
            return np.mean(x)
        x = torch.mean(x.float())
        return x.item() if x.numel() == 1 else x

    @staticmethod
    def _std(x):
        if isinstance(x, np.ndarray):
            return np.std(x)
        x = torch.std(x.float(), unbiased=False)
        return x.item() if x.numel() == 1 else x

    def mean(self, img):
        slices = self.get_slice(img)
        _sub = self._mean(img[slices])
        if isinstance(_sub, (torch.Tensor, np.ndarray)):
            _sub, *_ = convert_to_dst_type(_sub, img)
            _sub = _sub[slices]
        return _sub

    def std(self, img):
        slices = self.get_slice(img)
        _div = self._std(img[slices])
        if np.isscalar(_div):
            if _div == 0.0:
                _div = 1.0
        elif isinstance(_div, (torch.Tensor, np.ndarray)):
            _div, *_ = convert_to_dst_type(_div, img)
            _div = _div[slices]
            _div[_div == 0.0] = 1.0
        return _div

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[f"{key.split('_')[0]}_norm"] = [self.get_slice(d[key]), self.mean(d[key]), self.std(d[key])]
            d[key] = self.normalizer(d[key])
        return d


class RegMFReplace(RandomizableTransform, InvertibleTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, prob=0.1):
        super(RegMFReplace, self).__init__()
        RandomizableTransform.__init__(self, prob)

    @staticmethod
    def operate(d):
        for k in d.keys():
            if 'moving' in k:
                nk = k.replace('moving', 'fixed')
                if nk in d.keys():
                    temp = d[nk]
                else:
                    temp = d[k]
                d[nk] = d[k]
                d[k] = temp
            if 'fixed' in k:
                nk = k.replace('fixed', 'moving')
                if nk not in d.keys():
                    d[nk] = d[k]
                    d.pop(k)
        return d

    def inverse(self, data):
        d = deepcopy(dict(data))
        transform = self.get_most_recent_transform(d, 'all')
        # Check if random transform was actually performed (based on `prob`)
        if transform[TraceKeys.DO_TRANSFORM]:
            # Inverse is same as forward
            d = self.operate(d)
        # Remove the applied transform
        self.pop_transform(d, 'all')
        return d

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if self._do_transform:
            d = self.operate(d)
        self.push_transform(d, key='all', orig_size=(-1, -1))
        return d


def maximum(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.maximum(a, b)
    return np.maximum(a, b)


class ROICenterCrop(Transform):
    """
    Crop at the roi center of image with specified size.
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
    """

    backend = SpatialCrop.backend

    def __init__(self, roi_size) -> None:
        self.roi_size = roi_size

    def roi_center(self, img):
        data = img[0]
        r = np.any(data, axis=(1, 2))
        c = np.any(data, axis=(0, 2))
        z = np.any(data, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        x_mid = (rmin + rmax) // 2
        y_mid = (cmin + cmax) // 2
        z_mid = (zmin + zmax) // 2
        return x_mid, y_mid, z_mid

    def __call__(self, img, center=None):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        center = self.roi_center(img) if center is None else center
        cropper = SpatialCrop(roi_center=center, roi_size=roi_size)
        return cropper(img)


class ROICenterCropd(MapTransform, InvertibleTransform, Keys):
    backend = ROICenterCrop.backend

    def __init__(self, keys, roi_size, allow_missing_keys=False):
        super(ROICenterCropd, self).__init__(keys, allow_missing_keys)
        self.roi_size = roi_size
        self.cropper = ROICenterCrop(roi_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            orig_size = d[key].shape[1:]
            self.push_transform(d, key, extra_info={'roi_center': self.cropper.roi_center(d[key])}, orig_size=orig_size)
            if 'seg' in key:
                d[key] = self.cropper(d[key], center=self.cropper.roi_center(d['moving_t1']))
            else:
                d[key] = self.cropper(d[key])
        return d

    def inverse(self, data):
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.array(transform[TraceKeys.ORIG_SIZE]).astype(int)
            orig_ori_center = np.array(transform[TraceKeys.EXTRA_INFO]['roi']).astype(int)
            current_size = np.array(d[key].shape[1:])
            current_ori_center = np.array(self.cropper.roi_center(d[key]))
            pad_to_start = np.floor(orig_ori_center - current_ori_center).astype(int)
            pad_to_end = orig_size - current_size - pad_to_start
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class MyCenterSpatialCropd(CenterSpatialCropd, Keys):
    def __init__(self, keys, roi_size, allow_missing_keys=False):
        super(MyCenterSpatialCropd, self).__init__(keys, roi_size, allow_missing_keys)
        self.roi_size = roi_size

    def get_range(self, shape):
        roi_center = [i // 2 for i in shape[1:]]
        roi_center, *_ = convert_data_type(
            data=roi_center, output_type=np.ndarray, dtype=int, wrap_sequence=True
        )
        roi_size, *_ = convert_to_dst_type(src=self.roi_size, dst=roi_center, wrap_sequence=True)
        _zeros = np.zeros_like(roi_center)
        roi_start_torch = maximum(roi_center - np.floor_divide(roi_size, 2), _zeros)  # type: ignore
        roi_end_torch = maximum(roi_start_torch + roi_size, roi_start_torch)
        return roi_start_torch, roi_end_torch

    def landmark_trans(self, img, landmark):
        roi_start_torch, roi_end_torch = self.get_range(img.shape)
        landmark -= roi_start_torch
        if (landmark < 0).any():
            print('landmark cropped')
        return landmark

    def landmark_trans_inv(self, pad_to_start, landmark):
        if landmark.ndim != 2:
            raise RuntimeError
        landmark += pad_to_start
        if (landmark < 0).any():
            print('landmark cropped')
        return landmark

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            orig_size = d[key].shape[1:]
            if f'{key.split("_")[0]}_landmarks' in d.keys():
                d[f'{key.split("_")[0]}_landmarks'] = self.landmark_trans(d[key], d[f'{key.split("_")[0]}_landmarks'])

            d[key] = self.cropper(d[key])
            self.push_transform(d, key, orig_size=orig_size)
        return d

    def inverse(self, data):
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.array(transform[TraceKeys.ORIG_SIZE]).astype(int)
            current_size = np.array(d[key].shape[1:])
            pad_to_start = np.floor((orig_size - current_size) / 2).astype(int)
            # in each direction, if original size is even and current size is odd, += 1
            pad_to_start[np.logical_and(orig_size % 2 == 0, current_size % 2 == 1)] += 1
            pad_to_end = orig_size - current_size - pad_to_start
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            if f'{key.split("_")[0]}_landmarks' in d.keys():
                d[f'{key.split("_")[0]}_landmarks'] = self.landmark_trans_inv(pad_to_start,
                                                                              d[f'{key.split("_")[0]}_landmarks'])
            if f'pre_{key.split("_")[0]}_landmarks' in d.keys():
                d[f'pre_{key.split("_")[0]}_landmarks'] = self.landmark_trans_inv(pad_to_start,
                                                                                  d[
                                                                                      f'pre_{key.split("_")[0]}_landmarks'])

            d[key] = inverse_transform(d[key])
            if key == self.keys[0]:
                for ik in self.inv_keys:
                    d[ik] = inverse_transform(d[ik])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class MySpatialPadd(SpatialPadd, Keys):
    def __init__(self,
                 keys,
                 spatial_size,
                 allow_missing_keys: bool = False,
                 ):
        super(MySpatialPadd, self).__init__(keys,
                                            spatial_size,
                                            allow_missing_keys=allow_missing_keys)

    def determine_data_pad_width(self, img):
        return self.padder._determine_data_pad_width(img.shape[1:])

    def landmark_trans(self, img, landmark):
        all_pad_width = self.determine_data_pad_width(img)
        pad_start_width = np.array([i[0] for i in all_pad_width])
        landmark += pad_start_width
        if (landmark < 0).any():
            print('landmark cropped')
        return landmark

    def get_range(self, roi_center, roi_size):
        roi_center, *_ = convert_data_type(
            data=roi_center, output_type=np.ndarray, dtype=int, wrap_sequence=True
        )
        roi_size, *_ = convert_to_dst_type(src=roi_size, dst=roi_center, wrap_sequence=True)
        _zeros = np.zeros_like(roi_center)
        roi_start_torch = maximum(roi_center - np.floor_divide(roi_size, 2), _zeros)  # type: ignore
        roi_end_torch = maximum(roi_start_torch + roi_size, roi_start_torch)
        return roi_start_torch, roi_end_torch

    def landmark_trans_inv(self, roi_center, roi_size, landmark):
        roi_start_torch, roi_end_torch = self.get_range(roi_center, roi_size)
        landmark -= roi_start_torch
        if (landmark < 0).any():
            print('landmark cropped')
        return landmark

    def __call__(self, data):
        d = dict(data)
        for key, m in self.key_iterator(d, self.mode):
            if f'{key.split("_")[0]}_landmarks' in d.keys():
                d[f'{key.split("_")[0]}_landmarks'] = self.landmark_trans(d[key], d[f'{key.split("_")[0]}_landmarks'])
            self.push_transform(d, key, extra_info={"mode": m.value if isinstance(m, Enum) else m})
            d[key] = self.padder(d[key], mode=m)
        return d

    def inverse(self, data):
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = transform[TraceKeys.ORIG_SIZE]
            if self.padder.method == Method.SYMMETRIC:
                current_size = d[key].shape[1:]
                roi_center = [floor(i / 2) if r % 2 == 0 else (i - 1) // 2 for r, i in zip(orig_size, current_size)]
            else:
                roi_center = [floor(r / 2) if r % 2 == 0 else (r - 1) // 2 for r in orig_size]

            inverse_transform = SpatialCrop(roi_center, orig_size)
            # Apply inverse transform
            if f'{key.split("_")[0]}_landmarks' in d.keys():
                d[f'{key.split("_")[0]}_landmarks'] = self.landmark_trans_inv(roi_center, orig_size,
                                                                              d[f'{key.split("_")[0]}_landmarks'])
            if f'pre_{key.split("_")[0]}_landmarks' in d.keys():
                d[f'pre_{key.split("_")[0]}_landmarks'] = self.landmark_trans_inv(roi_center, orig_size,
                                                                                  d[
                                                                                      f'pre_{key.split("_")[0]}_landmarks'])
            d[key] = inverse_transform(d[key])
            if key == self.keys[0]:
                for ik in self.inv_keys:
                    d[ik] = inverse_transform(d[ik])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class MyRandFlipd(RandFlipd, Keys):
    def __init__(
            self,
            keys,
            prob: float = 0.1,
            spatial_axis=None,
            allow_missing_keys: bool = False,
    ) -> None:
        super(MyRandFlipd, self).__init__(keys,
                                          prob,
                                          spatial_axis,
                                          allow_missing_keys)
        self.spatial_axis = spatial_axis

    def landmark_trans(self, img, landmark):
        shape = img.shape[1:]
        for s in self.spatial_axis:
            landmark[:, s] = shape[s] - landmark[:, s] - 1
        return landmark

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)

        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], randomize=False)
                if f'{key.split("_")[0]}_landmarks' in d.keys():
                    d[f'{key.split("_")[0]}_landmarks'] = self.landmark_trans(d[key],
                                                                              d[f'{key.split("_")[0]}_landmarks'])
            self.push_transform(d, key)
        return d

    def inverse(self, data):
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Inverse is same as forward
                d[key] = self.flipper(d[key], randomize=False)
                if f'{key.split("_")[0]}_landmarks' in d.keys():
                    d[f'{key.split("_")[0]}_landmarks'] = self.landmark_trans(d[key],
                                                                              d[f'{key.split("_")[0]}_landmarks'])
                if f'pre_{key.split("_")[0]}_landmarks' in d.keys():
                    d[f'pre_{key.split("_")[0]}_landmarks'] = self.landmark_trans(d[key],
                                                                                  d[
                                                                                      f'pre_{key.split("_")[0]}_landmarks'])
                if key == self.keys[0]:
                    for ik in self.inv_keys:
                        d[ik] = self.flipper(d[ik], randomize=False)
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class MyToTensord(ToTensord, Keys):
    def __init__(self, keys):
        super(MyToTensord, self).__init__(keys)

    def inverse(self, data):
        d = deepcopy(dict(data))
        for ik in self.inv_keys:
            inverse_transform = ToNumpy()
            d[ik] = inverse_transform(d[ik])

        for key in self.key_iterator(d):
            # Create inverse transform
            inverse_transform = ToNumpy()
            # Apply inverse
            if f'{key.split("_")[0]}_landmarks' in d.keys():
                d[f'{key.split("_")[0]}_landmarks'] = inverse_transform(d[f'{key.split("_")[0]}_landmarks'])
            if f'pre_{key.split("_")[0]}_landmarks' in d.keys():
                d[f'pre_{key.split("_")[0]}_landmarks'] = inverse_transform(d[f'pre_{key.split("_")[0]}_landmarks'])
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class HistogramMatching(MapTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys, matching_key, activate=True):
        from .utils import histogram_matching
        super(HistogramMatching, self).__init__(keys=keys, allow_missing_keys=False)
        self.match_key = matching_key
        self.activate = activate
        self.his_match = histogram_matching

    def __call__(self, data):
        d = dict(data)
        if not self.activate:
            return d
        if self.match_key not in d.keys():
            raise ValueError

        for key in self.key_iterator(d):
            d[key] = self.his_match(d[key], d[self.match_key], channel_axis=None)
        return d


if __name__ == "__main__":
    print('ok')
