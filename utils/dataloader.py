from params import img_size, spacing, atlas_config, pseudo_config, crop_center, load_data_keys
from torch.utils.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    Orientationd,
    Spacingd,
    SpatialCropd,
    CropForeground
)
from monai.data import (
    Dataset,
    load_decathlon_datalist,
)
from .my_transfomer import *


def train_transforms(task, keys):
    return Compose(
        [
            RandomAtlasPath(keys='atlas', dir_path=atlas_config[task]['dir_path'],
                            activate=atlas_config[task]['activate'],
                            atlas_name=atlas_config[task]['atlas_name']),
            RandomSegPath(keys=keys, dir_path=pseudo_config[task]['dir_path'],
                          activate=pseudo_config[task]['activate']),
            MyLoadImaged(keys=keys),
            ReadLandmarkAsVol(keys=keys),
            RandomPseudo(keys=keys, activate=pseudo_config[task]['activate']),
            EnsureChannelFirstd(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=spacing[task],
                mode=["bilinear"] * len(keys),
            ),
            # CropForegroundd(keys=keys, source_key='moving_seg', select_fn=lambda x: x > 0, margin=20),
            # Resized(keys=keys, spatial_size=img_size[task]),
            SpatialCropd(keys=keys, roi_size=img_size[task], roi_center=crop_center(task)),
            # MyRandFlipd(
            #     keys=keys,
            #     spatial_axis=[0],
            #     prob=0.5,
            # ),
            # MyRandFlipd(
            #     keys=keys,
            #     spatial_axis=[1],
            #     prob=0.5,
            # ),
            # MyRandFlipd(
            #     keys=keys,
            #     spatial_axis=[2],
            #     prob=0.5,
            # ),
            MyScaleIntensityd(keys=keys, minv=0, maxv=1),
            ToTensord(keys=keys),
        ]
    )


def val_transforms(task, keys):
    return Compose(
        [
            RandomAtlasPath(keys='atlas', dir_path=atlas_config[task]['dir_path'],
                            activate=atlas_config[task]['activate'],
                            atlas_name=atlas_config[task]['atlas_name']),
            RandomSegPath(keys=keys, dir_path=pseudo_config[task]['dir_path'],
                          activate=pseudo_config[task]['activate']),
            MyLoadImaged(keys=keys),
            ReadLandmarkAsVol(keys=keys),
            RandomPseudo(keys=keys, activate=pseudo_config[task]['activate']),
            EnsureChannelFirstd(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=spacing[task],
                mode=["bilinear"] * len(keys),
            ),
            SpatialCropd(keys=keys, roi_size=img_size[task], roi_center=crop_center(task)),
            MyScaleIntensityd(keys=keys, minv=0, maxv=1),
            ToTensord(keys=keys),
        ]
    )


def get_datalist(path):
    train_files = load_decathlon_datalist(path, is_segmentation=False, data_list_key="training")
    val_files = load_decathlon_datalist(path, True, data_list_key="validation")
    return train_files, val_files


def get_dataset(task, load_data_keys, data_dir, k_fold, mode):
    if mode == 'train':
        train_trans = train_transforms
        val_trans = val_transforms
    elif mode == 'val':
        train_trans = val_trans = val_transforms
    else:
        raise ValueError

    # if isinstance(contrast, list) or isinstance(contrast, tuple):
    #     keys = [f"moving_{c}" for c in contrast]
    #     keys.extend(["moving_seg", "atlas"])
    # else:
    # if 'PseudoNLin' in data_dir:
    #     keys.insert(-1, 'normal')
    path = os.path.join(data_dir, f'data_fold_{k_fold}.json')
    train_files, val_files = get_datalist(path)
    train_dataset = Dataset(data=train_files,
                            transform=train_trans(task, load_data_keys)
                            )
    val_dataset = Dataset(data=val_files,
                          transform=val_trans(task, load_data_keys))

    return train_dataset, val_dataset


def get_loader(task, load_data_keys, data_dir, k_fold, batch_size, num_worker=0, mode='train'):
    train_dataset, val_dataset = get_dataset(task, load_data_keys, data_dir, k_fold, mode)
    if mode == 'train':
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=True, num_workers=num_worker, pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=False
        )
    return train_loader, val_loader
