import argparse
from ast import literal_eval


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("--name", default="", type=str, help="First level dir name")

    parser.add_argument("--task", default="MNINLin", type=str, help="task name")

    parser.add_argument("--data_dir", default="", type=str, help="Experiment name")
    parser.add_argument("--output_dir", "-o", default="./result", type=str,
                        help="Root dir name")

    parser.add_argument("--mark", default="", type=str, help="Second level dir name")
    parser.add_argument("--load_data_keys", default="default", type=str, help="MRI contrast(default, normal)")
    parser.add_argument("--contrast", "-c", default="t1", type=str, help="MRI contrast(t1, t1ce, t2, flair)")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_sample", '-ns', type=int, default=1)
    parser.add_argument("--num_works", '-nw', type=int, default=0)
    parser.add_argument("--cross_validation_start", "-s", type=int, default=0)
    parser.add_argument("--cross_validation_end", "-e", type=int, default=5)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--bz", type=int, default=1)
    parser.add_argument("--win", type=int, default=15)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--ncc_eps", type=float, default=0.1)
    parser.add_argument("--smooth", type=float, default=3)
    parser.add_argument("--iem_eps", type=float, default=None)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--weight", type=literal_eval, default=(1, 2, 0))

    parser.add_argument('--ignore', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--load_idx', type=int, default=None)
    parser.add_argument('--single', action='store_true', default=False)
    parser.add_argument('--random_walk', action='store_true', default=False)
    parser.add_argument('--masking', action='store_true', default=False)

    parser.add_argument('--best_tag', default='gen', type=str,
                        help='save the best metrics score (gen|inp|reg) checkpoint')

    parser.add_argument('--train_by_epoch', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False,
                        help="If True, go into test mode, load model for test")

    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument("--dpi", type=int, default=300, help="dpi of plot")
    parser.add_argument("--slices_n", type=int, default=8, help="slices_n of plot")
    parser.add_argument("--format", type=str, default='png', help="format of plot (png | pdf | svg)")

    parser.add_argument('--save_deformation', action='store_true', default=False)
    parser.add_argument('--save_seg', action='store_true', default=False)

    parser.add_argument('--cudnn_nondet', action='store_true',
                        help='disable cudnn determinism - might slow down training')

    parser.add_argument('--int_steps', type=int, default=7,
                        help='number of integration steps (default: 7)')
    parser.add_argument('--int_downsize', type=int, default=2,
                        help='flow downsample factor for integration (default: 2)')

    # loss hyperparameters
    parser.add_argument('--loss', default='ncc',
                        help='image reconstruction loss - can be mse or ncc (default: mse)')
    parser.add_argument("--pretrain", default={None: 0})
    args = parser.parse_args()
    return args


class AnyKey:
    def __init__(self, item):
        self.item = item

    def __getitem__(self, item):
        return self.item


atlas = {
    'False': {
        'activate': False,
        'dir_path': None,
        'atlas_name': None
    },
    'OASIS': {
        'activate': True,
        'dir_path': 'data/OASIS',
        'atlas_name': 'brain',
    },
    'OASISNLin': {
        'activate': True,
        'dir_path': 'data/OASISMNI152NLin',
        'atlas_name': 'brain',
    },
    'NLin400': {
        'activate': True,
        'dir_path': 'data/OASISMNI152NLin/OAS1_0400_MR1/brain.nii.gz',
        'atlas_name': 'brain',
    },
    'MNINLin': {
        'activate': True,
        'dir_path': 'data/MNI152/SymNLinFreeSurfer/MNI152SymNonLinear.nii.gz',
        'atlas_name': 'brain',
    },
    'MNI': {
        'activate': True,
        'dir_path': 'data/MNI152/SymLin/nmi152.nii.gz',
        'atlas_name': 'MNI152',
    },
    'Stroke': {
        'activate': True,
        'dir_path': 'data/StrokeMNI152NLin/sub-r039s002/brain.nii.gz',
        'atlas_name': 'brain',
    }
}

crop_center = lambda key='': (98, 116, 85) if key != 'landmark_ras' else (120, 110, 70)

mask_net = {
    'default': [
        [16, 32, 64, 128],
        [128, 64, 32, 32, 32, 16, 16]
    ],
    'deep': [
        [16, 32, 64, 128, 256],
        [256, 128, 64, 32, 32, 32, 16, 16]
    ],
    'wide': [
        [32, 64, 128, 128],
        [128, 128, 64, 32, 32, 32, 16]
    ]
}
inp_net = {
    'default': [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 32, 16, 16]
    ],
    'wide': [
        [16, 32, 64, 64],
        [64, 64, 32, 32, 32, 16, 16]
    ]
}

####################################################################################################
load_data_keys = {
    'default': ('moving_t1', 'atlas', 'moving_seg'),
    'normal': ('normal', 'atlas', 'moving_seg'),
    'landmark': ('moving_t1', 'fixed_t1', 'atlas', 'moving_landmarks', 'fixed_landmarks'),
    'landmark_ras': ('moving_t1', 'fixed_t1', 'atlas', 'moving_landmarks', 'fixed_landmarks'),
}

atlas_config = {
    '00': atlas['OASIS'], '02': atlas['OASIS'], 'swin2': atlas['OASIS'], 'NLin': atlas['OASISNLin'],
    'MNINLin': atlas['MNINLin'], '01': atlas['MNI'], 'NLin400': atlas['NLin400'],
    'Stroke': atlas['Stroke'], 'landmark': atlas['MNINLin'], 'landmark_ras': atlas['MNINLin'],
}

pseudo_config = {
    '00': {'activate': False, 'dir_path': None},
    'NLin': {'activate': False, 'dir_path': None},
    'MNINLin': {'activate': False, 'dir_path': None},
    'NLin400': {'activate': False, 'dir_path': None},
    'landmark': {'activate': False, 'dir_path': None},
    'landmark_ras': {'activate': False, 'dir_path': None},
    'landmark_atlas': {'activate': False, 'dir_path': None},
    '01': {'activate': False, 'dir_path': None},
    'Stroke': {'activate': False, 'dir_path': None},
    'StrokeMNI': {'activate': False, 'dir_path': None},
    '02': {
        'activate': True,
        'dir_path': 'data/BraTS2020NMI',
    },
}

img_size = AnyKey((160, 192, 144))

spacing = AnyKey((1, 1, 1))

net_config = {
    '00': {
        'mask': mask_net['default'],
        'inp': inp_net['default'],
    },
    'NLin': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    'NLin400': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    'landmark': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    'landmark_ras': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    'landmark_atlas': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    'MNINLin': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    'Stroke': {
        'mask': mask_net['default'],
        'inp': inp_net['wide'],
    },
    '01': {
        'mask': mask_net['wide'],
        'inp': inp_net['default'],
    },

    'gan': {
        'mask': mask_net['default'],
        'inp': mask_net['deep'],
    },
    'swin2': {
        'mask': mask_net['wide'],
        'inp': mask_net['default'],
    }
}

anatomical_structures_label = {
    'BrainStem': [16],
    'Thalamus': [10, 49],
    'CerebellumCortex': [8, 47],
    'LateralVentricle': [4, 43],
    'CerebralWhiteMatter': [2, 41],
    'Putamen': [12, 51],
    'Caudate': [11, 50],
    'Pallidum': [13, 52],
    'Hippocampus': [17, 53],
    '3rd ventricle': [14],
    '4th ventricle': [15],
    'Amygdala': [18, 54],
    'CSF': [24],
    'CerebralCortex': [3, 42]
}


def all_params(task):
    p = {}
    p.update(dict(task=task))
    p.update(dict(img_size=img_size[task]))
    p.update(dict(spacing=spacing[task]))
    p.update(dict(atlas_config=atlas_config[task]))
    p.update(dict(pseudo_config=pseudo_config[task]))
    p.update(dict(net_config=net_config[task]))
    return p
