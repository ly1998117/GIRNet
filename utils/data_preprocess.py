# -*- encoding: utf-8 -*-
"""
@File    :   data_preprocess.py   
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/5 10:42 AM   liu.yang      1.0         None
"""

import os
import random
import time
from multiprocessing import Pool
import nibabel as nib
import scipy.ndimage as ndimg
import numpy as np
import torch
from skimage import morphology
from scipy.ndimage import binary_fill_holes as fill_holes
from monai.transforms import SpatialCrop

try:
    from .utils import align_nonlinear, align, warp2displacementField, apply_warp, read_landmark, \
        get_landmarks_from_nii, write_landmark, histogram_matching
except ImportError:
    from utils import align_nonlinear, align, warp2displacementField, apply_warp, read_landmark, \
        get_landmarks_from_nii, write_landmark, histogram_matching

freesurfer = "/data_smr/liuy/Software/freesurfer/bin"

data_type = {
    'DT_SIGNED_SHORT': np.array(4).astype(np.int16),
    'DT_UINT16': np.array(512).astype(np.int16),
    'DT_FLOAT': np.array(16).astype(np.int16)
}


def mgz2nii(input_path_root, save_path_root):
    import glob
    for filepath in (sorted(glob.glob(input_path_root))):
        img_id = os.path.basename(filepath)
        save_path = os.path.join(save_path_root, img_id)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        names = ["brain", "aseg", "norm"]
        # Stack brain and segmentation
        for name in names:
            mgz = nib.freesurfer.load(os.path.join(filepath, 'mri/%s.mgz' % name))
            data_np = mgz.get_fdata()
            data_array, new_affine = orient('RAS', data_np, mgz.affine)
            new_affine[:, -1] = [0, 0, 0, 1]
            nib.save(nib.Nifti1Image(data_array, new_affine), f"{save_path}/{name}.nii.gz")


def to_affine_nd(r, affine_np, dtype=np.float64):
    """
    Using elements from affine, to create a new affine matrix by
    assigning the rotation/zoom/scaling matrix and the translation vector.

    When ``r`` is an integer, output is an (r+1)x(r+1) matrix,
    where the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(r, len(affine) - 1)`.

    When ``r`` is an affine matrix, the output has the same shape as ``r``,
    and the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(len(r) - 1, len(affine) - 1)`.

    :param r:
    :param affine_np: 2D affine matrix
    :param dtype:
    :return:
    """
    affine_np = affine_np.copy()
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=dtype, copy=True)
    if new_affine.ndim == 0:
        sr: int = int(new_affine.astype(np.uint))
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=dtype)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    return new_affine


def orient(axcodes, data_array, affine):
    """
    original orientation of `data_array` is defined by `affine`.
    :param axcodes:
    :param data_array:
    :param affine: (matrix) (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
    :return: data_array [reoriented in `axcodes`] if `self.image_only`, else
            (data_array [reoriented in `axcodes`], original axcodes, current axcodes).
    """
    labels = (("L", "R"), ("P", "A"), ("I", "S"))
    spatial_shape = data_array.shape
    sr = len(spatial_shape)
    if sr <= 0:
        raise ValueError("data_array must have at least one spatial dimension.")
    affine_: np.ndarray
    if affine is None:
        # default to identity
        affine_np = np.eye(sr + 1, dtype=np.float64)
        affine_ = np.eye(sr + 1, dtype=np.float64)
    else:
        affine_np = affine
        affine_ = to_affine_nd(sr, affine_np)

    src = nib.io_orientation(affine_)
    dst = nib.orientations.axcodes2ornt(axcodes[:sr], labels=labels)
    if len(dst) < sr:
        raise ValueError(
            f"axcodes must match data_array spatially, got axcodes={len(axcodes)}D data_array={sr}D"
        )
    spatial_ornt = nib.orientations.ornt_transform(src, dst)
    new_affine = affine_ @ nib.orientations.inv_ornt_aff(spatial_ornt, spatial_shape)

    axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    if axes:
        data_array = (
            np.flip(data_array, axis=axes)  # type: ignore
        )
    full_transpose = np.arange(len(data_array.shape))
    full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    if not np.all(full_transpose == np.arange(len(data_array.shape))):
        data_array = data_array.transpose(full_transpose)  # type: ignore
    new_affine = to_affine_nd(affine_np, new_affine)
    return data_array, new_affine


def skull_remove(img, seg, o_img, o_seg=None):
    """
    using brain mask to remove the skull
    :param params:
    :return:
    """
    img_nii = nib.load(img)
    seg_nii = nib.load(seg)
    img_np = img_nii.get_fdata()
    seg_np = seg_nii.get_fdata()

    img_np, new_img_affine = orient("RAS", img_np, img_nii.affine)
    seg_np, new_seg_affine = orient("RAS", seg_np, seg_nii.affine)
    new_img_affine[:, -1] = [0, 0, 0, 1]
    new_seg_affine[:, -1] = [0, 0, 0, 1]

    print('processing: ', id)
    print('seg max id: ', seg_np.max(), 'seg min id: ', seg_np.min())
    print('total class: ', np.unique(seg_np).shape[0])
    print(np.unique(seg_np))
    print('---------------------')
    mask_np = np.zeros(img_np.shape)
    mask_np[seg_np < 0.015] = 0
    mask_np[seg_np >= 0.015] = 1
    mask_np = mask_np.astype(bool)
    # bounder expand
    mask_np = ndimg.binary_dilation(mask_np, iterations=1)
    mask_np = fill_holes(mask_np)
    for i in range(mask_np.shape[0]):
        mask_np[i, :, :] = morphology.remove_small_objects(mask_np[i, :, :], 15, 2)
    for i in range(mask_np.shape[1]):
        mask_np[:, i, :] = morphology.remove_small_objects(mask_np[:, i, :], 15, 2)
    for i in range(mask_np.shape[2]):
        mask_np[:, :, i] = morphology.remove_small_objects(mask_np[:, :, i], 15, 2)

    mask_np = mask_np.astype(np.float32)
    img_np = img_np * mask_np

    nib.save(nib.Nifti1Image(img_np, new_img_affine), o_img)
    if o_seg:
        nib.save(nib.Nifti1Image(mask_np, new_seg_affine), o_seg)


##################################### Multi Tread function  ###########################################
def _affine_brain_transonly(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    commands = [
        os.path.join(freesurfer, 'mri_robust_register'),
        "--mov", img,
        "--dst", template,
        "--maskdst", template_mask,
        "--affine", "--iscale", "--satit", "--transonly",
        "--mapmov", o_img,
        "--lta", o_lta
    ]
    command = " ".join(commands)
    os.system(command)


def _affine_brain(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    commands = [
        os.path.join(freesurfer, 'mri_robust_register'),
        "--mov", img,
        "--dst", template,
        "--maskdst", template_mask,
        "--affine", "--iscale", "--satit",
        "--mapmov", o_img,
        "--lta", o_lta
    ]
    command = " ".join(commands)
    os.system(command)


def _affine_seg(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    commands = [os.path.join(freesurfer, 'mri_vol2vol'),
                "--lta", o_lta,
                "--targ", template_mask,
                "--mov", seg,
                "--nearest",
                "--o", o_seg
                ]
    command = " ".join(commands)
    os.system(command)


def _change_datatype(params):
    """
    using brain mask to remove the skull
    :param params:
    :return:
    """
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    data_dir = os.path.split(img)[0]
    for file in os.listdir(data_dir):
        if 'nii.gz' in file:
            file_path = os.path.join(data_dir, file)
            nii = nib.load(file_path)
            if nii.header['datatype'] == data_type['DT_FLOAT'] and \
                    isinstance(nii.header['datatype'].dtype, type(data_type['DT_FLOAT'].dtype)):
                continue
            print(file_path)
            nii.header['datatype'] = data_type['DT_FLOAT']
            try:
                nii = nib.Nifti1Image(nii.get_fdata().astype(float), nii.affine, nii.header)
                nib.save(nii, file_path)
            except:
                print('error: ', file_path)


def _affine_brain_linear_fsl(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    normal = os.path.join(os.path.split(img)[0], 'oasis_brain.nii.gz')
    o_dir_path = os.path.split(o_img)[0]
    in_mask = os.path.join(o_dir_path, 'in_mask.nii.gz')
    mask = nib.load(seg)
    mask_np = mask.get_fdata()
    mask_np = (mask_np > 0.1).astype(mask_np.dtype)
    nib.save(nib.Nifti1Image(1 - mask_np, mask.affine), in_mask)
    del mask, mask_np
    time.sleep(0.5)
    # No Cost Function Masking
    oaff_l = os.path.join(o_dir_path, 'transform.mat')
    oaff_n = os.path.join(o_dir_path, 'transform_normal.mat')

    out_l = os.path.join(o_dir_path, 'brain2atlas_linear.nii.gz')
    out_n = os.path.join(o_dir_path, 'normal2atlas_linear.nii.gz')

    # linear registration to get affine matrx, and
    align(inp=img, ref=template, out=out_l, oaff=oaff_l, cost='corratio')
    align(inp=normal, ref=template, out=out_n, oaff=oaff_n, cost='corratio')

    # Cost Function Masking
    oaff_masking = os.path.join(o_dir_path, 'transform_masking.mat')
    out_masking = os.path.join(o_dir_path, 'brain2atlas_linear_masking.nii.gz')
    align(inp=img, ref=template, out=out_masking, oaff=oaff_masking, inweight=in_mask, cost='corratio')


def _affine_brain_nonlinear_fsl(params):
    # run linear first to get the affine matrix
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    normal = os.path.join(os.path.split(img)[0], 'oasis_brain.nii.gz')
    o_dir_path = os.path.split(o_img)[0]
    in_mask = os.path.join(o_dir_path, 'in_mask.nii.gz')
    assert os.path.isfile(in_mask), "please run affine_brain_linear_fsl first"
    # No Cost Function Masking
    warp = os.path.join(o_dir_path, 'warp.nii.gz')
    warp_n = os.path.join(o_dir_path, 'warp_normal.nii.gz')

    o_img = os.path.join(o_dir_path, 'brain2atlas.nii.gz')
    o_img_n = os.path.join(o_dir_path, 'normal2atlas.nii.gz')

    oaff = os.path.join(o_dir_path, 'transform.mat')
    oaff_n = os.path.join(o_dir_path, 'transform_normal.mat')

    field = os.path.join(o_dir_path, 'field.nii.gz')
    field_n = os.path.join(o_dir_path, 'field_normal.nii.gz')

    align_nonlinear(inp=img, ref=template, ref_mask=template_mask, aff=oaff,
                    out=o_img, warp=warp)
    warp2displacementField(warp=warp, ref=template, field=field)

    align_nonlinear(inp=normal, ref=template, ref_mask=template_mask, aff=oaff_n,
                    out=o_img_n, warp=warp_n)
    warp2displacementField(warp=warp_n, ref=template, field=field_n)

    # Cost Function Masking
    warp_masking = os.path.join(o_dir_path, 'warp_masking.nii.gz')
    o_img_masking = os.path.join(o_dir_path, 'brain2atlas_masking.nii.gz')
    oaff_masking = os.path.join(o_dir_path, 'transform_masking.mat')
    field_masking = os.path.join(o_dir_path, 'field_masking.nii.gz')

    align_nonlinear(inp=img, ref=template, ref_mask=template_mask, aff=oaff_masking,
                    in_mask=in_mask, out=o_img_masking, warp=warp_masking)
    warp2displacementField(warp=warp_masking, ref=template, field=field_masking)


def _affine_brain_nonlinear_fsl_seg(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    aseg = os.path.join(os.path.split(img)[0], 'aseg.nii.gz')
    o_dir_path = os.path.split(o_img)[0]
    in_mask = os.path.join(o_dir_path, 'in_mask.nii.gz')
    assert os.path.isfile(in_mask), "please run affine_brain_linear_fsl first"
    # No Cost Function Masking
    warp = os.path.join(o_dir_path, 'warp.nii.gz')
    warp_n = os.path.join(o_dir_path, 'warp_normal.nii.gz')
    warp_masking = os.path.join(o_dir_path, 'warp_masking.nii.gz')

    oaff = os.path.join(o_dir_path, 'transform.mat')
    oaff_n = os.path.join(o_dir_path, 'transform_normal.mat')
    oaff_masking = os.path.join(o_dir_path, 'transform_masking.mat')

    normal2yaseg = os.path.join(o_dir_path, 'normal2yaseg.nii.gz')
    x2yaseg = os.path.join(o_dir_path, 'x2yaseg.nii.gz')
    x2yaseg_masking = os.path.join(o_dir_path, 'x2yaseg_masking.nii.gz')
    for out, warp, xfm in zip([normal2yaseg, x2yaseg, x2yaseg_masking],
                              [warp_n, warp, warp_masking],
                              [oaff_n, oaff, oaff_masking]):
        apply_warp(ref=template, inp=aseg, out=out, xfm=xfm, warp=warp)


def _skull_remove_with_mask(params):
    """
    using brain mask to remove the skull
    :param params:
    :return:
    """
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    out_dir = os.path.split(o_img)[0]
    skull_remove(img, template_mask, os.path.join(out_dir, 'brain.nii.gz'))
    seg_nii = nib.load(seg)
    seg_np, new_seg_affine = orient("RAS", seg_nii.get_fdata(), seg_nii.affine)
    new_seg_affine[:, -1] = [0, 0, 0, 1]
    nib.save(nib.Nifti1Image(seg_np, new_seg_affine), os.path.join(out_dir, 'seg.nii.gz'))


def _skull_remove_with_freesurfer(params):
    """
    freesurfer tools: [SynthStrip]
    SynthStrip: Skull-Stripping for Any Brain Image
    Andrew Hoopes, Jocelyn S. Mora, Adrian V. Dalca, Bruce Fischl , Malte Hoffmann
    NeuroImage 260, 2022, 119474. DOI: 10.1016/j.neuroimage.2022.119474

    :param params:
    :return:
    """
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    out_dir = os.path.split(o_img)[0]
    command = f"cp {seg} {os.path.join(out_dir, 'seg.nii.gz')}"
    os.system(command)

    commands = [
        os.path.join(freesurfer, 'mri_synthstrip'),
        "-i", img,
        "-o", os.path.join(out_dir, 'brain.nii.gz'),
        "-m", os.path.join(out_dir, 'whole_brain_mask.nii.gz')
    ]
    command = " ".join(commands)
    os.system(command)


def _orient2RAS(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    img_nii = nib.load(img)
    seg_nii = nib.load(seg)
    img_np = img_nii.get_fdata()
    seg_np = seg_nii.get_fdata()

    img_np, new_img_affine = orient("RAS", img_np, img_nii.affine)
    seg_np, new_seg_affine = orient("RAS", seg_np, seg_nii.affine)
    new_img_affine[:, -1] = [0, 0, 0, 1]
    new_seg_affine[:, -1] = [0, 0, 0, 1]
    nib.save(nib.Nifti1Image(img_np, new_img_affine), o_img)
    nib.save(nib.Nifti1Image(seg_np, new_seg_affine), o_seg)
    print(f'{os.path.basename(img)} done')


def simple_pseudo(params):
    """
    create Pseudo dataset: (Lesions of BraTS paste to OASIS)
    :param params:
    :return:
    """
    bra_brain_path, bra_seg_path, oasis_brain_path, oasis_seg_path, output_dir = params
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"cp {oasis_seg_path} {os.path.join(output_dir, f'aseg.nii.gz')}")
    if bra_brain_path is not None and len(bra_brain_path) != 0:
        os.system(f"cp {bra_seg_path} {os.path.join(output_dir, f'seg.nii.gz')}")
        os.system(f"cp {bra_brain_path} {os.path.join(output_dir, f'bra_brain.nii.gz')}")
        os.system(f"cp {oasis_brain_path} {os.path.join(output_dir, f'oasis_brain.nii.gz')}")
        seg_data = nib.load(bra_seg_path).get_fdata()
        p_brain = nib.load(bra_brain_path).get_fdata()
        o_brain = nib.load(oasis_brain_path)
        o_data = o_brain.get_fdata()
        # his_matching
        p_brain = histogram_matching(p_brain, o_data)
        o_data[seg_data > .5] = p_brain[seg_data > .5]
        nib.save(nib.Nifti1Image(o_data, o_brain.affine), os.path.join(output_dir, f'brain.nii.gz'))
    print('done')


def _volume_copy2targetPath(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    command = f"cp {img} {o_img}"
    os.system(command)

    command = f"cp {seg} {o_seg}"
    os.system(command)


def _crop(cropper, img_path, o_img):
    img_nii = nib.load(img_path)
    img_np = img_nii.get_fdata()
    img_np, new_img_affine = orient("RAS", img_np, img_nii.affine)
    img_np = img_np.reshape(1, *img_np.shape)
    img_np = cropper(img_np)
    img_np = img_np.squeeze()
    if isinstance(img_np, torch.Tensor):
        img_np = img_np.numpy()
    img_np, new_img_affine = orient("RAS", img_np, img_nii.affine)
    new_img_affine[:, -1] = [0, 0, 0, 1]
    nib.save(nib.Nifti1Image(img_np, new_img_affine), o_img)


def _center_crop(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    dir_path = os.path.split(img)[0]
    o_dir_path = os.path.split(o_img)[0]
    file_names = os.listdir(dir_path)
    cropper = SpatialCrop(roi_size=(160, 192, 144), roi_center=(98, 116, 85))

    for f in file_names:
        if '.nii.gz' in f:
            img_path, o_img = os.path.join(dir_path, f), os.path.join(o_dir_path, f)
            _crop(cropper, img_path, o_img)
    print(f'{os.path.basename(img)} done')


def _landmark_ras_nii(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    dir_path = os.path.split(img)[0]
    o_dir_path = os.path.split(o_img)[0]
    pre_name = post_name = None
    for d in os.listdir(dir_path):
        if 't1.nii.gz' in d:
            if '0000' in d:
                pre_name = ['_'.join(d.split('_')[:-1]), 'pre']
            else:
                post_name = ['_'.join(d.split('_')[:-1]), 'post']
    for old_name, new_name in (pre_name, post_name):
        img_nii = nib.load(os.path.join(dir_path, f'{old_name}_t1.nii.gz'))
        img_np, affine = img_nii.get_fdata(), img_nii.affine

        landmark = read_landmark(os.path.join(dir_path, f"{old_name}_landmarks.csv"))
        landmark = np.array(landmark) * np.array([[-1, -1, 1]])
        coordinates = nib.affines.apply_affine(affine, landmark).astype(int)

        for k, i in enumerate(coordinates):
            landmark = np.zeros_like(img_np)
            # for a in range(-1, 2):
            #     for b in range(-1, 2):
            #         for c in range(-1, 2):
            #             landmark[i[0] + a, i[1] + b, i[2] + c] = 128
            landmark[i[0], i[1], i[2]] = 255
            landmark, new_land_affine = orient("RAS", landmark, affine)
            new_land_affine[:, -1] = [0, 0, 0, 1]
            coordinates[k] = list(zip(*np.where(landmark == 255)))[0]
            nib.save(nib.Nifti1Image(landmark, new_land_affine),
                     os.path.join(o_dir_path,
                                  os.path.join(o_dir_path, f"{new_name}_landmarks_{k}.nii.gz")))

        write_landmark(coordinates, os.path.join(o_dir_path, f'{new_name}_landmarks.csv'))
        img_np, new_img_affine = orient("RAS", img_np, affine)
        new_img_affine[:, -1] = [0, 0, 0, 1]
        nib.save(nib.Nifti1Image(img_np, new_img_affine), os.path.join(o_dir_path, f'{new_name}_t1.nii.gz'))


def _landmark2nii_with_failed(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    dir_path = os.path.split(img)[0]
    o_dir_path = os.path.split(o_img)[0]
    _landmark_ras_nii(params)
    idx = os.path.split(dir_path)[-1].split('_')[-1]
    failed = {'010': ['post', 'pre'], '054': ['pre', 'post'], '097': ['post', 'pre'], '134': ['pre', 'post']}
    if idx in failed.keys():
        mov = os.path.join(o_dir_path, f'{failed[idx][0]}_t1.nii.gz')
        dst = os.path.join(o_dir_path, f'{failed[idx][1]}_t1.nii.gz')
        lta = os.path.join(o_dir_path, f'o.lta')
        commands = [
            os.path.join(freesurfer, 'mri_robust_register'),
            "--mov", mov,
            "--dst", dst,
            "--affine", "--iscale", "--satit", "--transonly",
            "--mapmov", mov,
            "--lta", lta
        ]
        command = " ".join(commands)
        os.system(command)
        for d in os.listdir(o_dir_path):
            if f'{failed[idx][0]}_landmarks_' in d:
                commands = [os.path.join(freesurfer, 'mri_vol2vol'),
                            "--lta", lta,
                            "--targ", dst,
                            "--mov", os.path.join(o_dir_path, d),
                            "--nearest",
                            "--o", os.path.join(o_dir_path, d)
                            ]
                command = " ".join(commands)
                os.system(command)
                seg = nib.load(os.path.join(o_dir_path, d))
                seg, affine = seg.get_fdata(), seg.affine
                seg[seg < seg.max()] = 0
                landmark = get_landmarks_from_nii(seg)
                if len(landmark) == 0:
                    print(f'[{idx} ERROR]: ', failed[idx], d)
                    continue
                else:
                    landmark = landmark[0]
                seg = np.zeros_like(seg)
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            seg[landmark[0] + a, landmark[1] + b, landmark[2] + c] = 128
                seg[landmark] = 255
                nib.save(nib.Nifti1Image(seg, affine), os.path.join(o_dir_path, d))


def _nii2landmark(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    dir_path = os.path.split(img)[0]
    o_dir_path = os.path.split(o_img)[0]
    pre_landmark_tag = 'pre_landmarks_'
    post_landmark_tag = 'post_landmarks_'
    pre = []
    post = []

    for name in sorted(os.listdir(dir_path)):
        if pre_landmark_tag in name and 'csv' not in name:
            idx = name.split('_')[-1].split('.')[0]
            pre_landmark = nib.load(os.path.join(dir_path, name)).get_fdata()
            post_landmark = nib.load(os.path.join(dir_path, post_landmark_tag + idx + '.nii.gz')).get_fdata()
            pre_landmark[pre_landmark < pre_landmark.max()] = 0
            post_landmark[post_landmark < pre_landmark.max()] = 0

            pre_landmark = get_landmarks_from_nii(pre_landmark)
            post_landmark = get_landmarks_from_nii(post_landmark)

            if len(pre_landmark) != 0 and len(post_landmark) != 0:
                pre.append(pre_landmark[0])
                post.append(post_landmark[0])
            write_landmark(pre_landmark, os.path.join(o_dir_path, pre_landmark_tag + f'{idx}.csv'))
            write_landmark(post_landmark, os.path.join(o_dir_path, post_landmark_tag + f'{idx}.csv'))

            os.system(f'rm {os.path.join(dir_path, name)}')
            os.system(f'rm {os.path.join(o_dir_path, post_landmark_tag + idx + ".nii.gz")}')
    if len(pre) == 0 or len(post) == 0:
        print(o_dir_path, 'Empty')
    write_landmark(pre, os.path.join(o_dir_path, f'pre_landmarks.csv'))
    write_landmark(post, os.path.join(o_dir_path, f'post_landmarks.csv'))


def _affine_landmark_seg(params):
    img, seg, o_img, o_seg, o_lta, template, template_mask = params
    dir_path = os.path.split(img)[0]
    o_dir_path = os.path.split(o_img)[0]

    mode = os.path.split(img)[-1].split('_')[0]
    for name in os.listdir(dir_path):
        if f'{mode}_landmarks_' in name:
            commands = [os.path.join(freesurfer, 'mri_vol2vol'),
                        "--lta", os.path.join(o_dir_path, f'{mode}.lta'),
                        "--targ", template_mask,
                        "--mov", os.path.join(dir_path, name),
                        "--nearest",
                        "--o", os.path.join(o_dir_path, name)
                        ]
            command = " ".join(commands)
            os.system(command)


###################################################################################################
class MultiThreadPreprocess:
    def __init__(self, input_path='/data_smr/liuy/Challenge/PennBraTSReg/data_jRRS/BraTSSeg',
                 output_path='/data_smr/liuy/Challenge/PennBraTSReg/data_jRRS/BraTSSegmni152',
                 template_path='/data_smr/liuy/Challenge/PennBraTSReg/DataProcessingForRegistration/DatasetsProcess/mni152/mni152_LPS.nii.gz',
                 template_mask_path='/data_smr/liuy/Challenge/PennBraTSReg/DataProcessingForRegistration/DatasetsProcess/mni152/mni152_mask_LPS.nii.gz',
                 brain_condition=None,
                 seg_condition=None,
                 skip_condition=lambda x: False,
                 outputdir_fn=None,
                 processes=128):
        """
        :param input_path:
        :param output_path:
        :param template_path: atlas
        :param template_mask_path: atlas mask
        :param brain_condition: condition function to get file name you need
        :param seg_condition: condition function to get file name you need
        :param skip_condition: condition function to skip the dir you want, when deep first search the dir
        :param processes: number of threads
        """
        self.processes = processes
        self.image_paths = []
        self.seg_paths = []
        self.output_nii_paths = []
        self.output_seg_paths = []
        self.output_lta_paths = []
        self.template_paths = [template_path]
        self.template_mask_paths = [template_mask_path]

        def op(filename, dir_path):
            if brain_condition(filename):
                if outputdir_fn is None:
                    output_dir = os.path.join(output_path, dir_path.split(input_path)[-1].split('/')[-1])
                else:
                    output_dir = os.path.join(output_path, outputdir_fn(dir_path.split(input_path)[-1]))

                os.makedirs(output_dir, exist_ok=True)
                self.image_paths.append(os.path.join(dir_path, filename))
                self.output_nii_paths.append(os.path.join(output_dir, filename))
                self.output_lta_paths.append(os.path.join(output_dir, f'{"_".join(filename.split("_")[:-1])}.lta'))

            if seg_condition(filename):
                if outputdir_fn is None:
                    output_dir = os.path.join(output_path, dir_path.split(input_path)[-1].split('/')[-1])
                else:
                    output_dir = os.path.join(output_path, outputdir_fn(dir_path.split(input_path)[-1]))

                os.makedirs(output_dir, exist_ok=True)
                self.seg_paths.append(os.path.join(dir_path, filename))
                self.output_seg_paths.append(os.path.join(output_dir, filename))

        self.deepsearch(input_path, op, skip_condition=skip_condition)
        self.template_paths *= len(self.image_paths)
        self.template_mask_paths *= len(self.image_paths)

    def deepsearch(self, dir_path, op, skip_condition=lambda x: False):
        """
        deep search dir, find target file
        :param dir_path:
        :param op: an interface
        :param skip_condition: function that decide to skip current dir
        :return:
        """
        for sub_dir in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, sub_dir)):
                if skip_condition(os.path.join(dir_path, sub_dir)):
                    continue
                self.deepsearch(os.path.join(dir_path, sub_dir), op, skip_condition)
            else:
                op(sub_dir, dir_path)

    def __call__(self, fn):
        p = Pool(self.processes)
        p.map(fn, zip(
            self.image_paths,
            self.seg_paths,
            self.output_nii_paths,
            self.output_seg_paths,
            self.output_lta_paths,
            self.template_paths,
            self.template_mask_paths)
              )
        p.close()
        p.join()


def brats2mni152():
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020/MICCAI_BraTS2020_TrainingData',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020RAS',
        brain_condition=lambda x: 't1.nii.gz' in x,
        seg_condition=lambda x: 'seg.nii.gz' in x,
        processes=128
    )
    mp(_orient2RAS)
    del mp
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020RAS',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020MNI152NLin',
        template_path=template_path,
        template_mask_path=template_mask_path,
        brain_condition=lambda x: 't1.nii.gz' in x,
        seg_condition=lambda x: 'seg.nii.gz' in x,
        processes=128,
    )
    mp(_affine_brain)
    mp(_affine_seg)
    os.system('rm -r /data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020RAS')


def oasis2mni152():
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISNII',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISRAS',
        brain_condition=lambda x: 'brain.nii.gz' == x,
        seg_condition=lambda x: 'aseg.nii.gz' == x,
        processes=128
    )
    mp(_orient2RAS)
    del mp
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISRAS',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISMNI152NLin',
        template_path=template_path,
        template_mask_path=template_mask_path,
        brain_condition=lambda x: x == 'brain.nii.gz',
        seg_condition=lambda x: x == 'aseg.nii.gz',
        processes=128
    )
    mp(_affine_brain)
    mp(_affine_seg)
    os.system('rm -r /data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISRAS')


def stroke2mni152():
    # mp = MultiThreadPreprocess(
    #     input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeSkullByMask',
    #     output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeSkullRAS',
    #     brain_condition=lambda x: x == 'brain.nii.gz',
    #     seg_condition=lambda x: x == 'seg.nii.gz',
    #     processes=128
    # )
    # mp(_orient2RAS)
    # print('done')

    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeSkullByMask',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeMNI152NLin',
        template_path=template_path,
        template_mask_path=template_mask_path,
        brain_condition=lambda x: x == 'brain.nii.gz',
        seg_condition=lambda x: x == 'seg.nii.gz',
        processes=128
    )
    mp(_affine_brain)
    mp(_affine_seg)
    print('finish')


def skull_strip(freesurfer=False):
    if freesurfer:
        mp = MultiThreadPreprocess(
            input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeORI/Training',
            output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeSkullByFreesurfer',
            template_path=template_path,
            template_mask_path=template_mask_path,
            brain_condition=lambda x: 'MNI152NLin2009aSym_T1w.nii.gz' in x,
            seg_condition=lambda x: 'label-L_desc-T1lesion_mask.nii.gz' in x,
            outputdir_fn=lambda dir_path: dir_path.split('/')[-3],
            processes=128
        )
        mp(_skull_remove_with_freesurfer)
    else:
        mp = MultiThreadPreprocess(
            input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeORI/Training',
            output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeSkullByMask',
            template_path=template_path,
            template_mask_path=template_mask_path,
            brain_condition=lambda x: 'MNI152NLin2009aSym_T1w.nii.gz' in x,
            seg_condition=lambda x: 'label-L_desc-T1lesion_mask.nii.gz' in x,
            outputdir_fn=lambda dir_path: dir_path.split('/')[-3],
            processes=128
        )
        mp(_skull_remove_with_mask)


def create_pseudo(bra_dir='../data/BraTS2020MNI152NLin', oasis_dir='../data/OASISMNI152NLin',
                  stroke_dir='../data/StrokeMNI152NLin', target_bp_path=None, target_sp_path=None,
                  processes=64, num=None):
    # target_bp_path = os.path.split(bra_dir)[0] + '/BraTSPseudo'
    # target_sp_path = os.path.split(bra_dir)[0] + '/StrokePseudo'
    oasis_list = list(os.listdir(oasis_dir))
    random.shuffle(oasis_list)

    if target_bp_path:

        bra_seg_paths = []
        bra_brain_paths = []

        oasis_brain_paths = []
        oasis_seg_paths = []
        output_bp_dirs = []

        if num:
            bra_list = ['_'.join(s.split('_')[:-1]) for s in
                        open(os.path.join(bra_dir, 'volume.txt')).read().splitlines()[:num]]
        else:
            bra_list = list(os.listdir(bra_dir))
        random.shuffle(bra_list)

        for b, o in zip(bra_list, oasis_list):
            # if b is None:
            #     b = bra_list[random.randint(0, len(bra_list) - 1)]
            bra_seg_paths.append(os.path.join(bra_dir, b, f'{b}_seg.nii.gz'))
            bra_brain_paths.append(os.path.join(bra_dir, b, f'{b}_t1.nii.gz'))
            oasis_brain_paths.append(os.path.join(oasis_dir, o, f'brain.nii.gz'))
            oasis_seg_paths.append(os.path.join(oasis_dir, o, f'aseg.nii.gz'))

            output_bp_dirs.append(os.path.join(target_bp_path, f'{o}-{b.replace("_Training", "")}'))
        p = Pool(processes)
        p.map(simple_pseudo, zip(bra_brain_paths, bra_seg_paths, oasis_brain_paths, oasis_seg_paths, output_bp_dirs))
        p.close()
        p.join()

    if target_sp_path:
        oasis_brain_paths = []
        stroke_seg_paths = []
        stroke_brain_paths = []
        output_sp_dirs = []
        stroke_volume = open(os.path.join(stroke_dir, 'volume.txt')).read().splitlines()[:140]
        stroke_list = [s.split('_')[0] for s in stroke_volume]
        random.shuffle(stroke_list)

        for s, o in zip(stroke_list, oasis_list):
            # if s is None:
            #     s = stroke_list[random.randint(0, len(stroke_list) - 1)]
            # if o is None:
            #     continue
            stroke_brain_paths.append(os.path.join(stroke_dir, s, f'brain.nii.gz'))
            stroke_seg_paths.append(os.path.join(stroke_dir, s, f'seg.nii.gz'))
            oasis_brain_paths.append(os.path.join(oasis_dir, o, f'brain.nii.gz'))
            output_sp_dirs.append(os.path.join(target_sp_path, f'{o}-{s.replace("sub-", "")}'))

        p = Pool(processes)
        p.map(simple_pseudo,
              zip(stroke_brain_paths, stroke_seg_paths, oasis_brain_paths, oasis_seg_paths, output_sp_dirs))
        p.close()
        p.join()


def stroke_volume_clip(roi_txt_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/stroke.txt', threshold=0.01):
    """
    Give the file path of roi size of each data (sorted by size)
    copy the data larger than threshold to new dir
    the content in the file : [data_id  roi_size]
    :param threshold:
    :return:
    """
    data_list = open(roi_txt_path).read().splitlines()
    data_list = list(map(lambda x: [x.split(' ')[0], float(x.split(' ')[1])], data_list))
    for i in range(len(data_list) - 1, -1, -1):
        if data_list[i][1] <= 0.01:
            del data_list[i]
    data_list = list(map(lambda x: x[0], data_list))

    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokemniLin',
        output_path=f'/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokemniLinThresh{threshold}',
        template_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeORI/mni152NLin2009aSymSkullStripRAS.nii.gz',
        template_mask_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/StrokeORI/mni152NLin2009aSymSkullStrip_maskRAS.nii.gz',
        brain_condition=lambda x: x == 'brain.nii.gz',
        seg_condition=lambda x: x == 'seg.nii.gz',
        skip_condition=lambda x: x.split('/')[-1] not in data_list and 'sub-r' in x.split('/')[-1],
        processes=1
    )
    mp(_volume_copy2targetPath)
    print('finish')


def orient_center_crop():
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020mni152NLin',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020mni152NLinCropped',
        brain_condition=lambda x: 't1.nii.gz' in x,
        seg_condition=lambda x: 'seg.nii.gz' in x,
        processes=128
    )
    mp(_center_crop)
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISPseudoNLin',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/OASISPseudoNLinCropped',
        template_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152NLin2009aSym.nii.gz',
        template_mask_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152NLin2009aSym_mask.nii.gz',
        brain_condition=lambda x: x == 'brain.nii.gz',
        seg_condition=lambda x: x == 'seg.nii.gz',
        processes=128
    )
    mp(_center_crop)
    print('done')


def registration_fsl():
    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSPseudo',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSPseudo2MNINLin',
        template_path=template_path,
        template_mask_path=template_mask_path,
        brain_condition=lambda x: x == 'brain.nii.gz',
        seg_condition=lambda x: x == 'seg.nii.gz',
        processes=128
    )
    # mp(_affine_brain_linear_fsl)
    # mp(_affine_brain_nonlinear_fsl)
    mp(_affine_brain_nonlinear_fsl_seg)
    print('done')


def landmark():
    # mp = MultiThreadPreprocess(
    #     input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSRegWithLandmark',
    #     output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSRegWithLandmarkRAS',
    #     template_path=template_path,
    #     template_mask_path=template_mask_path,
    #     brain_condition=lambda x: '0000_t1.nii.gz' in x,
    #     seg_condition=lambda x: '0000_landmarks.csv' in x,
    #     processes=128
    # )
    # # mp(_landmark2nii_with_failed)
    # mp(_landmark_ras_nii)
    # print('done')
    #
    # for i in ('pre', 'post'):
    #     mp = MultiThreadPreprocess(
    #         input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSRegWithLandmarkRAS',
    #         output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSRegWithLandmarkNLin',
    #         template_path=template_path,
    #         template_mask_path=template_mask_path,
    #         brain_condition=lambda x: f'{i}_t1' in x,
    #         seg_condition=lambda x: f'{i}_landmarks_0.nii.gz' in x,
    #         processes=128
    #     )
    #     mp(_affine_brain)
    #     mp(_affine_landmark_seg)
    #     print('finish')

    mp = MultiThreadPreprocess(
        input_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSRegWithLandmarkNLin',
        output_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTSRegWithLandmarkNLin',
        template_path=template_path,
        template_mask_path=template_mask_path,
        brain_condition=lambda x: 'pre_t1.nii.gz' in x,
        seg_condition=lambda x: 'pre_landmarks_0.nii.gz' in x,
        processes=128
    )
    mp(_nii2landmark)
    print('finish')


def change_datatype(input_path):
    print('Strat change datatype')
    mp = MultiThreadPreprocess(
        input_path=input_path,
        output_path=input_path,
        template_path=template_path,
        template_mask_path=template_mask_path,
        brain_condition=lambda x: 'brain.nii.gz' in x,
        seg_condition=lambda x: 'seg.nii.gz' in x,
        processes=128
    )
    mp(_change_datatype)
    print('finish')


# skull_remove(
#     img='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152/SymNLin/MNI152SymNonLinearSkull.nii',
#     seg='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152/SymNLin/MNI152SymNonLinearSkull_mask.nii',
#     o_img='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152/SymNLin/MNI152SymNonLinear.nii.gz',
#     o_seg='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152/SymNLin/MNI152SymNonLinear_mask.nii.gz',
# )

if __name__ == "__main__":
    import argparse
    from monai.utils import set_determinism

    set_determinism(42)
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument("--task", default="None", type=str)
    args = parser.parse_args()

    template_path = '/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152/SymNLinFreeSurfer/MNI152SymNonLinear.nii.gz'
    template_mask_path = '/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/MNI152/SymNLinFreeSurfer/MNI152SymNonLinear_mask.nii.gz'
    if 'atlas' in args.task:
        if 'brats' in args.task:
            brats2mni152()
        if 'oasis' in args.task:
            oasis2mni152()
        if 'stroke' in args.task:
            stroke2mni152()

    elif 'pseudo' in args.task:
        create_pseudo(
            bra_dir='GIRNet/data/BraTS2020MNI152NLin',
            oasis_dir='GIRNet/data/OASISMNI152NLin',
            stroke_dir='GIRNet/data/StrokeMNI152NLin',
            target_bp_path='GIRNet/data/BraTSPseudoHisto',
            target_sp_path=None,
            processes=128,
            num=300
        )
    elif 'volume_clip' in args.task:
        stroke_volume_clip(0.01)
    elif 'fsl' in args.task:
        registration_fsl()
    elif 'center_crop' in args.task:
        orient_center_crop()
    elif 'landmark' in args.task:
        landmark()
    elif 'skull' in args.task:
        skull_strip(False)
    elif 'change_datatype' in args.task:
        change_datatype('/data_58/liuy/GIRNet/data/BraTSPseudoHisto')
    else:
        raise KeyError
