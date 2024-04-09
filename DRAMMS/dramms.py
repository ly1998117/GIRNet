import os
import nibabel as nib
import numpy as np
import torch
import sys

sys.path.append('/data_58/liuy/GIRNet')
from monai.transforms import SpatialCrop
from params import crop_center

dramms_path = '/data_58/liuy/dramms-1.5.1/bin'
result_dir = '/data_58/liuy/GIRNet/result/DRAMMS'

data_type = {
    'DT_SIGNED_SHORT': np.array(4).astype(np.int16),
    'DT_UINT16': np.array(512).astype(np.int16),
    'DT_FLOAT': np.array(16).astype(np.int16)
}


def changetype_crop(path, o_path):
    x = nib.load(path)
    x.header['datatype'] = data_type['DT_FLOAT']
    data = x.get_fdata()
    data = cropper(data.reshape(1, *data.shape)).squeeze()
    nib.save(nib.Nifti1Image(data, np.eye(4), x.header), o_path)


def orient_RAS(path):
    """
    original orientation of `data_array` is defined by `affine`.
    :param axcodes:
    :param data_array:
    :param affine: (matrix) (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
    :return: data_array [reoriented in `axcodes`] if `self.image_only`, else
            (data_array [reoriented in `axcodes`], original axcodes, current axcodes).
    """
    x = nib.load(path)
    data_array = x.get_fdata()
    # affine = x.affine
    # labels = (("L", "R"), ("P", "A"), ("I", "S"))
    # src = nib.io_orientation(affine)
    # dst = nib.orientations.axcodes2ornt('RAS', labels=labels)
    # spatial_ornt = nib.orientations.ornt_transform(src, dst)
    # axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    # if axes:
    #     data_array = (
    #         np.flip(data_array, axis=axes)  # type: ignore
    #     )
    # full_transpose = np.arange(len(data_array.shape))
    # full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    # if not np.all(full_transpose == np.arange(len(data_array.shape))):
    #     data_array = data_array.transpose(full_transpose)  # type: ignore
    nib.save(nib.Nifti1Image(data_array, np.eye(4), x.header),
             os.path.join(os.path.dirname(path), 'RAS_' + os.path.basename(path)))


def _registrate(params):
    img, k_fold = params
    in_dir = os.path.dirname(img)
    out_dir = os.path.join(result_dir, img.split('/')[-3], f'data_fold_{k_fold}', img.split('/')[-2])
    os.makedirs(out_dir, exist_ok=True)
    try:
        os.system(f'chmod 777 -R {out_dir}')
    except:
        pass
    if os.path.exists(os.path.join(in_dir, 'post_t1.nii.gz')):
        template = os.path.join(in_dir, 'post_t1.nii.gz')
        changetype_crop(template, os.path.join(out_dir, 'post_t1.nii.gz'))
        template = os.path.join(out_dir, 'post_t1.nii.gz')
    else:
        template = '/data_58/liuy/GIRNet/DRAMMS/MNI152.nii.gz'

    in_img, in_aseg, o_img, o_def, o_aseg = os.path.join(out_dir, os.path.basename(img)), \
        os.path.join(out_dir, 'aseg.nii.gz'), \
        os.path.join(out_dir, 'x2y.nii.gz'), \
        os.path.join(out_dir, 'x2y_def.nii.gz'), \
        os.path.join(out_dir, 'o_aseg.nii.gz')

    if os.path.exists(img) and not os.path.exists(in_img):
        changetype_crop(img, in_img)

    if os.path.exists(os.path.join(os.path.dirname(img), 'aseg.nii.gz')) and not os.path.exists(in_aseg):
        changetype_crop(os.path.join(os.path.dirname(img), 'aseg.nii.gz'), in_aseg)

    if os.path.exists(in_img) and not os.path.exists(o_img):
        print(img, k_fold)
        commands = [
            os.path.join(dramms_path, 'dramms'),
            "--source", in_img,
            "--target", template,
            "--outimg", o_img,
            "--outdef", o_def,
            "-g", '0.4',
            "-c", '2',
        ]
        command = " ".join(commands)
        os.system(command)

    if os.path.exists(in_aseg):
        commands = [os.path.join(dramms_path, 'dramms-warp'), in_aseg, o_def, o_aseg, '-n']
        command = " ".join(commands)
        os.system(command)
    else:
        commands = [os.path.join(dramms_path, 'dramms-warp'), in_img, o_def, o_img, '-n']
        command = " ".join(commands)
        os.system(command)

    df = []
    for direction in ['z', 'x', 'y']:
        df.append(nib.load(os.path.join(out_dir, f'DF{direction}.nii.gz')).get_fdata())
    df = np.stack(df, axis=-1)
    nib.save(nib.Nifti1Image(df, np.eye(4)), os.path.join(out_dir, 'DF.nii.gz'))

    if os.path.exists(os.path.join(out_dir, 'MutualSaliencyMap_x2y.nii.gz')):
        orient_RAS(os.path.join(out_dir, 'MutualSaliencyMap_x2y.nii.gz'))
    if os.path.exists(o_img):
        orient_RAS(o_img)
    if os.path.exists(o_aseg):
        orient_RAS(o_aseg)
    if os.path.exists(o_def):
        orient_RAS(o_def)


def transform_torch(x_path, field):
    img = nib.load(x_path)
    field = nib.load(field)
    x_data = torch.from_numpy(img.get_fdata()).float().reshape(1, 1, *img.shape)
    field_data = torch.from_numpy(field.get_fdata().squeeze()).float()
    grid = np.mgrid[0:x_data.shape[0], 0:x_data.shape[1], 0:x_data.shape[2]].transpose(1, 2, 3, 0)[
        ..., [2, 1, 0]].unsqueeze(0)
    grid = grid + field_data
    y_data = torch.nn.functional.grid_sample(x_data, grid, mode='nearest', align_corners=True)
    return y_data


def main(data_dir, k_fold):
    from multiprocessing import Pool
    from monai.data import load_decathlon_datalist

    path = os.path.join('/data_58/liuy/GIRNet', data_dir, f'data_fold_{k_fold}.json')
    val_list = [[data_dict['moving_t1'], k_fold] for data_dict in
                load_decathlon_datalist(path, True, data_list_key="validation")]
    pool = Pool(32)
    pool.map(_registrate, val_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # cropper = SpatialCrop(roi_size=(160, 192, 144), roi_center=crop_center(''))
    cropper = SpatialCrop(roi_size=(160, 192, 144), roi_center=crop_center('landmark'))

    for i in range(5):
        # main('data_json/train_BraTSPseudoHistoNLin', i)
        main('data_json/train_BraTSRegLandmarkNLin', i)
        # main('data_json/train_BraTSNLin', i)
    # x = '/data_58/liuy/GIRNet/DRAMMS/result/BraTSPseudoHisto/data_fold_0/OAS1_0019_MR1-BraTS20_308/brain.nii.gz'
    # d = '/data_58/liuy/GIRNet/DRAMMS/result/BraTSPseudoHisto/data_fold_0/OAS1_0019_MR1-BraTS20_308/x2y_def.nii.gz'
    # transform(x, d)
