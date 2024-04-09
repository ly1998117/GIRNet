# third party
import json
import os

import SimpleITK as sitk
import torch
import csv
import nibabel as nib
import time
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gc
import numpy as np
from scipy import linalg
from ast import literal_eval
from nilearn.image import math_img, resample_to_img, new_img_like

import warnings

warnings.filterwarnings("ignore")


########################################################################################
class TXTLogs:
    def __init__(self, dir_path, file_name='logs'):
        dir_path = os.path.join(dir_path, 'TXTLogger')
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, f'{file_name}_{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.txt')

    def log(self, text):
        with open(self.path, 'a+') as f:
            print(text, file=f)


class CSVLogs:
    def __init__(self, dir_path, file_name='logs'):
        dir_path = os.path.join(dir_path, 'CSVLogger')
        os.makedirs(dir_path, exist_ok=True)
        self.df = pd.DataFrame()
        self.path = os.path.join(dir_path, f'{file_name}.csv')
        self.exist = False

    def resume(self, key, value):
        if os.path.exists(self.path):
            try:
                self.df = pd.read_csv(self.path)
                self.df = self.df.drop(self.df[(self.df[key] >= value)].index)
                self.path = os.path.join(os.path.dirname(self.path), 'resume_' + os.path.basename(self.path))
                self.df.to_csv(self.path, index=False)
            except:
                print(f'{self.path}: resume failed')

    def cache(self, text: dict):
        if isinstance(text, dict):
            text = [text]
        try:
            self.df = pd.concat([self.df, pd.DataFrame(text)], axis=0).reset_index(drop=True)
        except:
            import pdb
            pdb.set_trace()

    def write(self):
        self.exist = True
        self.df.to_csv(self.path, index=False)

    def __call__(self, text: dict):
        self.cache(text)
        self.write()

    def dataframe(self):
        if len(self.df) == 0:
            df = pd.read_csv(self.path)
            for c in df.columns:
                try:
                    df[c] = df[c].map(lambda x: literal_eval(x))
                except Exception:
                    pass
            self.df = df
        return self.df


def read_landmark(path):
    with open(path, 'r', newline='') as f:
        land_l = list(csv.DictReader(f, skipinitialspace=True))
        land_l = list(map(lambda x:
                          (int(float(x['X'])), int(float(x['Y'])), int(float(x['Z']))),
                          land_l))
    return land_l


def write_landmark(land_l, path):
    land_d = []
    for idx, i in enumerate(land_l):
        land_d.append({'Landmark': idx, 'X': i[0], 'Y': i[1], 'Z': i[2]})
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Landmark', 'X', 'Y', 'Z'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(land_d)


def set_zero_near_point_nii(landmarks):
    landmarks[landmarks < 0.1] = 0
    landmark_index = list(zip(*np.where(landmarks)))
    # repeated item set to 0
    while len(landmark_index) > 0:
        sr = []
        i = len(landmark_index) - 1
        top = landmark_index[i]
        for j in range(i, -1, -1):
            if abs(np.array(top) - np.array(landmark_index[j])).sum() < 8:
                repeat = landmark_index.pop(j)
                sr.append([repeat, landmarks[repeat]])
                i -= 1
        if len(sr) == 1:
            continue
        sr.sort(key=lambda x: x[-1])
        for index in sr[:-1]:
            landmarks[index[0]] = 0
    return landmarks


def get_landmarks_from_nii(landmarks):
    landmarks = set_zero_near_point_nii(landmarks)
    landmark_index = list(zip(*np.where(landmarks)))
    return landmark_index


def char_color(s, front=50, color='g'):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    if color == 'g':
        word = 32
    elif color == 'r':
        word = 31
    elif color == 'b':
        word = 34
    else:
        raise KeyError
    new_char = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"
    return new_char


def make_dirs(path):
    try:
        os.makedirs(path, exist_ok=True)
        os.chmod(path, mode=0o777)
    except Exception:
        pass


########################################################################################
# Image Operation Utils

def save_img(I_img, path, header=None, affine=None):
    if isinstance(I_img, torch.Tensor):
        I_img = I_img.detach().cpu().numpy()
    I_img = I_img.squeeze()
    if I_img.shape[0] == 3:
        I_img = I_img.transpose(1, 2, 3, 0)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)
    nib.save(new_img, path)


def read_img(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine


def save_flow(flow, path):
    if isinstance(flow, torch.Tensor):
        flow = flow.squeeze().cpu().numpy()
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(flow, affine, header=None)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    nib.save(new_img, path)


class NIFTFileReader:
    """
    Read NIFT image, return the data array and affine
    """

    def __init__(self, data_path, txt_path=None, txt_key=None, name_list=None, crop=False, scale=False, affine=False,
                 datatype=torch.Tensor):
        """
        if datatype is tensor, return B C W H D. array (C W H D)
        :param data_path:
        :param name_list:
        :param crop:
        :param scale:
        :param affine:
        :param datatype:
        """
        self.dir_path = data_path
        self.txt_path = txt_path
        self.txt_key = txt_key
        self.patient_list = os.listdir(data_path)
        self.txt_list = os.listdir(txt_path) if txt_path is not None else None
        self._preprocesser(crop, scale)
        self.name_list = name_list
        self.affine = affine
        self.datatype = datatype

    def _preprocesser(self, crop, scale):
        from monai.transforms import SpatialCrop, ScaleIntensity
        from .data_preprocess import orient
        from params import crop_center, img_size
        # self.orient = lambda d, a: orient(axcodes='RAS', data_array=d, affine=a)
        if crop:
            self.cropper = SpatialCrop(roi_size=img_size[''], roi_center=crop_center())
        else:
            self.cropper = None
        if scale:
            self.scaler = ScaleIntensity(minv=0, maxv=1)
        else:
            self.scaler = None

    def channel_first(self, data):
        if len(data.shape) == 3:
            data = data.reshape(1, *data.shape)
        if data.shape[-1] == 3:
            data = data.transpose(3, 0, 1, 2)
        return data

    def get_exist_name(self, sub_list, idx):
        if idx in sub_list:
            return idx
        else:
            for i in sub_list:
                if idx in i:
                    return i
        return ''

    def read(self, idx, data_name):
        if self.txt_path:
            p = self.get_exist_name(self.txt_list, idx)
            with open(os.path.join(self.txt_path, p,
                                   self.get_exist_name(os.listdir(os.path.join(self.txt_path, p)),
                                                       'f_n.txt'))) as f:
                txt = json.load(f)
                idx = txt[self.txt_key]
        p = self.get_exist_name(self.patient_list, idx)
        data_name = self.get_exist_name(os.listdir(os.path.join(self.dir_path, p)), data_name)
        data, affine = read_img(os.path.join(self.dir_path, p, data_name))
        data = self.channel_first(data)
        if self.cropper:
            data = self.cropper(data)
            affine = np.diag([1, 1, 1, 1])

        if data.shape[0] == 1 and self.scaler:
            data = self.scaler(data)
            affine = np.diag([1, 1, 1, 1])
        if self.datatype == torch.Tensor:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            while len(data.shape) != 5:
                data = data.unsqueeze(0)
        else:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

        if self.affine:
            return data, affine
        else:
            return data

    def __call__(self, idx=None, names=None):
        if names is None:
            names = self.name_list
        datas = []
        for n in names:
            datas.append(self.read(idx, n))
        return datas


def image_resample(image, resize_factor=(1., 1., 1.), target_size=None, dimension=3, is_label=False):
    istorch = False
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        istorch = True
    shape = list(image.shape)
    image = image.squeeze()
    image = sitk.GetImageFromArray(image)
    img_sz = image.GetSize()
    if target_size is None:
        factor = np.flipud(resize_factor)
        after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
        spacing_factor = [(after_size[i] - 1) / (img_sz[i] - 1) for i in range(len(img_sz))]
    else:
        target_size = np.flipud(target_size)
        factor = [target_size[i] / img_sz[i] for i in range(len(img_sz))]
        spacing_factor = [(target_size[i] - 1) / (img_sz[i] - 1) for i in range(len(img_sz))]
    resize = not all([f == 1 for f in factor])
    if resize:
        resampler = sitk.ResampleImageFilter()
        affine = sitk.AffineTransform(dimension)
        matrix = np.array(affine.GetMatrix()).reshape((dimension, dimension))
        after_size = [round(img_sz[i] * factor[i]) for i in range(dimension)]
        after_size = [int(sz) for sz in after_size]
        if target_size is not None:
            for i in range(len(target_size)):
                assert target_size[i] == after_size[i]
        matrix[0, 0] = 1. / spacing_factor[0]
        matrix[1, 1] = 1. / spacing_factor[1]
        matrix[2, 2] = 1. / spacing_factor[2]
        affine.SetMatrix(matrix.ravel())
        resampler.SetSize(after_size)
        resampler.SetTransform(affine)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
        img_resampled = resampler.Execute(image)
    else:
        img_resampled = image
    image = sitk.GetArrayFromImage(img_resampled)
    shape[-3:] = image.shape[-3:]
    image = image.reshape(shape)
    if istorch:
        image = torch.from_numpy(image).float()
    return image


def erode(image, kernel_size=3):
    """
    erode image after padding
    :param image: tensor
    :param kernel_size:
    :return:
    """
    shape = image.shape
    pad = (kernel_size - 1) // 2
    image = F.pad(image, [pad, pad] * 3, mode='replicate')
    # B C H W Z k k k
    for i in range(2, 5):
        image = image.unfold(dimension=i, size=kernel_size, step=1)
    eroded, _ = image.reshape(*shape, -1).min(dim=-1)
    return eroded


def dilate(image, kernel_size=3):
    """
    erode image after padding
    :param image: tensor
    :param kernel_size:
    :return:
    """

    shape = image.shape
    dim = len(shape) - 2
    pad = (kernel_size - 1) // 2
    image = F.pad(image, [pad, pad] * dim, mode='replicate')
    # B C H W Z k k k
    for i in range(2, 2 + dim):
        image = image.unfold(dimension=i, size=kernel_size, step=1)
    eroded, _ = image.reshape(*shape, -1).max(dim=-1)
    return eroded


def boundary(m):
    def conv(x):
        x = F.pad(x, (1,) * 4, mode='replicate')
        x = F.conv2d(x, kernel, padding=0, stride=1)
        return x

    kernel = torch.tensor([
        [[0., -1., 0.],
         [-1, 4., -1.],
         [0., -1., 0.]],
    ]).view((1, 1, 3, 3))
    fore_boundary = conv(m) >= 0.5
    back_boundary = conv(1 - m) >= 0.5
    return fore_boundary + back_boundary


def roi_center(img):
    """
    :param img:
    :return: the brain center
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    x_mid = (rmin + rmax) // 2
    y_mid = (cmin + cmax) // 2
    z_mid = (zmin + zmax) // 2
    return x_mid, y_mid, z_mid


def array2image(img, img_array, direction=None, origin=None, spacing=None):
    new_image = sitk.GetImageFromArray(img_array)
    if direction is None:
        new_image.SetDirection(img.GetDirection())
    else:
        new_image.SetDirection(direction)
    if origin is None:
        new_image.SetOrigin(img.GetOrigin())
    else:
        new_image.SetOrigin(origin)
    if spacing is None:
        new_image.SetSpacing(img.GetSpacing())
    else:
        new_image.SetSpacing(spacing)

    for k in img.GetMetaDataKeys():
        new_image.SetMetaData(k, img.GetMetaData(k))
    return new_image


from skimage.exposure import match_histograms


def histogram_matching(x, y, channel_axis=None):
    return match_histograms(x, y, channel_axis=channel_axis)


################################################################################################################
# FSL Registration Utils
################################################################################################################


def gen_mask(t1w_head, t1w_brain, mask):
    import os.path as op

    t1w_brain_mask = f"{op.dirname(t1w_head)}/t1w_brain_mask.nii.gz"

    if mask is not None and op.isfile(mask):
        img = nib.load(t1w_head)
        print(f"Using {mask}...")
        mask_img = nib.load(mask)
        nib.save(
            resample_to_img(
                mask_img,
                img),
            t1w_brain_mask)
    else:
        # Perform skull-stripping if mask not provided.
        try:
            t1w_brain_mask = deep_skull_strip(t1w_head, t1w_brain_mask)
        except RuntimeError as e:
            print(e, 'Deepbrain extraction failed...')

    # Threshold T1w brain to binary in anat space
    math_img("img > 0.0001", img=nib.load(t1w_brain_mask)
             ).to_filename(t1w_brain_mask)

    t1w_brain = apply_mask_to_image(t1w_head, t1w_brain_mask, t1w_brain)

    assert op.isfile(t1w_brain)
    assert op.isfile(t1w_brain_mask)

    return t1w_brain, t1w_brain_mask


def deep_skull_strip(t1w_data, t1w_brain_mask):
    print('Stripping skull and extracting brain...')
    commands = [
        os.path.join('/data_smr/liuy/Software/freesurfer/bin', 'mri_synthstrip'),
        "-i", t1w_data,
        # "-o", os.path.join(out_dir, 'brain.nii.gz'),
        "-m", t1w_brain_mask
    ]
    command = " ".join(commands)
    os.system(command)
    return t1w_brain_mask


def roi2t1w_align(
        roi,
        t1w_brain,
        mni2t1_xfm,
        mni2t1w_warp,
        roi_in_t1w,
        template,
        simple):
    """
    A function to perform alignment of a roi from MNI space --> T1w.
    """

    roi_img = nib.load(roi)
    template_img = nib.load(template)

    roi_img_res = resample_to_img(
        roi_img, template_img, interpolation="nearest"
    )
    roi_res = f"{roi.split('.nii')[0]}_res.nii.gz"
    nib.save(roi_img_res, roi_res)

    # Apply warp or transformer resulting from the inverse
    # MNI->T1w created earlier
    if simple is False:
        apply_warp(t1w_brain, roi_res, roi_in_t1w,
                   warp=mni2t1w_warp)
    else:
        applyxfm(t1w_brain, roi_res, mni2t1_xfm,
                 roi_in_t1w)

    time.sleep(0.5)

    return roi_in_t1w


def segment_t1w(t1w, basename, nclass=3, beta=0.1, max_iter=100):
    """
    A function to use FSL's FAST to segment an anatomical
    image into GM, WM, and CSF prob maps.

    Parameters
    ----------
    t1w : str
        File path to an anatomical T1-weighted image.
    basename : str
        A basename to use for output files.

    Returns
    -------
    out : str
        File path to the probability map Nifti1Image consisting of GM, WM,
        and CSF in the 4th dimension.

    """
    from dipy.segment.tissue import TissueClassifierHMRF

    print("Segmenting T1w...")

    t1w_img = nib.load(t1w)
    hmrf = TissueClassifierHMRF()
    PVE = hmrf.classify(t1w_img.get_fdata(), nclass, beta,
                        max_iter=max_iter)[2]

    new_img_like(t1w_img, PVE[..., 2]).to_filename(
        f"{os.path.dirname(os.path.abspath(t1w))}/{basename}_{'pve_0.nii.gz'}")

    new_img_like(t1w_img, PVE[..., 1]).to_filename(
        f"{os.path.dirname(os.path.abspath(t1w))}/{basename}_{'pve_1.nii.gz'}")

    new_img_like(t1w_img, PVE[..., 0]).to_filename(
        f"{os.path.dirname(os.path.abspath(t1w))}/{basename}_{'pve_2.nii.gz'}")

    out = {}  # the outputs
    out["wm_prob"] = f"{os.path.dirname(os.path.abspath(t1w))}/{basename}_" \
                     f"pve_0.nii.gz"
    out["gm_prob"] = f"{os.path.dirname(os.path.abspath(t1w))}/{basename}_" \
                     f"pve_1.nii.gz"
    out["csf_prob"] = f"{os.path.dirname(os.path.abspath(t1w))}/{basename}_" \
                      f"pve_2.nii.gz"

    del PVE
    t1w_img.uncache()
    gc.collect()

    return out


def align(
        inp,
        ref,
        oaff=None,
        out=None,
        inweight=None,
        dof=12,
        searchrad=True,
        bins=256,
        interp=None,
        cost="mutualinfo",
        sch=None,
        wmseg=None,
        init=None,
):
    """
    Aligns two images using linear registration (FSL's FLIRT).

    Parameters
    ----------
    inp : str
        File path to input Nifti1Image to be aligned for registration.
    ref : str
        File path to reference Nifti1Image to use as the target for
        alignment.
    xfm : str
        File path for the transformation matrix output in .xfm.
    out : str
        File path to input Nifti1Image output following registration
        alignment.
    inweight: str
        use weights for input volume, a zero for areas of a pathological nature
        and one elsewhere (do not put zero outside the brain)
    dof : int
        Number of degrees of freedom to use in the alignment.
    searchrad : bool
        Indicating whether to use the predefined searchradius parameter
        (180 degree sweep in x, y, and z). Default is True.
    bins : int
        Number of histogram bins. Default is 256.
    interp : str
        Interpolation method to use. Default is mutualinfo.
    sch : str
        Optional file path to a FLIRT schedule file.
    cost : str
        {mutualinfo,corratio,normcorr,normmi,leastsq,labeldiff,bbr}        (default is mutualinfo)
    wmseg : str
        Optional file path to white-matter segmentation Nifti1Image for
        boundary-based registration (BBR).
    init : str
        File path to a transformation matrix in .xfm format to use as an
        initial guess for the alignment.

    """
    cmd = f"flirt -in {inp} -ref {ref}"
    if oaff is not None:
        cmd += f" -omat {oaff}"
    if out is not None:
        cmd += f" -out {out}"
    if inweight is not None:
        cmd += f" -inweight {inweight}"
    if dof is not None:
        cmd += f" -dof {dof}"
    if bins is not None:
        cmd += f" -bins {bins}"
    if interp is not None:
        cmd += f" -interp {interp}"
    if cost is not None:
        cmd += f" -cost {cost}"
    if searchrad is True:
        cmd += " -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
    if sch is not None:
        cmd += f" -schedule {sch}"
    if wmseg is not None:
        cmd += f" -wmseg {wmseg}"
    if init is not None:
        cmd += f" -init {init}"
    print(cmd)
    os.system(cmd)
    return


def align_nonlinear(
        inp,
        ref,
        aff,
        out,
        warp,
        ref_mask=None,
        in_mask=None,
        config='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/utils/T1_2_MNI152_1mm.conf'):
    """
    Aligns two images using nonlinear registration and stores the transform
    between them.

    Parameters
    ----------
    inp : str
        File path to input Nifti1Image to be aligned for registration.
    ref : str
        File path to reference Nifti1Image to use as the target for
        alignment.
    xfm : str
        File path for the transformation matrix output in .xfm.
    out : str
        File path to input Nifti1Image output following registration
        alignment.
    warp : str
        File path to input Nifti1Image output for the nonlinear warp
        following alignment.
    ref_mask : str
        Optional file path to a mask in reference image space. NMI 152 mask
    in_mask : str
        Optional file path to a mask in input image space. cost function masking
    config : str
        Optional file path to config file specifying command line
        arguments.

    """
    cmd = f"fnirt --in={inp} --ref={ref} --aff={aff} --iout={out} " \
          f"--cout={warp} --warpres=8,8,8"
    if ref_mask is not None:
        cmd += f" --refmask={ref_mask}"
    if in_mask is not None:
        cmd += f" --inmask={in_mask}"
    if config is not None:
        cmd += f" --config={config}"
    print(cmd)
    os.system(cmd)
    return


def warp2displacementField(warp, ref, field, jac=None, withaff=True):
    """
    This utility is used to convert field->coefficients, coefficients->field,
     coefficients->other_coefficients etc.

    :param warp:a 4D-file with three (for the x-, y- and z-directions) volumes of coefficients
            for quadratic/cubic splines.

    :param ref:
    :param field: displacement field
    :param jac: calculates a map of Jacobian determinants (reflecting expansions/contractions)
            given a coefficient/field-file
    :param withaff: with the affine transform included as part of the field,
            can be useful when one wants to use software that cannot decode the FSL coefficient-file format.
    :return:
    """
    cmd = f"fnirtfileutils --in={warp} --ref={ref} --out={field}"
    if jac is not None:
        cmd += f' --jac={jac}'
    if withaff:
        cmd += " --withaff"
    print(cmd)
    os.system(cmd)
    return


def applyxfm(ref, inp, xfm, aligned, interp="trilinear", dof=6):
    """
    Aligns two images with a given transform.

    Parameters
    ----------
    inp : str
        File path to input Nifti1Image to be aligned for registration.
    ref : str
        File path to reference Nifti1Image to use as the target for
        alignment.
    xfm : str
        File path for the transformation matrix output in .xfm.
    aligned : str
        File path to input Nifti1Image output following registration
        alignment.
    interp : str
        Interpolation method to use. Default is trilinear.
    dof : int
        Number of degrees of freedom to use in the alignment.

    """
    cmd = f"flirt -in {inp} -ref {ref} -out {aligned} -init {xfm} -interp" \
          f" {interp} -dof {dof} -applyxfm"
    print(cmd)
    os.system(cmd)
    return


def apply_warp(
        ref,
        inp,
        out,
        warp=None,
        xfm=None,
        mask=None,
        interp=None,
        sup=False):
    """
    Applies a warp to a Nifti1Image which transforms the image to the
    reference space used in generating the warp.

    Parameters
    ----------
    ref : str
        File path to reference Nifti1Image to use as the target for
        alignment.
    inp : str
        File path to input Nifti1Image to be aligned for registration.
    out : str
        File path to input Nifti1Image output following registration
        alignment.
    warp : str
        File path to input Nifti1Image output for the nonlinear warp
        following alignment.
    xfm : str
        File path for the transformation matrix input in .xfm.
    mask : str
        Optional file path to a mask in reference image space.
    interp : str
        Interpolation method to use.
    sup : bool
        Intermediary supersampling of output. Default is False.

    """
    cmd = f"applywarp --ref={ref} --in={inp} --out={out}"
    if xfm is not None:
        cmd += f" --premat={xfm}"
    if warp is not None:
        cmd += f" --warp={warp}"
    if mask is not None:
        cmd += f" --mask={mask}"
    if interp is not None:
        cmd += f" --interp={interp}"
    if sup is True:
        cmd += " --super --superlevel=a"
    print(cmd)
    os.system(cmd)
    return


def inverse_warp(ref, out, warp):
    """
    Generates the inverse of a warp from a reference image space to
    the input image used in generating the warp.

    Parameters
    ----------
    ref : str
        File path to reference Nifti1Image to use as the target for
        alignment.
    out : str
        File path to input Nifti1Image output following registration
        alignment.
    warp : str
        File path to input Nifti1Image output for the nonlinear warp
        following alignment.

    """
    cmd = f"invwarp --warp={warp} --out={out} --ref={ref}"
    print(cmd)
    os.system(cmd)
    return


def combine_xfms(xfm1, xfm2, xfmout):
    """
    A function to combine two transformations, and output the resulting
    transformation.

    Parameters
    ----------
    xfm1 : str
        File path to the first transformation.
    xfm2 : str
        File path to the second transformation.
    xfmout : str
        File path to the output transformation.

    """
    cmd = f"convert_xfm -omat {xfmout} -concat {xfm1} {xfm2}"
    print(cmd)
    os.system(cmd)
    return


def invert_xfm(in_mat, out_mat):
    import os
    cmd = f"convert_xfm -omat {out_mat} -inverse {in_mat}"
    print(cmd)
    os.system(cmd)
    return out_mat


def apply_mask_to_image(input, mask, output):
    img = math_img("input*mask", input=nib.load(input), mask=nib.load(mask))
    img.dataobj[img.dataobj < 0.001] = 0
    img.to_filename(output)
    img.uncache()

    return output


def get_wm_contour(wm_map, mask, wm_edge):
    import os
    cmd = f"fslmaths {wm_map} -edge -bin -mas {mask} {wm_edge}"
    print(cmd)
    os.system(cmd)
    return wm_edge


def orient_reslice(
        infile,
        outdir,
        vox_size,
        bvecs=None,
        overwrite=True):
    """
    An API to reorient any image to RAS+ and resample
    any image to a given voxel resolution.

    Parameters
    ----------
    infile : str
        File path to a Nifti1Image.
    outdir : str
        Path to base derivatives directory.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    bvecs : str
        File path to corresponding bvecs file if infile is a dwi.
    overwrite : bool
        Boolean indicating whether to overwrite existing outputs.
        Default is True.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile
        is a dwi.

    """
    import time
    img = nib.load(infile)
    vols = img.shape[-1]

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        # Check orientation
        if ("reor-RAS" not in infile) or (overwrite is True):
            [infile, bvecs] = reorient_dwi(
                infile, bvecs, outdir, overwrite=overwrite)
            time.sleep(0.25)
        # Check dimensions
        if ("res-" not in infile) or (overwrite is True):
            outfile = match_target_vox_res(
                infile, vox_size, outdir, overwrite=overwrite
            )
            time.sleep(0.25)
            print(outfile)
        else:
            outfile = infile
    elif (vols > 1) and (bvecs is None):
        # func case
        # Check orientation
        if ("reor-RAS" not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir, overwrite=overwrite)
            time.sleep(0.25)
        # Check dimensions
        if ("res-" not in infile) or (overwrite is True):
            outfile = match_target_vox_res(
                infile, vox_size, outdir, overwrite=overwrite
            )
            time.sleep(0.25)
            print(outfile)
        else:
            outfile = infile
    else:
        # t1w case
        # Check orientation
        if ("reor-RAS" not in infile) or (overwrite is True):
            infile = reorient_img(infile, outdir, overwrite=overwrite)
            time.sleep(0.25)
        # Check dimensions
        if ("res-" not in infile) or (overwrite is True):
            outfile = match_target_vox_res(
                infile, vox_size, outdir, overwrite=overwrite
            )
            time.sleep(0.25)
            print(outfile)
        else:
            outfile = infile

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs


def normalize_xform(img):
    """ Set identical, valid qform and sform matrices in an image
    Selects the best available affine (sform > qform > shape-based), and
    coerces it to be qform-compatible (no shears).
    The resulting image represents this same affine as both qform and sform,
    and is marked as NIFTI_XFORM_ALIGNED_ANAT, indicating that it is valid,
    not aligned to template, and not necessarily preserving the original
    coordinates.
    If header would be unchanged, returns input image.
    """
    # Let nibabel convert from affine to quaternions, and recover xform
    tmp_header = img.header.copy()
    tmp_header.set_qform(img.affine)
    xform = tmp_header.get_qform()
    xform_code = 2

    # Check desired codes
    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)
    if all(
            (
                    qform is not None and np.allclose(qform, xform),
                    sform is not None and np.allclose(sform, xform),
                    int(qform_code) == xform_code,
                    int(sform_code) == xform_code,
            )
    ):
        return img

    new_img = img.__class__(
        np.asarray(
            img.dataobj),
        xform,
        img.header)

    # Unconditionally set sform/qform
    new_img.set_sform(xform, xform_code)
    new_img.set_qform(xform, xform_code)

    return new_img


def reorient_dwi(dwi_prep, bvecs, out_dir, overwrite=True):
    """
    A function to reorient any dwi image and associated bvecs to RAS+.

    Parameters
    ----------
    dwi_prep : str
        File path to a dwi Nifti1Image.
    bvecs : str
        File path to corresponding bvecs file.
    out_dir : str
        Path to output directory.

    Returns
    -------
    out_fname : str
        File path to the reoriented dwi Nifti1Image.
    out_bvec_fname : str
        File path to corresponding reoriented bvecs file.

    """
    import os

    fname = dwi_prep
    bvec_fname = bvecs

    out_bvec_fname = (
        f"{out_dir}/"
        f"{dwi_prep.split('/')[-1].split('.nii')[0]}"
        f"_bvecs_reor.bvec"
    )

    input_img = nib.load(fname)
    input_axcodes = nib.aff2axcodes(input_img.affine)
    reoriented = nib.as_closest_canonical(input_img)
    normalized = normalize_xform(reoriented)
    # Is the input image oriented how we want?
    new_axcodes = ("R", "A", "S")
    if normalized is not input_img:
        out_fname = (
            f"{out_dir}/"
            f"{dwi_prep.split('/')[-1].split('.nii')[0]}_"
            f"reor-RAS.nii.gz"
        )
        if (
                overwrite is False
                and os.path.isfile(out_fname)
                and os.path.isfile(out_bvec_fname)
        ):
            pass
        else:
            print(f"Reorienting {dwi_prep} to RAS+...")

            # Flip the bvecs
            transform_orientation = nib.orientations.ornt_transform(
                nib.orientations.axcodes2ornt(input_axcodes),
                nib.orientations.axcodes2ornt(new_axcodes),
            )
            bvec_array = np.genfromtxt(bvec_fname)
            if bvec_array.shape[0] != 3:
                bvec_array = bvec_array.T
            if not bvec_array.shape[0] == transform_orientation.shape[0]:
                raise ValueError("Unrecognized bvec format")

            output_array = np.zeros_like(bvec_array)
            for this_axnum, (axnum, flip) in enumerate(
                    transform_orientation):
                output_array[this_axnum] = bvec_array[int(axnum)
                                           ] * float(flip)
            np.savetxt(out_bvec_fname, output_array, fmt="%.8f ")
    else:
        out_fname = (
            f"{out_dir}/{dwi_prep.split('/')[-1].split('.nii')[0]}_"
            f"noreor-RAS.nii.gz"
        )
        out_bvec_fname = bvec_fname

    if (
            overwrite is False
            and os.path.isfile(out_fname)
            and os.path.isfile(out_bvec_fname)
    ):
        pass
    else:
        normalized.to_filename(out_fname)
        normalized.uncache()
        input_img.uncache()
        del normalized, input_img
    return out_fname, out_bvec_fname


def reorient_img(img, out_dir, overwrite=True):
    """
    A function to reorient any non-dwi image to RAS+.

    Parameters
    ----------
    img : str
        File path to a Nifti1Image.
    out_dir : str
        Path to output directory.

    Returns
    -------
    out_name : str
        File path to reoriented Nifti1Image.

    """
    # Load image, orient as RAS
    orig_img = nib.load(img)
    normalized = normalize_xform(nib.as_closest_canonical(orig_img))

    # Image may be reoriented
    if normalized is not orig_img:
        print(f"{'Reorienting '}{img}{' to RAS+...'}")
        out_name = (
            f"{out_dir}/{img.split('/')[-1].split('.nii')[0]}_"
            f"reor-RAS.nii.gz"
        )
    else:
        out_name = (
            f"{out_dir}/{img.split('/')[-1].split('.nii')[0]}_"
            f"noreor-RAS.nii.gz"
        )

    if overwrite is False and os.path.isfile(out_name):
        pass
    else:
        normalized.to_filename(out_name)
        orig_img.uncache()
        normalized.uncache()
        del orig_img

    return out_name


def match_target_vox_res(img_file, vox_size, out_dir, overwrite=True,
                         remove_orig=True):
    """
    A function to resample an image to a given isotropic voxel
    resolution.

    Parameters
    ----------
    img_file : str
        File path to a Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    out_dir : str
        Path to output directory.

    Returns
    -------
    img_file : str
        File path to resampled Nifti1Image.

    """
    from dipy.align.reslice import reslice

    # Check dimensions
    orig_img = img_file
    img = nib.load(img_file, mmap=False)

    zooms = img.header.get_zooms()[:3]
    if vox_size == "1mm":
        new_zooms = (1.0, 1.0, 1.0)
    elif vox_size == "2mm":
        new_zooms = (2.0, 2.0, 2.0)

    if (abs(zooms[0]), abs(zooms[1]), abs(zooms[2])) != new_zooms:
        img_file_res = (
            f"{out_dir}/"
            f"{os.path.basename(img_file).split('.nii')[0]}_"
            f"res-{vox_size}.nii.gz"
        )
        if overwrite is False and os.path.isfile(img_file_res):
            img_file = img_file_res
            pass
        else:
            import gc
            print(f"Reslicing image {img_file} "
                  f"to {vox_size}...")
            data2, affine2 = reslice(
                img.get_fdata(dtype=np.float32), img.affine, zooms, new_zooms
            )
            nib.save(
                nib.Nifti1Image(
                    data2,
                    affine=affine2),
                img_file_res)
            img_file = img_file_res
            del data2
            gc.collect()
    else:
        img_file_nores = (
            f"{out_dir}/"
            f"{os.path.basename(img_file).split('.nii')[0]}_"
            f"nores-{vox_size}"
            f".nii.gz")
        if overwrite is False and os.path.isfile(img_file_nores):
            img_file = img_file_nores
            pass
        else:
            nib.save(img, img_file_nores)
            img_file = img_file_nores

    if os.path.isfile(orig_img) and remove_orig is True:
        os.remove(orig_img)

    return img_file


#################################################################################################################


def display_images_with_alpha(image_z, alpha, image1, image2):
    """
    Display a plot with a slice from the 3D images that is alpha blended.
    It is assumed that the two images have the same physical charecteristics (origin,
    spacing, direction, size), if they do not, an exception is thrown.
    """
    img = (1.0 - alpha) * image1[:, :, image_z] + alpha * image2[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis("off")
    plt.show()


def np2sitk_image(df_numpy, origin=None, direction=None, spacing=None):
    df = sitk.GetImageFromArray(df_numpy, isVector=True)
    if origin is not None:
        df.SetOrigin(origin)
    if direction is not None:
        df.SetDirection(direction)
    if spacing is not None:
        df.SetSpacing(spacing)
    return df


def deformation_field_2_sitk(field):
    """
    @param df_numpy: W H D C
    @return:
    """
    if isinstance(field, sitk.Transform):
        return field
    if isinstance(field, torch.Tensor):
        field = field.cpu().squeeze().numpy()
    if field.shape[0] == 3:
        field = field.transpose(3, 2, 1, 0)

    df = np2sitk_image(field)
    df = sitk.Cast(df, sitk.sitkVectorFloat64)
    df = sitk.DisplacementFieldTransform(df)
    return df


def save_displacement_field(path, field, nii=True):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    df = deformation_field_2_sitk(field)
    if nii:
        sitk.WriteImage(df.GetDisplacementField(), path)
    else:
        try:
            sitk.WriteTransform(df, path)
        except RuntimeError:
            print("set a file suffix as tfm")


def read_displacement_field(path):
    field = sitk.ReadImage(path)
    field = sitk.DisplacementFieldTransform(field)
    return field


def uniform_random_points(bounds, num_points):
    """
    Generate random (uniform withing bounds) nD point cloud. Dimension is based on the number of pairs in the bounds input.

    Args:
        bounds (list(tuple-like)): list where each tuple defines the coordinate bounds.
        num_points (int): number of points to generate.

    Returns:
        list containing num_points numpy arrays whose coordinates are within the given bounds.
    """
    internal_bounds = [sorted(b) for b in bounds]
    # Generate rows for each of the coordinates according to the given bounds, stack into an array,
    # and split into a list of points.
    mat = np.vstack([np.random.uniform(b[0], b[1], num_points) for b in internal_bounds])
    return list(mat[:len(bounds)].T)


def target_registration_errors(tx, point_list, reference_point_list):
    """
    Distances between points transformed by the given transformation and their
    location in another coordinate system. When the points are only used to evaluate
    registration accuracy (not used in the registration) this is the target registration
    error (TRE).
    """
    return [np.linalg.norm(np.array(tx.TransformPoint(p)) - np.array(p_ref))
            for p, p_ref in zip(point_list, reference_point_list)]


def print_transformation_differences(tx1, tx2):
    """
    Check whether two transformations are "equivalent" in an arbitrary spatial region
    either 3D or 2D, [x=(-10,10), y=(-100,100), z=(-1000,1000)]. This is just a sanity check,
    as we are just looking at the effect of the transformations on a random set of points in
    the region.
    """
    if tx1.GetDimension() == 2 and tx2.GetDimension() == 2:
        bounds = [(-10, 10), (-100, 100)]
    elif tx1.GetDimension() == 3 and tx2.GetDimension() == 3:
        bounds = [(-10, 10), (-100, 100), (-1000, 1000)]
    else:
        raise ValueError('Transformation dimensions mismatch, or unsupported transformation dimensionality')
    num_points = 10
    point_list = uniform_random_points(bounds, num_points)
    tx1_point_list = [tx1.TransformPoint(p) for p in point_list]
    differences = target_registration_errors(tx2, point_list, tx1_point_list)
    print(tx1.GetName() + '-' +
          tx2.GetName() +
          ':\tminDifference: {:.2f} maxDifference: {:.2f}'.format(min(differences), max(differences)))


def compose_transform(tx1, tx2):
    tx1.AddTransform(tx2)
    return tx1


# def registration(fixed_image, moving_image):
#     registration_method = sitk.ImageRegistrationMethod()
#     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#     registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
#     registration_method.SetMetricSamplingPercentage(0.01)
#     registration_method.SetInterpolator(sitk.sitkLinear)
#     registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
#                                                       estimateLearningRate=registration_method.Once)
#     registration_method.SetOptimizerScalesFromPhysicalShift()
#     init = sitk.DisplacementFieldJacobianDeterminant()
#     registration_method.SetInitialTransform(initial_transform, inPlace=False)
#     registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
#     registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
#     registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
#
#     final_transform = registration_method.Execute(fixed_image, moving_image)
#     print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
#     print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#     return (final_transform, registration_method.GetMetricValue())

def registration_from_displacement_field_itk(moving, fixed, displacementField):
    istorch = False
    if isinstance(moving, torch.Tensor):
        shape = moving.shape
        moving = moving.cpu().squeeze().numpy()
        fixed = fixed.cpu().squeeze().numpy()
        istorch = True
    moving = sitk.GetImageFromArray(moving)
    fixed = sitk.GetImageFromArray(fixed)
    outTx = deformation_field_2_sitk(displacementField)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    # resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)
    out = sitk.GetArrayFromImage(out)
    if istorch:
        return torch.from_numpy(out).reshape(shape)
    return out


def _registration_error(
        transformed_fixed_point_list,
        reference_moving_point_list):
    if isinstance(transformed_fixed_point_list, list):
        transformed_fixed_point_list = np.array(transformed_fixed_point_list)
        reference_moving_point_list = np.array(reference_moving_point_list)
    errors = linalg.norm(transformed_fixed_point_list - reference_moving_point_list, axis=1)
    return np.mean(errors)


def transform_point(field, point):
    # point : (z, y, x)
    if isinstance(field, sitk.Transform):
        return field.TransformPoint(point)
    elif isinstance(field, torch.Tensor):
        return field[0, point[0], point[1], point[2]].cpu().numpy()


def transform_tensor(field, mask, grid=None, itk=False):
    """
    :param field:
    :param mask:
    :param grid:
    :param itk:
    :return: transform point array (2d)
    """
    if mask is None:
        mask = torch.ones_like(field)
    if field.shape[1] == 3:
        field = field.permute(0, 2, 3, 4, 1)
    if mask.shape[1] == 1:
        mask = mask.permute(0, 2, 3, 4, 1)
    if grid is None:
        from model.SymNet import generate_grid
        grid = torch.from_numpy(generate_grid(field.shape[1:-1])).reshape(field).to(field.device)

    if itk:
        mask = mask.cpu().squeeze()
        indexs = (mask > 0).nonzero().numpy()
        field = deformation_field_2_sitk(field)
        indexs = [transform_point(field, p.tolist()) for p in indexs]
    else:
        field = field + grid
        field = field * mask
        indexs = field.flatten(0, -2).cpu().numpy()
        indexs = indexs[indexs.sum(1) != 0.]
    return np.array(indexs)


def mean_registration_error(field, gt, mask=None, grid=None, itk=False):
    trans_indexes1 = transform_tensor(field, mask, grid, itk)
    trans_indexes2 = transform_tensor(gt, mask, grid, itk)
    return _registration_error(trans_indexes1, trans_indexes2)


def _warp_image_nn_3d(moving, phi):
    istensor = False
    if isinstance(moving, torch.Tensor):
        moving = moving.cpu().numpy()
        istensor = True
    shape = moving.shape
    moving = moving.squeeze()
    # get image dimensions
    dim1 = moving.shape[-3]
    dim2 = moving.shape[-2]
    dim3 = moving.shape[-1]

    # round the deformation map to integer
    phi_round = np.round(phi).astype('int')
    idx_x = np.reshape(phi_round[..., 2], (dim1 * dim2 * dim3, 1))
    idx_y = np.reshape(phi_round[..., 1], (dim1 * dim2 * dim3, 1))
    idx_z = np.reshape(phi_round[..., 0], (dim1 * dim2 * dim3, 1))

    # deal with extreme cases
    idx_x[idx_x < 0] = 0
    idx_x[idx_x > dim1 - 1] = dim1 - 1
    idx_y[idx_y < 0] = 0
    idx_y[idx_y > dim2 - 1] = dim2 - 1
    idx_z[idx_z < 0] = 0
    idx_z[idx_z > dim3 - 1] = dim3 - 1

    # get the wrapped results
    ind = np.ravel_multi_index([idx_x, idx_y, idx_z], [dim1, dim2, dim3])
    result = moving.flatten()[ind]
    result = np.reshape(result, shape)
    if istensor:
        result = torch.from_numpy(result).float()
    return result


if __name__ == "__main__":
    pass
