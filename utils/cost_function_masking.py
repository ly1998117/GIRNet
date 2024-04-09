# -*- encoding: utf-8 -*-
"""
@File    :   cost_function_masking.py   
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/7 10:21 PM   liu.yang      1.0         None
"""
import os
import nibabel as nib

try:
    from . import utils
except ImportError:
    import utils
from nilearn.image import math_img


class FmriReg(object):
    """
    A Class for Registering an atlas to a subject's MNI-aligned T1w image in
    native epi space.
    References
    ----------
    .. [1] Brett M, Leff AP, Rorden C, Ashburner J (2001) Spatial Normalization
      of Brain Images with Focal Lesions Using Cost Function Masking.
      NeuroImage 14(2) doi:10.006/nimg.2001.0845.
    .. [2] Zhang Y, Brady M, Smith S. Segmentation of brain MR images through a
      hidden Markov random field model and the expectation-maximization
      algorithm. IEEE Trans Med Imaging. 2001 Jan;20(1):45–57.
      doi:10.1109/42.906424.
    """

    def __init__(
            self,
            basedir_path,
            t1w,
            template_path,
            template_mask_path,
            vox_size='1mm',
            simple=True):
        import os.path as op

        self.t1w = t1w
        self.simple = simple
        self.vox_size = vox_size
        self.basedir_path = basedir_path
        # self.input_mni =
        self.input_mni_brain = template_path
        self.input_mni_mask = template_mask_path

        self.reg_path = f"{basedir_path}{'/reg'}"
        self.reg_path_mat = f"{self.reg_path}{'/mats'}"
        self.reg_path_warp = f"{self.reg_path}{'/warps'}"
        self.reg_path_img = f"{self.reg_path}{'/imgs'}"
        self.t12mni_xfm_init = f"{self.reg_path_mat}{'/xfm_t1w2mni.mat'}"
        self.t12mni_xfm = f"{self.reg_path_mat}{'/xfm_t1w2mni.mat'}"
        self.mni2t1_xfm = f"{self.reg_path_mat}{'/xfm_mni2t1.mat'}"
        self.mni2t1w_warp = f"{self.reg_path_warp}{'/mni2t1w_warp.nii.gz'}"
        self.warp_t1w2mni = f"{self.reg_path_warp}{'/t1w2mni_warp.nii.gz'}"
        self.t1_aligned_mni = (
            f"{self.reg_path_img}{'/'}{'aligned_mni.nii.gz'}"
        )
        self.t1w_brain = f"{self.reg_path_img}{'/'}" \
                         f"{'brain.nii.gz'}"
        self.t1w_head = f"{self.reg_path_img}{'/'}" \
                        f"{'head.nii.gz'}"
        self.t1w_brain_mask = (
            f"{self.reg_path_img}{'/'}"
            f"{'brain_mask.nii.gz'}"
        )
        self.map_name = f"{'seg'}"
        self.gm_mask = f"{self.reg_path_img}{'/'}" \
                       f"{'gm.nii.gz'}"
        self.gm_mask_thr = f"{self.reg_path_img}{'/'}" \
                           f"{'gm_thr.nii.gz'}"
        self.wm_mask = f"{self.reg_path_img}{'/'}" \
                       f"{'wm.nii.gz'}"
        self.wm_mask_thr = f"{self.reg_path_img}{'/'}" \
                           f"{'wm_thr.nii.gz'}"
        self.wm_edge = f"{self.reg_path_img}{'/'}" \
                       f"{'wm_edge.nii.gz'}"

        # Create empty tmp directories that do not yet exist
        reg_dirs = [
            self.reg_path,
            self.reg_path_mat,
            self.reg_path_warp,
            self.reg_path_img,
        ]
        for i in range(len(reg_dirs)):
            if not op.isdir(reg_dirs[i]):
                os.makedirs(reg_dirs[i], exist_ok=True)

    def gen_mask(self, mask):
        import os.path as op

        if op.isfile(self.t1w_brain) is False:
            import shutil
            shutil.copyfile(self.t1w, self.t1w_head)
            [self.t1w_brain, self.t1w_brain_mask] = utils.gen_mask(
                self.t1w_head, self.t1w_brain, mask
            )
        return

    def gen_tissue(self, wm_mask_existing, gm_mask_existing, overwrite):
        """
        A function to segment and threshold tissue types from T1w.
        """

        # Segment the t1w brain into probability maps
        if (
                wm_mask_existing is not None
                and gm_mask_existing is not None
                and overwrite is False
        ):
            print("Existing segmentations detected...")
            gm_mask = utils.orient_reslice(
                gm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
            wm_mask = utils.orient_reslice(
                wm_mask_existing, self.basedir_path, self.vox_size,
                overwrite=False)
        else:
            try:
                maps = utils.segment_t1w(self.t1w_brain, self.map_name)
                gm_mask = maps["gm_prob"]
                wm_mask = maps["wm_prob"]
            except RuntimeError as e:
                import sys
                print(e,
                      "Segmentation failed. Does the input anatomical image "
                      "still contained skull?"
                      )

        # Threshold GM to binary in func space
        math_img("img > 0.01", img=nib.load(gm_mask)).to_filename(
            self.gm_mask_thr)
        self.gm_mask = utils.apply_mask_to_image(gm_mask,
                                                    self.gm_mask_thr,
                                                    self.gm_mask)

        # Threshold WM to binary in dwi space
        math_img("img > 0.50", img=nib.load(wm_mask)).to_filename(
            self.wm_mask_thr)

        self.wm_mask = utils.apply_mask_to_image(wm_mask,
                                                    self.wm_mask_thr,
                                                    self.wm_mask)
        # Extract wm edge
        self.wm_edge = utils.get_wm_contour(wm_mask, self.wm_mask_thr,
                                               self.wm_edge)

        return

    def t1w2mni_align(self):
        """
        A function to perform alignment from T1w --> MNI.
        """
        import time

        # Create linear transform/ initializer T1w-->MNI
        utils.align(
            self.t1w_brain,
            self.input_mni_brain,
            xfm=self.t12mni_xfm_init,
            bins=None,
            interp="spline",
            out=None,
            dof=12,
            cost="mutualinfo",
            searchrad=True,
        )
        time.sleep(0.5)
        # Attempt non-linear registration of T1 to MNI template
        if self.simple is False:
            try:
                print(
                    f"Learning a non-linear mapping from T1w --> "
                )
                # Use FNIRT to nonlinearly align T1w to MNI template
                utils.align_nonlinear(
                    self.t1w_brain,
                    self.input_mni_brain,
                    xfm=self.t12mni_xfm_init,
                    out=self.t1_aligned_mni,
                    warp=self.warp_t1w2mni,
                    ref_mask=self.input_mni_mask,
                )
                time.sleep(0.5)
                # Get warp from T1w --> MNI
                utils.inverse_warp(
                    self.t1w_brain, self.mni2t1w_warp, self.warp_t1w2mni
                )
                time.sleep(0.5)
                # Get mat from MNI -> T1w
                self.mni2t1_xfm = utils.invert_xfm(self.t12mni_xfm_init,
                                                      self.mni2t1_xfm)

            except BaseException:
                # Falling back to linear registration
                utils.align(
                    self.t1w_brain,
                    self.input_mni_brain,
                    xfm=self.t12mni_xfm,
                    init=self.t12mni_xfm_init,
                    bins=None,
                    dof=12,
                    cost="mutualinfo",
                    searchrad=True,
                    interp="spline",
                    out=self.t1_aligned_mni,
                    sch=None,
                )
                time.sleep(0.5)
                # Get mat from MNI -> T1w
                self.t12mni_xfm = utils.invert_xfm(self.t12mni_xfm,
                                                      self.mni2t1_xfm,
                                                      )
        else:
            # Falling back to linear registration
            utils.align(
                self.t1w_brain,
                self.input_mni_brain,
                xfm=self.t12mni_xfm,
                init=self.t12mni_xfm_init,
                bins=None,
                dof=12,
                cost="mutualinfo",
                searchrad=True,
                interp="spline",
                out=self.t1_aligned_mni,
                sch=None,
            )
            time.sleep(0.5)
            # Get mat from MNI -> T1w
            self.t12mni_xfm = utils.invert_xfm(self.t12mni_xfm,
                                                  self.mni2t1_xfm
                                                  )
        return


if __name__ == "__main__":
    reg = FmriReg(
        basedir_path='result',
        t1w='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/BraTS2020NMI/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz',
        template_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/nmi152.nii.gz',
        template_mask_path='/data_smr/liuy/Challenge/PennBraTSReg/GIRNet/data/nmi152_mask.nii.gz'
    )
    reg.gen_mask(None)
    reg.t1w2mni_align()
