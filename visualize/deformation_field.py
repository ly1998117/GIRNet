# -*- encoding: utf-8 -*-
"""
@File    :   deformation_field.py   
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/5 8:45 PM   liu.yang      1.0         None
"""
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def get_grid(imgshape):
    return np.mgrid[0:imgshape[0], 0:imgshape[1], 0:imgshape[2]].transpose(1, 2, 3, 0)[..., [2, 1, 0]]


def get_field_img(pos_flow):
    assert pos_flow.ndim == 3
    pos_flow = pos_flow.transpose(1, 2, 0)
    grid = get_grid(pos_flow.shape[:-1])
    pos_flow = pos_flow + grid
    for i in range(2):
        pos_flow[i, ...] = 2 * (pos_flow[i, ...] / (pos_flow.shape[:-1][i] - 1) - 0.5)
    return pos_flow


def field2contour(grid, ax=None, img=None):
    '''
    grid--image_grid used to show deform field
    type: numpy ndarray, shape： (h, w, 2), value range：(-1, 1)
    '''
    assert grid.ndim == 3
    for i in range(2):
        grid[..., i] = 2 * (grid[..., i] / (grid.shape[:-1][i] - 1) - 0.5)

    x = np.arange(-1, 1, 2 / grid.shape[0])
    y = np.arange(-1, 1, 2 / grid.shape[1])
    X, Y = np.meshgrid(y, x)
    Z1 = grid[:, :, 0]  # remove the dashed line
    # Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1]

    if ax is None:
        if img is not None:
            plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.contour(X, Y, Z1, 20, colors='r', linewidths=1, linestyles='solid')
        # plt.clabel(contours, inline=True, fontsize=.5)
        plt.contour(X, Y, Z2, 20, colors='r', linewidths=1, linestyles='solid')
        # plt.clabel(contours, inline=True, fontsize=.5)

        plt.xticks(()), plt.yticks(())  # remove x, y ticks
        plt.title('deform field')
    else:
        if img is not None:
            ax.imshow(img, cmap='gray')
        ax.contour(X, Y, Z1, 30, colors='r', linewidths=.5)
        ax.contour(X, Y, Z2, 30, colors='r', linewidths=.5)
        ax.axis('off')


if __name__ == "__main__":
    df = nib.load(
        '/data_58/liuy/GIRNet/result/GIRNet5_BraTSPseudoHisto/noexclude221_MNI152_ntuc_histomatch_sym_y2xrec_win15/fold_0_t1/deformation_field/198/valid/OAS1_0019_MR1-BraTS20_308/x2y_df.nii.gz').get_fdata()
    x = nib.load(
        '/data_58/liuy/GIRNet/result/GIRNet5_BraTSPseudoHisto/noexclude221_MNI152_ntuc_histomatch_sym_y2xrec_win15/fold_0_t1/deformation_field/198/valid/OAS1_0019_MR1-BraTS20_308/x.nii.gz').get_fdata()
    regular_grid = get_grid(x.shape) + df
    # x = (x - np.min(x)) / (np.max(x) - np.min(x))
    regular_grid = regular_grid[..., [2, 1, 0]]
    # for i in range(3):
    #     regular_grid[..., i] = regular_grid[..., i] / (x.shape[i] - 1) * 2 - 1
    field2contour(regular_grid[..., 80, :], img=x[..., 80])
    plt.show()
