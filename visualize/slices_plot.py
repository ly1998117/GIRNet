# -*- encoding: utf-8 -*-
"""
@File    :   slices_plot.py   
@Contact :   liu.yang.mine@gmail.com
@License :   (C)Copyright 2018-2022
 
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/1 2:56 PM   liu.yang      1.0         None
"""
import os

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting
from .deformation_field import field2contour
from utils.utils import make_dirs


def single_flow(slices_in,  # the 2D slices
                ax,
                img_indexing=True,  # whether to match the image view, i.e. flip y axis
                quiver_width=None,
                scale=1):  # note quiver essentially draws quiver length = 1/scale

    # input processing
    nb_plots = 1
    assert len(slices_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
    if slices_in.shape[-1] != 2:
        slices_in = slices_in[..., :-1]

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    if img_indexing:
        slices_in = np.flipud(slices_in)

    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns

    # turn off axis
    ax.axis('off')

    u, v = slices_in[..., 0], slices_in[..., 1]
    colors = np.arctan2(u, v)
    colors[np.isnan(colors)] = 0
    norm = Normalize()
    norm.autoscale(colors)

    colormap = cm.winter

    # show figure
    ax.quiver(u, v,
              color=colormap(norm(colors).flatten()),
              angles='xy',
              units='xy',
              width=quiver_width,
              scale=scale[0])
    ax.axis('equal')

    return ax


def plot_slices(slices_in,  # the 2D slices
                coordinate=None,  # the landmarks
                titles=None,  # list of titles
                cmaps=None,  # list of colormaps
                norms=None,  # list of normalizations
                do_colorbars=False,  # option to show colorbars on each slice
                grid=False,  # option to plot the images in a grid or a single row
                width=15,  # width in in
                show=True,  # option to actually show the plot (plt.show())
                axes_off=True,
                imshow_args=None,
                dpi=200,
                labelsize=6,
                fontsize=10,
                rotate=0
                ):
    '''
    plot a grid of slices (2d images)
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    slices_in = [np.rot90(img, -rotate // 90, axes=(0, 1)) for img in slices_in]
    fig, axs = plt.subplots(rows, cols, dpi=dpi)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            # ax.title.set_text(titles[i], fontsize=fontsize)
            ax.set_title(titles[i], fontsize=fontsize)

        # show figure
        if 'flow' in titles[i]:
            single_flow(slices_in[i], ax)
        elif 'contour' in titles[i]:
            field2contour(slices_in[i], ax)
        else:
            if slices_in[i].shape[-1] == 2:
                from mmcv import flow2rgb
                slices_in[i] = flow2rgb(slices_in[i])
            # if slices_in[i].shape[-1] == 3:
            #     cmaps[i] = 'viridis'
            im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])
            if coordinate is not None:
                # plot landmark's coordinate
                if 'moving' in titles[i] or 'fix' in titles[i]:
                    ax.plot(coordinate[i][1], coordinate[i][0], 'o', color='r')
                    ax.plot(coordinate[i][1], coordinate[i][0], '.', color='r')

            # colorbars
            # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
            if do_colorbars and cmaps[i] is not None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.02)
                cb = fig.colorbar(im_ax, cax=cax)
                cb.ax.tick_params(labelsize=labelsize)

    # clear axes that are unnecessary
    for i in range(nb_plots, cols * rows):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if show:
        plt.tight_layout()
        plt.show()

    return (fig, axs)


def normalize(x):
    return (255 * (x - x.min()) / (x.max() - x.min() + 1e-10)).astype(np.uint8).clip(0, 255)


def plot_img3d(images, title, rows=10, width=15, name=None, fontsize=50, cmap=None, logs=None):
    slices = []
    for img in images:
        img = img[..., ::img.shape[-1] // rows]
        img = img[..., : rows]
        img = img.permute(0, 4, 1, 2, 3).squeeze(0).cpu()
        slices.append(img)
    del images
    plot_img(slices, title, width, name, cmap, fontsize, logs)


def plot_img(slices, title=None, width=15, name=None, cmap=None, fontsize=25, logs=None):
    if cmap is None:
        cmap = ['gray'] * len(slices)
    elif isinstance(cmap, str):
        cmap = [cmap] * len(slices)

    slices = [normalize(i.permute(0, 2, 3, 1).squeeze(-1).cpu().detach().numpy()) for i in slices]

    rows, cols = slices[0].shape[0], len(slices)
    fig, axs = plt.subplots(rows, cols, dpi=300)
    if logs is not None:
        plt.suptitle(logs, size=fontsize)
    nb_plots = slices[0].shape[0] * cols

    slices_in = []
    cmaps = []
    titles = []
    for sls in zip(*slices):
        slices_in.extend(sls)
        cmaps.extend(cmap)
        titles.extend(title)

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')
        ax.set_title(titles[i], fontsize=fontsize)
        ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest")

    # clear axes that are unnecessary
    for i in range(nb_plots, cols * rows):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]
        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if name is None:
        plt.show()
    else:
        make_dirs(os.path.dirname(name))
        plt.savefig(name)
        plt.close('all')


if __name__ == "__main__":
    pass
