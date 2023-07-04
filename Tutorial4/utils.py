import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms.functional as Ft

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, row_title=None, figsize=(200, 200), **imshow_kwargs):
    if not isinstance(imgs[0], list) and imgs[0].ndim == 3:
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(
        figsize=figsize, nrows=num_rows, ncols=num_cols, squeeze=False
    )
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def show(imgs, figsize=(10, 10)):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = Ft.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def save_gif(n_sample, n_classes, x_gen_store, save_dir="./", ep=0, w=0):
    fig, axs = plt.subplots(
        nrows=int(n_sample / n_classes),
        ncols=n_classes,
        sharex=True,
        sharey=True,
        figsize=(8, 3),
    )

    def animate_diff(i, x_gen_store):
        print(
            f"gif animating frame {i} of {x_gen_store.shape[0]}",
            end="\r",
        )
        plots = []
        for row in range(int(n_sample / n_classes)):
            for col in range(n_classes):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                plots.append(
                    axs[row, col].imshow(
                        -x_gen_store[i, (row * n_classes) + col, 0],
                        cmap="gray",
                        vmin=(-x_gen_store[i]).min(),
                        vmax=(-x_gen_store[i]).max(),
                    )
                )
        return plots

    ani = FuncAnimation(
        fig,
        animate_diff,
        fargs=[x_gen_store],
        interval=200,
        blit=False,
        repeat=True,
        frames=x_gen_store.shape[0],
    )
    ani.save(
        save_dir + f"gif_ep{ep}_w{w}.gif",
        dpi=100,
        writer=PillowWriter(fps=5),
    )
    print("saved image at " + save_dir + f"gif_ep{ep}_w{w}.gif")
