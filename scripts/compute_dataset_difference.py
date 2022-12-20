import argparse
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import cv2

from os import listdir
from os.path import join
from numpy.random import choice


def image_norm(img: np.ndarray) -> np.ndarray:
    img = img - np.min(img, axis=(0, 1), keepdims=True)
    img = img / np.max(img, axis=(0, 1), keepdims=True)
    return img.astype(np.float32)


def imread(path: str, *flags):
    return cv2.imread(str(path), *flags)


def main(template_dir: str, originals_dir: str, fakes_dir: str, result_filename: str):
    plt.rc("legend", fontsize=15)
    templates = listdir(template_dir)
    diff_t_o = []
    diff_t_f = []
    diff_o_f = []
    for path in templates:
        try:
            t = image_norm(imread(join(template_dir, path), -1).astype(np.float32))
            o = image_norm(imread(join(originals_dir, path), -1).astype(np.float32))
            f = image_norm(imread(join(fakes_dir, path), -1).astype(np.float32))
            diff_t_o.append(np.abs(t - o))
            diff_t_f.append(np.abs(t - f))
            diff_o_f.append(np.abs(o - f))
        except Exception:
            continue
    diff_t_o = np.array(diff_t_o)
    diff_t_f = np.array(diff_t_f)
    diff_o_f = np.array(diff_o_f)
    A = diff_t_o.sum(0) / diff_t_o.shape[0]
    B = diff_t_f.sum(0) / diff_t_f.shape[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in zip(axs, [A, B], ["Templates/Originals", "Templates/Fakes"]):
        img = img / img.max()
        img[img < 0.75] = 0
        ax.imshow(img, cmap="Greys_r", interpolation="none")
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=15)
    fig.savefig("idk.pdf")
    import pdb

    pdb.set_trace()

    plt.hist(diff_o_f, alpha=0.35, label="Difference originals fakes")
    plt.hist(diff_t_f, alpha=0.35, label="Difference templates fakes")
    plt.hist(diff_t_o, alpha=0.35, label="Difference templates originals")
    plt.xlabel("MSE")
    plt.ylabel("Amount")
    plt.title("HPI 76 Indigo 1x1 Dataset analysis", fontsize=15)
    plt.legend()
    plt.savefig(result_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--template-dir",
        type=str,
        required=True,
        help="Directory containing templates.",
    )
    parser.add_argument(
        "-o",
        "--originals-dir",
        type=str,
        required=True,
        help="Directory containing originals.",
    )
    parser.add_argument(
        "-f",
        "--fakes-dir",
        type=str,
        required=True,
        help="Directory containing fakes.",
    )
    parser.add_argument(
        "-r",
        "--result-filename",
        type=str,
        required=True,
        help="Where the result will be saved.",
    )
    args = parser.parse_args()
    main(args.template_dir, args.originals_dir, args.fakes_dir, args.result_filename)
