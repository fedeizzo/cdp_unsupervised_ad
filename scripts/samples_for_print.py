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


def main(template_dir: str, seed: int, output_filepath: str):
    rnd.seed(seed)
    img_names = choice(np.array(listdir(template_dir)), 8, replace=False)
    fig, axs = plt.subplots(4, 2, figsize=(8.27, 11.69))
    axs = axs.flatten()
    for path, ax in zip(img_names, axs):
        img = image_norm(imread(join(template_dir, path), -1).astype(np.float32))
        img[np.where(img == 1)] = 255
        ax.imshow(img, cmap="Greys_r", interpolation="none")
        #ax.set_title(path)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    print(f"Savings {img_names}")
    fig.savefig(output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", type=int, required=True, help="Seed used to sample CDPs."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing templates or esimates.",
    )
    parser.add_argument(
        "-o",
        "--output-filepath",
        type=str,
        required=True,
        help="Filepath where the result will be saved.",
    )
    args = parser.parse_args()
    main(args.input_dir, args.seed, args.output_filepath)
