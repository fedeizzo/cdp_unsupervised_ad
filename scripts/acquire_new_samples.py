# https://stackoverflow.com/questions/60359398/python-detect-a-qr-code-from-an-image-and-crop-using-opencv
import argparse
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import cv2

from os import listdir, makedirs
from os.path import join
from numpy.random import choice
from typing import List


def acquire(img_names: List[str], filename: str):
    # Load imgae, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(filename)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter for QR code
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c, path in zip(cnts, img_names):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        ar = w / float(h)
        # if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.3):
        if len(approx) == 4 and area > 1000:
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            ROI = original[y : y + h, x : x + w]
            cv2.imwrite(path, ROI)

    # cv2.imshow("thresh", thresh)
    # cv2.imshow("close", close)
    # cv2.imshow("image", image)
    # cv2.imshow("ROI", ROI)
    # cv2.waitKey()


def main(template_dir: str, output_dir: str, seed: int, acquisition: str):
    makedirs(output_dir, exist_ok=True)
    rnd.seed(seed)
    img_names = choice(np.array(listdir(template_dir)), 8, replace=False)
    img_names = list(map(lambda x: join(output_dir, x), img_names))
    acquire(img_names, acquisition)


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
        "--output-dir",
        type=str,
        required=True,
        help="Directory where new acquisitions will be saved.",
    )
    parser.add_argument(
        "-a",
        "--acquisition",
        type=str,
        required=True,
        help="PNG/PDF of the scanned file.",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.seed, args.acquisition)
