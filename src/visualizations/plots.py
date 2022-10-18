import os
import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from skimage.metrics import structural_similarity

from src.data.cdp_dataset import CDPDataset

# Plot style
sns.set_style("darkgrid")
font = {
    'size': 14,
}
matplotlib.rc('font', **font)

# Paths
PROJECT_DIR = "/Users/bp/Desktop/Projects/cdp_unsupervised_ad"
DATA_DIR = os.path.join(PROJECT_DIR, "datasets/mobile")
ORIG_DIR = os.path.join(DATA_DIR, "orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG")
FAKE_DIR = os.path.join(DATA_DIR, "fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55")

# Common strings
IPHONE = "iPhone12Pro"
SAMSUNG = "SamsungGN20U"
CONF = "ss100_focal12_apperture1"


def mse(t, y):
    return torch.mean((t - y) ** 2)


def pcorr(t, y):
    return cv2.matchTemplate(t.numpy().flatten(), y.numpy().flatten(), cv2.TM_CCOEFF_NORMED).ravel()[0]


def ssim(t, y):
    return structural_similarity(t.numpy().squeeze(), y.numpy().squeeze())


def plot_hist_and_roc(fns, fn_names, dataset, o_names, f_names):
    metrics = {}
    for fn, fn_name in zip(fns, fn_names):
        o_scores = [[] for _ in range(len(dataset.x_dirs))]
        f_scores = [[] for _ in range(len(dataset.f_dirs))]

        # Collecting scores for all originals and fakes
        for codes in dataset:
            t = codes["template"]

            for i, o in enumerate(codes["originals"]):
                o_scores[i].append(fn(t, o))

            for i, f in enumerate(codes["fakes"]):
                f_scores[i].append(fn(t, f))

        metrics[fn_name] = (o_scores, f_scores)

    # Plotting histograms
    for metric in metrics.keys():
        o_scores, f_scores = metrics[metric]
        for name, o_s in zip(o_names, o_scores):
            plt.hist(o_s, label=name, density=True)

        for name, f_s in zip(f_names, f_scores):
            plt.hist(f_s, label=name, density=True)
        plt.title(f"Histogram with {metric}")
        plt.legend()
        plt.xlabel(metric)
        plt.show()

    # Plotting ROC curves
    for o_idx, o_name in enumerate(o_names):
        for metric in metrics.keys():
            o_scores, f_scores = metrics[metric]
            y_true = [0 for _ in range(len(o_scores[o_idx]))]
            y_score = [s for s in o_scores[o_idx]]

            for f_name, f_s in zip(f_names, f_scores):
                y_true = y_true + [1 for _ in range(len(f_s))]
                y_score = y_score + [s for s in f_s]

            fpr, tpr, t = roc_curve(y_true, y_score)
            area = auc(fpr, tpr)
            if area < 0.5:
                fpr, tpr, t = roc_curve(-(np.array(y_true) - 1), y_score)
                area = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{metric} (area = {area:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"Originals {o_name} and fakes", fontdict={"size": 18, "weight": "bold"})
        plt.legend()
        plt.savefig(f"{o_name} ({fn_names})")
        plt.show()


def main():
    # Directories of data
    t_dir = os.path.join(DATA_DIR, "orig_template")
    x_dirs = [os.path.join(ORIG_DIR, f"{IPHONE}_run{run}_{CONF}/rcod_hist") for run in range(1, 7)]
    # x_dirs.extend([os.path.join(ORIG_DIR, f"{SAMSUNG}_run{run}_{CONF}/rcod_hist") for run in range(1, 7)])
    f_dirs = [os.path.join(FAKE_DIR, f"{IPHONE}_run{run}_{CONF}/rcod_hist") for run in range(1, 7)]
    # f_dirs.extend([os.path.join(FAKE_DIR, f"{SAMSUNG}_run{run}_{CONF}/rcod_hist") for run in range(1, 7)])

    # Names of originals and fakes
    o_names = [f"iPhone ({run})" for run in range(1, 7)]
    # o_names.extend([f"Samsung ({run})" for run in range(1, 7)])
    f_names = ["Fake " + name for name in o_names]

    # Dataset
    print("Loading dataset...")
    dataset = CDPDataset(t_dir, x_dirs, f_dirs, np.arange(14_440 + 1))
    print(f"Dataset loaded. Found {len(dataset)} shared CDP across all directories.")

    # Plots
    plot_hist_and_roc([mse, pcorr, ssim], ["MSE", "PCorr", "SSIM"], dataset, o_names, f_names)


if __name__ == "__main__":
    main()
