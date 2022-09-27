import os
import json
import random
from enum import Enum
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import torch

# Arguments keys
CONF = 'conf'
DATA_DIR = 'data'
EPOCHS = "epochs"
MODE = "mode"
ORIGINALS = "originals"
BS = "bs"
LR = "lr"
TP = "tp"
VP = "vp"
NO_TRAIN = "no_train"
RESULT_DIR = "result_dir"
ORIG_NAMES = "orig_names"
FAKE_NAMES = "fake_names"
SEED = "seed"


# Modes
MODES = ["t2x", "t2xa", "x2t", "x2ta", "both", "both_a"]

def set_reproducibility(seed):
    """Sets the reproducibility of the experiments with the given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    """Parses program arguments and returns a dictionary adressable with the above-defined macros"""
    parser = ArgumentParser()
    parser.add_argument(f"--{CONF}", type=str, help="Path to the file containing the configuration")

    args = vars(parser.parse_args())

    if args[CONF] is not None:
        f = open(args[CONF], "r")
        args = json.load(f)
        f.close()
    else:
        print("ERROR: Program takes --conf as argument.")
        exit()

    return args


def get_device(verbose=True):
    """Gets a CUDA device if available"""
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if verbose:
        print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if cuda else ""))
    return device


def join(path1, path2):
    """Joins two paths"""
    return os.path.join(path1, path2)


def create_dir(path):
    """Creates a directory if it does not exist already"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_roc_auc_score(o_scores, f_scores):
    """Returns the ROC AUC score for the given original and fake scores. Originals should score lower."""
    y_true = [*[0 for _ in range(len(o_scores))], *[1 for _ in range(len(f_scores))]]
    y_score = [*o_scores, *f_scores]
    return roc_auc_score(y_true, y_score)


def store_scores(o_scores, f_scores, dest):
    """Stores scores into NumPy arrays in the dest folder."""
    o_scores, f_scores = np.array(o_scores), np.array(f_scores)
    np.save(os.path.join(dest, "o_scores.npy"), o_scores)
    np.save(os.path.join(dest, "f_scores.npy"), f_scores)


def store_hist_picture(o_scores, f_scores, dest,
                       title="Anomaly scores", orig_names=None, fakes_names=None, pic_name="anomaly_scores.png", alpha=0.5):
    """Computes and stores the histogram for the original and fakes, based on their scores"""
    if orig_names is None:
        orig_names = [f"Originals {i+1}" for i in range(len(o_scores))]

    if fakes_names is None:
        fakes_names = [f"Fakes {i+1}" for i in range(len(o_scores))]


    o_scores, f_scores = np.array(o_scores), np.array(f_scores)
    n_bins = len(o_scores[0]) // 4


    for o_s, name in zip(o_scores, orig_names):
        plt.hist(o_s, bins=n_bins, alpha=alpha, label=name)

    for f_s, name in zip(f_scores, fakes_names):
        plt.hist(f_s, bins=n_bins, alpha=alpha, label=name)


    auc_roc_scores = {}
    for o_name, o_score in zip(orig_names, o_scores):
        auc_roc_scores[o_name] = {}
        for f_name, f_score in zip(fakes_names, f_scores):
            auc_roc_scores[o_name][f_name] = get_roc_auc_score(o_score, f_score)

    auc_roc_scores = json.dumps(auc_roc_scores, indent=4)
    with open(os.path.join(dest, "auc_scores.json"), "w") as file:
        file.write(auc_roc_scores)
        file.close()

    plt.legend()
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(title)
    plt.savefig(os.path.join(dest, pic_name))
