import os
from enum import Enum
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import torch

# Arguments keys
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
SEED = "seed"

# Originals and Fakes
AVAILABLE_ORIGINALS = ("55", "76")
FAKE_NAMES = ("Fakes 55/55", "Fakes 55/76", "Fakes 76/55", "Fakes 76/76")
FAKE_NUMBERS = ("55/55", "55/76", "76/55", "76/76")


# Modes
class Mode(Enum):
    """Enum with all possible modalities for program execution"""
    MODE_T2X = 't2x'
    MODE_T2XA = 't2xa'
    MODE_X2T = 'x2t'
    MODE_X2TA = 'x2ta'
    MODE_BOTH = 'both'
    MODE_BOTH_A = 'both_a'


def set_reproducibility(seed):
    """Sets the reproducibility of the experiments with the given seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    """Parses program arguments and returns a dictionary adressable with the above-defined macros"""
    parser = ArgumentParser()
    parser.add_argument(f"--{DATA_DIR}", type=str, help="Data root directory path")
    parser.add_argument(f"--{EPOCHS}", type=int, help="Number of epochs", default=100)
    parser.add_argument(f"--{MODE}", type=Mode, choices=list(Mode), help="Kind of model used", default=list(Mode)[0])
    parser.add_argument(f"--{ORIGINALS}", choices=AVAILABLE_ORIGINALS, help="Originals to be used for training",
                        default=AVAILABLE_ORIGINALS[0])
    parser.add_argument(f"--{BS}", type=int, help="Batch size", default=16)
    parser.add_argument(f"--{LR}", type=float, help="Learning rate", default=0.001)
    parser.add_argument(f"--{TP}", type=float, help="Training data percentage", default=0.4)
    parser.add_argument(f"--{VP}", type=float, help="Validation data percentage", default=0.1)
    parser.add_argument(f"--{NO_TRAIN}", action="store_true", help="Whether to train a new model")
    parser.add_argument(f"--{RESULT_DIR}", type=str, help="Path where all results will be stored", default="./results")
    parser.add_argument(f"--{SEED}", type=int, help="Randomizing seed", default=0)

    return vars(parser.parse_args())


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
        os.mkdir(path)


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
                       title="Anomaly scores", pic_name="anomaly_scores.png", fakes_names=FAKE_NAMES, alpha=0.5):
    """Computes and stores the histogram for the original and fakes, based on their scores"""
    o_scores, f_scores = np.array(o_scores), np.array(f_scores)
    n_bins = len(o_scores) // 4
    plt.hist(o_scores, bins=n_bins, alpha=alpha, label="Originals")
    auc_roc_scores = []
    for f_name, f_score in zip(fakes_names, f_scores):
        plt.hist(f_score, bins=n_bins, alpha=alpha, label=f_name)
        auc_roc_scores.append(get_roc_auc_score(o_scores, f_score))

    np.save(os.path.join(dest, "auc_scores.npy"), np.array(auc_roc_scores))

    plt.legend()
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(title)
    plt.savefig(os.path.join(dest, pic_name))
