import torch
from argparse import ArgumentParser

# Arguments keys
DATA_DIR = 'data'
EPOCHS = "epochs"
BS = "bs"
LR = "lr"
TP = "tp"
FC = "fc"
NL = "nl"
PRETRAINED = "pretrained"
MODEL = "model"
SEED = "seed"


def set_reproducibility(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(f"--{DATA_DIR}", type=str, help="Data root directory path")
    parser.add_argument(f"--{EPOCHS}", type=int, help="Number of epochs", default=1)
    parser.add_argument(f"--{BS}", type=int, help="Batch size", default=8)
    parser.add_argument(f"--{LR}", type=float, help="Learning rate", default=0.001)
    parser.add_argument(f"--{TP}", type=float, help="Training data percentage", default=0.3)
    parser.add_argument(f"--{FC}", type=int, help="Features channels", default=256)
    parser.add_argument(f"--{NL}", type=int, help="Number of affine coupling layers", default=16)
    parser.add_argument(f"--{PRETRAINED}", action="store_true", help="Whether to use a pre-trained backbone")
    parser.add_argument(f"--{MODEL}", type=str, help="Trained model to test", default=None)
    parser.add_argument(f"--{SEED}", type=int, help="Randomizing seed", default=0)

    return vars(parser.parse_args())
