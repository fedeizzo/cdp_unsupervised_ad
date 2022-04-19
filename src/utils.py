import torch
from argparse import ArgumentParser

# Arguments keys
DATA_DIR = 'data'
CATEGORY = 'category'
EPOCHS = "epochs"
ORIGINALS = "originals"
BS = "bs"
LR = "lr"
TP = "tp"
VP = "vp"
FC = "fc"
NL = "nl"
RL = "rl"
PRETRAINED = "pretrained"
FREEZE_BACKBONE = "freeze_backbone"
MODEL = "model"
SEED = "seed"


def set_reproducibility(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(f"--{DATA_DIR}", type=str, help="Data root directory path")
    parser.add_argument(f"--{CATEGORY}", type=str, default="bottle", help="Category for the mvtec dataset")
    parser.add_argument(f"--{EPOCHS}", type=int, help="Number of epochs", default=1)
    parser.add_argument(f"--{ORIGINALS}", choices=["55", "76"], help="Originals to be used for training.", default="55")
    parser.add_argument(f"--{BS}", type=int, help="Batch size", default=8)
    parser.add_argument(f"--{LR}", type=float, help="Learning rate", default=0.001)
    parser.add_argument(f"--{TP}", type=float, help="Training data percentage", default=0.4)
    parser.add_argument(f"--{VP}", type=float, help="Validation data percentage", default=0.1)
    parser.add_argument(f"--{FC}", type=int, help="Features channels", default=256)
    parser.add_argument(f"--{NL}", type=int, help="Number of affine coupling layers", default=16)
    parser.add_argument(f"--{RL}", type=int, help="Final resnet layer used as feature extractor", default=3)
    parser.add_argument(f"--{PRETRAINED}", action="store_true", help="Whether to use a pre-trained backbone")
    parser.add_argument(f"--{FREEZE_BACKBONE}", action="store_true", help="Whether to freeze the backbone or not.")
    parser.add_argument(f"--{MODEL}", type=str, help="Model path where net will be stored / restored", default=None)
    parser.add_argument(f"--{SEED}", type=int, help="Randomizing seed", default=0)

    return vars(parser.parse_args())


def get_device(verbose=True):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if verbose:
        print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if cuda else ""))
    return device
