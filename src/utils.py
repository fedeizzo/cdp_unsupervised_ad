from argparse import ArgumentParser

# Arguments keys
DATA_DIR = 'data'
EPOCHS = "epochs"
BS = "bs"
LR = "lr"
TP = "tp"
FC = "fc"
NL = "nl"
SEED = "seed"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(f"--{DATA_DIR}", type=str, help="Data root directory path")
    parser.add_argument(f"--{EPOCHS}", type=int, help="Number of epochs", default=1)
    parser.add_argument(f"--{BS}", type=int, help="Batch size", default=8)
    parser.add_argument(f"--{LR}", type=float, help="Learning rate", default=0.001)
    parser.add_argument(f"--{TP}", type=float, help="Training data percentage", default=0.3)
    parser.add_argument(f"--{FC}", type=int, help="Features channels", default=8)
    parser.add_argument(f"--{NL}", type=int, help="Number of affine coupling layers", default=8)
    parser.add_argument(f"--{SEED}", type=int, help="Randomizing seed", default=0)

    return vars(parser.parse_args())
