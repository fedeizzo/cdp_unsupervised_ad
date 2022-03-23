import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

from torchvision.models.resnet import resnet18

from data.cdp_dataset import get_split
from models.models import NormalizingFlowModel, adjust_resnet_input
from utils import parse_args, DATA_DIR, EPOCHS, BS, LR, TP, FC, NL, SEED


def main():
    # Getting program parameters
    args = parse_args()
    data_dir = args[DATA_DIR]  # Data directory
    n_epochs = args[EPOCHS]    # Number of epochs
    bs = args[BS]              # Batch size
    lr = args[LR]              # Learning rate
    tp = args[TP]              # Training data percentage
    fc = args[FC]              # Features channels
    n_layers = args[NL]        # Number of affine coupling layers
    seed = args[SEED]          # Random seed

    # Logging program arguments
    print("Running main program with the following arguments:")
    print(args)

    # Reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loading data
    t_dir = os.path.join(data_dir, 'templates')
    # TODO: Multiple models for each original
    # x_dirs = [os.path.join(data_dir, 'originals_55'), os.path.join(data_dir, 'originals_76')]
    x_dirs = [os.path.join(data_dir, 'originals_55')]
    f_dirs = [os.path.join(data_dir, 'fakes_55_55'), os.path.join(data_dir, 'fakes_55_76'),
              os.path.join(data_dir, 'fakes_76_55'), os.path.join(data_dir, 'fakes_76_76')]

    n_orig, n_fakes = len(x_dirs), len(f_dirs)

    train_set, _, test_set = get_split(t_dir, x_dirs, f_dirs, train_percent=tp, val_percent=0)
    train_loader, test_loader = DataLoader(train_set, batch_size=bs), DataLoader(test_set, batch_size=bs)

    # Resnet Backbone
    resnet = adjust_resnet_input(resnet18, in_channels=1, pretrained=False)
    modules = list(resnet.children())[:-2]
    modules.append(nn.Conv2d(512, fc, (1, 1)))
    resnet = nn.Sequential(*modules)

    # FastFlow Model
    flow_model = NormalizingFlowModel(resnet, in_channels=fc, n_layers=n_layers).to(device)
    optimizer = Adam(flow_model.parameters(), lr=lr)

    # Multivariate Gaussian with mean 0 and identity covariance
    normal = MultivariateNormal(torch.zeros(fc * 22 * 22).to(device), torch.eye(fc * 22 * 22).to(device))

    # Training loop
    best_loss = float("inf")
    for epoch in range(n_epochs):
        loss = 0.0
        for batch in train_loader:
            batch_loss = 0.0
            for o_idx in range(n_orig):
                # Getting the model output (normally distributed for in-distribution data) and log of det of Jacobian
                x = batch['originals'][o_idx].to(device)
                _, o, log_det_j = flow_model(x)

                # Computing Normalizing flows loss
                batch_loss -= torch.mean(
                    normal.log_prob(o.reshape(-1, fc * 22 * 22)) +
                    log_det_j
                )
            # Optimizing
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Collecting epoch loss
            loss += batch_loss.item() / n_orig

        # Logging epoch loss
        log_str = f"Epoch {epoch + 1} loss: {loss:.3f}"

        if best_loss > loss:
            best_loss = loss
            torch.save(flow_model, "flow_model.pt")
            log_str += " --> Stored best model ever."
        print(log_str)

    # Testing loop
    o_probs, f_probs = [[] for _ in range(n_orig)], [[] for _ in range(n_fakes)]
    for batch in test_loader:
        # Collecting log probabilities for originals
        for o_idx in range(n_orig):
            x = batch["originals"][o_idx].to(device)
            _, out, _ = flow_model(x)
            prob = normal.log_prob(out.reshape(-1, fc * 22 * 22))
            o_probs[o_idx].extend([p.item() for p in prob])

        # Collecting log probabilities for fakes
        for f_idx in range(n_fakes):
            x = batch["fakes"][f_idx].to(device)
            _, out, _ = flow_model(x)
            prob = normal.log_prob(out.reshape(-1, fc * 22 * 22))
            f_probs[f_idx].extend([p.item() for p in prob])

    for o_idx, o_name in zip(range(n_orig), ["55", "76"]):
        plt.hist(o_probs[o_idx], label=f"Originals {o_name}")

    for f_idx, f_name in zip(range(n_fakes), ["55/55", "55/76", "76/55", "76/76"]):
        plt.hist(f_probs[f_idx], label=f"Fakes {f_name}")

    plt.title("Anomaly score for test CDPs")
    plt.legend()
    plt.savefig("anomaly_scores.png")
    print("Program completed successfully!\n\n")


if __name__ == '__main__':
    main()
