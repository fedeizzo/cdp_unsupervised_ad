import os

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.optim import Adam

from utils import *
from data.utils import load_cdp_data
from models.models import NormalizingFlowModel

DEFAULT_FILE_PATH = "simple_flow_model_sd.pt"


def train_simple_model(nf_model, n_epochs, train_loader, optim, device):
    nf_model.train()

    best_loss = float("inf")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            originals = batch["originals"][0].to(device)

            _, out, log_det = nf_model(originals)
            batch_loss = torch.mean(0.5 * torch.sum(out ** 2, dim=[1, 2, 3]) - log_det)

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            epoch_loss += batch_loss.item() / len(originals)
        epoch_str = f"Epoch {epoch + 1}/{n_epochs} loss: {epoch_loss}"

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(nf_model.state_dict(), DEFAULT_FILE_PATH)
            epoch_str += " --> stored best model ever"

        print(epoch_str)


def test_simple_model(nf_model, test_loader, n_fakes, device):
    original_anomaly_scores = []
    fakes_anomaly_scores = [[] for _ in range(n_fakes)]

    nf_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            original = batch["originals"][0].to(device)

            _, out, _ = nf_model(original)
            original_anomaly_scores.extend(torch.sum(0.5 * out ** 2, dim=[1, 2, 3]).flatten().numpy())

            for f_idx in range(n_fakes):
                fake = batch["fakes"][f_idx].to(device)

                _, out, _ = nf_model(fake)
                fakes_anomaly_scores[f_idx].extend(torch.sum(0.5 * out ** 2, dim=[1, 2, 3]).flatten().numpy())

    # Plotting anomaly scores
    plt.plot(np.arange(len(original_anomaly_scores)), original_anomaly_scores, label="Originals")

    for i, fake_anomaly_scores in enumerate(fakes_anomaly_scores):
        plt.plot(np.arange(len(fake_anomaly_scores)), fake_anomaly_scores, label=f"Fakes {f_idx + 1}")

    plt.title("Anomaly score for test CDPs")
    plt.legend()
    plt.savefig("anomaly_scores.png")


def main():
    # Program arguments
    args = parse_args()
    data_dir = args[DATA_DIR]
    n_epochs = args[EPOCHS]
    lr = args[LR]
    tp = args[TP]
    bs = args[BS]

    # Program device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading CDP data
    train_loader, test_loader, n_original, n_fakes = load_cdp_data(data_dir, tp, bs, return_stack=True)

    # Creating Normalizing Flow (NF) model
    nf_model = NormalizingFlowModel(nn.Identity(), 2, n_layers=16, freeze_backbone=False, permute=False).to(device)
    optim = Adam(nf_model.parameters(), lr)

    # Training model if not found
    if not os.path.isfile(DEFAULT_FILE_PATH):
        print("Training a simple Normalizing flow model")
        train_simple_model(nf_model, n_epochs, train_loader, optim, device)

    # Loading pre-trained model
    nf_model.load_state_dict(torch.load(DEFAULT_FILE_PATH, map_location=device))

    # Evaluating model
    test_simple_model(nf_model, test_loader, n_fakes, device)


if __name__ == '__main__':
    main()
