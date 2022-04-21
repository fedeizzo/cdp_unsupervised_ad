import os
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam

from torchvision.models.resnet import resnet50, wide_resnet50_2

from data.utils import load_cdp_data
from models.normalizing_flows import NormalizingFlowModel
from models.utils import get_backbone_resnet
from utils import *


def train_flow_model(train_loader, backbone, fc, n_layers, n_epochs, lr, freeze, n_orig, device):
    # FastFlow Model
    flow_model = NormalizingFlowModel(backbone, in_channels=fc, n_layers=n_layers, freeze_backbone=freeze).to(device)
    optimizer = Adam(flow_model.parameters(), lr=lr)

    # Training loop
    flow_model.train()
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
                batch_loss += torch.mean(
                    torch.mean(0.5 * o ** 2, dim=[1, 2, 3]) -
                    log_det_j
                )

            # Optimizing
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Collecting epoch loss
            loss += batch_loss.item() / (len(train_loader) * n_orig)

        # Logging epoch loss
        log_str = f"Epoch {epoch + 1}/{n_epochs} loss: {loss:.3f}"

        # Storing best model yet
        if best_loss > loss:
            best_loss = loss
            torch.save(flow_model.state_dict(), "flow_model_sd.pt")
            log_str += " --> Stored best model ever."
        print(log_str)
    return flow_model


def test_flow_model(flow_model, test_loader, n_orig, n_fakes, device):
    flow_model.eval()
    o_scores, f_scores = [[] for _ in range(n_orig)], [[] for _ in range(n_fakes)]

    # Computing all anomaly scores
    with torch.no_grad():
        for batch in test_loader:
            # Collecting log probabilities for originals
            for o_idx in range(n_orig):
                x = batch["originals"][o_idx].to(device)
                _, out, _ = flow_model(x)
                scores = torch.mean(0.5 * out ** 2, dim=[1, 2, 3])
                o_scores[o_idx].extend([s.item() for s in scores])

            # Collecting log probabilities for fakes
            for f_idx in range(n_fakes):
                x = batch["fakes"][f_idx].to(device)
                _, out, _ = flow_model(x)
                scores = torch.mean(0.5 * out ** 2, dim=[1, 2, 3])
                f_scores[f_idx].extend([s.item() for s in scores])

    # Plotting histogram of anomaly scores
    for o_idx, o_name in zip(range(n_orig), ["55", "76"]):
        plt.plot(np.arange(len(o_scores[o_idx])), o_scores[o_idx], label=f"Originals {o_name}")

    for f_idx, f_name in zip(range(n_fakes), ["55/55", "55/76", "76/55", "76/76"]):
        plt.plot(np.arange(len(f_scores[f_idx])), f_scores[f_idx], label=f"Fakes {f_name}")
    plt.title("Anomaly score for test CDPs")
    plt.legend()
    plt.savefig("anomaly_scores.png")

    # Storing anomaly scores to files
    np.save("o_scores.npy", np.array(o_scores))
    np.save("f_scores.npy", np.array(f_scores))


def main():
    # Getting program parameters
    args = parse_args()
    data_dir = args[DATA_DIR]  # Data directory
    n_epochs = args[EPOCHS]  # Number of epochs
    bs = args[BS]  # Batch size
    lr = args[LR]  # Learning rate
    tp = args[TP]  # Training data percentage
    fc = args[FC]  # Features channels
    n_layers = args[NL]  # Number of affine coupling layers
    originals = args[ORIGINALS]
    rl = args[RL]
    pretrained = args[PRETRAINED]  # Whether backbone will be pre-trained on ImageNet or not
    fb = args[FREEZE_BACKBONE]
    model_path = args[MODEL]  # Path to pre-trained model (if any)
    seed = args[SEED]  # Random seed

    # Logging program arguments
    print("Running main program with the following arguments:")
    print(args)

    # Reproducibility
    set_reproducibility(seed)

    # Device
    device = get_device()

    # Loading data
    train_loader, _, test_loader, n_orig, n_fakes = load_cdp_data(data_dir, tp, 0, bs, originals=originals)

    # Resnet Backbone
    resnet = get_backbone_resnet(wide_resnet50_2, 1, 1024, fc, pretrained, rl)

    # Getting the flow model
    if model_path is not None and os.path.isfile(model_path):
        # Loading pre-trained model
        flow_model = NormalizingFlowModel(resnet, fc, n_layers)
        flow_model.load_state_dict(torch.load(model_path, map_location=device))
        flow_model = flow_model.to(device)
    else:
        # Training loop
        if model_path is not None and not os.path.isfile(model_path):
            print(f"Could not find pre-trained state dict at {model_path}. Training a Flow Model from scratch.")
        flow_model = train_flow_model(train_loader, resnet, fc, n_layers, n_epochs, lr, fb, n_orig, device)

    # Testing loop
    test_flow_model(flow_model, test_loader, n_orig, n_fakes, device)
    print("Program completed successfully!\n\n")


if __name__ == '__main__':
    main()
