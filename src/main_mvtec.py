import os
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet50, wide_resnet50_2

from anomalib.data.mvtec import MVTec
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from models.models import NormalizingFlowModel
from models.utils import get_backbone_resnet
from utils import *


def load_data(data_dir, category, bs):
    transform = A.Compose([A.Normalize(), ToTensorV2()])
    train_set = MVTec(data_dir, category, pre_process=transform, is_train=True)
    test_set = MVTec(data_dir, category, pre_process=transform, is_train=False)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    return train_loader, test_loader


def train_flow_model(train_loader, backbone, fc, n_layers, n_epochs, lr, freeze, device):
    # FastFlow Model
    flow_model = NormalizingFlowModel(backbone, in_channels=fc, n_layers=n_layers, freeze_backbone=freeze).to(device)
    optimizer = Adam(flow_model.parameters(), lr=lr)

    # Training loop
    flow_model.train()
    best_loss = float("inf")
    for epoch in range(n_epochs):
        loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            _, o, log_det_j = flow_model(x)
            batch_loss = torch.mean(torch.mean(0.5 * o ** 2, dim=[1, 2, 3]) - log_det_j)

            # Optimizing
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Collecting epoch loss
            loss += batch_loss.item() / len(train_loader)

        # Logging epoch loss
        log_str = f"Epoch {epoch + 1}/{n_epochs} loss: {loss:.3f}"

        # Storing best model yet
        if best_loss > loss:
            best_loss = loss
            torch.save(flow_model.state_dict(), "flow_model_sd.pt")
            log_str += " --> Stored best model ever."
        print(log_str)
    return flow_model


def test_flow_model(flow_model, test_loader, device):
    flow_model.eval()

    # Getting anomaly scores
    all_scores = {}
    with torch.no_grad():
        for batch in test_loader:
            labels = [ip.split("/")[-2] for ip in batch["image_path"]]
            images = batch["image"].to(device)

            _, out, _ = flow_model(images)
            scores = torch.mean(0.5 * out ** 2, dim=[1, 2, 3])

            for label, score in zip(labels, scores):
                try:
                    all_scores[label].append(scores)
                except Exception:
                    all_scores[label] = [score]

    # Plotting histogram of anomaly scores and storing the scores
    for label in all_scores.keys():
        scores = all_scores[label]
        plt.plot(np.arange(len(scores)), scores, label=label)
        np.save(f"{label}_scores.npy", np.array(scores))

    plt.title("Anomaly score for test CDPs")
    plt.legend()
    plt.savefig("anomaly_scores.png")


def main():
    # Getting program parameters
    args = parse_args()
    data_dir = args[DATA_DIR]  # Data directory
    category = args[CATEGORY]  # MVTec AD Category
    n_epochs = args[EPOCHS]  # Number of epochs
    bs = args[BS]  # Batch size
    lr = args[LR]  # Learning rate
    fc = args[FC]  # Features channels
    n_layers = args[NL]  # Number of affine coupling layers
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else ""))

    # Loading data
    train_loader, test_loader = load_data(data_dir, category, bs)

    # Resnet Backbone
    resnet = get_backbone_resnet(wide_resnet50_2, 3, 1024, fc, pretrained, rl)

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
        flow_model = train_flow_model(train_loader, resnet, fc, n_layers, n_epochs, lr, fb, device)

    # Testing loop
    test_flow_model(flow_model, test_loader, device)
    print("Program completed successfully!\n\n")


if __name__ == '__main__':
    main()
