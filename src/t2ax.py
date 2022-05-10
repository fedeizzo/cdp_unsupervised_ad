from torch.optim import Adam

from utils import *
from models.t2x import get_t2ax_model
from data.utils import load_cdp_data
from data.transforms import *

DEFAULT_MODEL_PATH = "model.pt"


def anomaly_fn_weighted_mse(printed, estimated, doubt):
    """Returns the MSEs (per samples) between printed and estimated images weighted by the doubt tensor"""
    # Normalizing doubt
    doubt = doubt - torch.min(doubt)
    doubt = doubt / torch.max(doubt)

    # Returning weighted (according to doubt) MSE
    return torch.mean((1 - doubt) * (printed - estimated) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


def anomaly_fn_weighted_std(printed, estimated, doubt):
    """Returns the STDs (per samples) between printed and estimated images weighted by the doubt tensor"""
    # Normalizing doubt
    doubt = doubt - torch.min(doubt)
    doubt = doubt / torch.max(doubt)

    # Returning weighted (according to doubt) STD
    return torch.std((1 - doubt) * (printed - estimated) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


def train(model, optim, train_loader, val_loader, device, epochs, model_path=DEFAULT_MODEL_PATH):
    """Trains the and stores the best validation model in the specified path"""
    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss, val_loss = 0.0, 0.0
        for batch in train_loader:
            t = batch["template"].to(device)
            o = batch["originals"][0].to(device)
            o_hat, doubt = model(t).chunk(2, 1)
            error = (o - o_hat) ** 2

            batch_loss = torch.mean(error) + torch.mean((error - doubt) ** 2)

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            epoch_loss += batch_loss.item() / len(train_loader)

        for batch in val_loader:
            t = batch["template"].to(device)
            o = batch["originals"][0].to(device)
            o_hat, doubt = model(t).chunk(2, 1)
            error = (o - o_hat) ** 2

            val_loss += (torch.mean(error) + torch.mean((error - doubt) ** 2)) / len(val_loader)

        epoch_str = f"Epoch {epoch + 1}/{epochs}\tTrain loss: {epoch_loss:.3f}\tVal loss: {val_loss:.3f}"
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, model_path)
            epoch_str += " --> Stored best model"
        print(epoch_str)


def test(test_loader, device, model_path=DEFAULT_MODEL_PATH, title=None, anomaly_fn=anomaly_fn_weighted_mse, dest="./"):
    """Tests the trained model on test data and stores anomaly scores, AUC scores and histograms."""
    model = torch.load(model_path, map_location=device)
    model.eval()

    o_scores = []
    f_scores = [[] for _ in range(4)]
    for batch in test_loader:
        t = batch["template"].to(device)
        o = batch["originals"][0].to(device)
        o_hat, doubt = model(t).chunk(2, 1)

        o_scores.extend(anomaly_fn(o, o_hat, doubt))

        for idx, f in enumerate(batch["fakes"]):
            f = f.to(device)
            f_scores[idx].extend(anomaly_fn(f, o_hat, doubt))

    store_scores(o_scores, f_scores, dest)
    store_hist_picture(o_scores, f_scores, dest, title)


def main():
    """Trains (if pre-trained is not specified) and stores a model. Then, tests the model and stores test infos."""

    # Parameters
    args = parse_args()
    n_epochs = args[EPOCHS]
    base_dir = args[DATA_DIR]
    result_dir = args[RESULT_DIR]
    originals = args[ORIGINALS]
    lr = args[LR]
    bs = args[BS]
    tp = args[TP]
    vp = args[VP]
    model_path = args[MODEL]
    print(args)

    if model_path is None:
        model_path = os.path.join(result_dir, DEFAULT_MODEL_PATH)

    # Creating result directory
    create_dir(result_dir)

    # Getting program device
    device = get_device()

    # Loading data
    train_loader, val_loader, test_loader, _, _ = load_cdp_data(base_dir, tp, vp, bs, originals=originals)

    # Training loop
    if not os.path.isfile(model_path):
        print(f"Model at {model_path} not found: Training a new model.")

        # Creating one-convolution model
        model = get_t2ax_model().to(device)
        optim = Adam(model.parameters(), lr=lr)

        # Training loop
        train(model, optim, train_loader, val_loader, device, n_epochs, model_path)

    # Testing loop
    a_fn = anomaly_fn_weighted_std
    print(f"\n\nTesting trained model at {model_path}")
    test(test_loader, device, model_path, f"Originals {originals} and fakes ({a_fn.__name__})", a_fn, result_dir)


if __name__ == '__main__':
    main()
