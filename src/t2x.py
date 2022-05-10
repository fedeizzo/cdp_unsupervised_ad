import matplotlib.pyplot as plt

from torch.optim import Adam

from utils import *
from models.t2x import get_t2x_model
from data.utils import load_cdp_data
from data.transforms import *

DEFAULT_MODEL_PATH = "model.pt"


def join(path1, path2):
    return os.path.join(path1, path2)


def anomaly_fn_mse(printed, estimated):
    return torch.mean((printed - estimated) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


def anomaly_fn_max(printed, estimated):
    return torch.amax(torch.abs(printed - estimated), dim=[1, 2, 3]).detach().cpu().numpy()


def run_epoch(model, loader, optimizer, device, optimize=True, run_fakes=False):
    epoch_loss = 0.0
    o_scores, f_scores = [], [[] for _ in range(4)]
    for batch in loader:
        t = batch["template"].to(device)
        o = batch["originals"][0].to(device)
        o_hat = model(t)

        se = (o_hat - o) ** 2
        batch_loss = torch.mean(se)
        epoch_loss += batch_loss.item() / len(loader)

        if optimize:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        if run_fakes:
            o_scores.extend(torch.mean(se, dim=[1, 2, 3]).detach().cpu().numpy())

            fakes = batch["fakes"]
            for f_idx, f in enumerate(fakes):
                f = f.to(device)
                o_hat = model(f)
                f_scores[f_idx].extend(torch.mean((o_hat - o) ** 2, dim=[1, 2, 3]).detach().cpu().numpy())

    if run_fakes:
        return epoch_loss, np.array(o_scores), np.array(f_scores)
    return epoch_loss, None, None


def train(model, optim, train_loader, val_loader, device, epochs, model_path=DEFAULT_MODEL_PATH):
    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss, val_loss = 0.0, 0.0
        for batch in train_loader:
            t = batch["template"].to(device)
            o = batch["originals"][0].to(device)
            o_hat = model(t)

            batch_loss = torch.mean((o_hat - o) ** 2)

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            epoch_loss += batch_loss.item() / len(train_loader)

        for batch in val_loader:
            t = batch["template"].to(device)
            o = batch["originals"][0].to(device)
            o_hat = model(t)

            val_loss += torch.mean((o_hat - o) ** 2) / len(val_loader)

        epoch_str = f"Epoch {epoch + 1}/{epochs}\tTrain loss: {epoch_loss:.3f}\tVal loss: {val_loss:.3f}"
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, model_path)
            epoch_str += " --> Stored best model"
        print(epoch_str)


def test(test_loader, device, model_path=DEFAULT_MODEL_PATH, title=None, anomaly_fn=anomaly_fn_mse, dest="./"):
    model = torch.load(model_path, map_location=device)
    model.eval()

    o_scores = []
    f_scores = [[] for _ in range(4)]
    for batch in test_loader:
        t = batch["template"].to(device)
        o = batch["originals"][0].to(device)
        o_hat = model(t)

        o_scores.extend(anomaly_fn(o, o_hat))

        for idx, f in enumerate(batch["fakes"]):
            f = f.to(device)
            f_scores[idx].extend(anomaly_fn(f, o_hat))

    o_scores, f_scores = np.array(o_scores), np.array(f_scores)
    np.save(os.path.join(dest, "o_scores.npy"), o_scores)
    np.save(os.path.join(dest, "f_scores.npy"), f_scores)

    n_bins = len(o_scores) // 4
    plt.hist(o_scores, bins=n_bins, label="Originals")
    auc_roc_scores = []
    for f_name, f_score in zip(["Fakes 55/55", "Fakes 55/76", "Fakes 76/55", "Fakes 76/76"], f_scores):
        plt.hist(f_score, bins=n_bins, label=f_name)
        auc_roc_scores.append(get_roc_auc_score(o_scores, f_score))

    np.save(os.path.join(dest, "auc_scores.npy"), np.array(auc_roc_scores))

    plt.legend()
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(title)
    plt.savefig(os.path.join(dest, "anomaly_scores.png"))


def main():
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
        model = get_t2x_model().to(device)
        optim = Adam(model.parameters(), lr=lr)

        # Training loop
        train(model, optim, train_loader, val_loader, device, n_epochs, model_path)

    # Testing loop
    a_fn = anomaly_fn_mse
    print(f"\n\nTesting trained model at {model_path}")
    test(test_loader, device, model_path, f"Originals {originals} and fakes ({a_fn.__name__})", a_fn, result_dir)


if __name__ == '__main__':
    main()
