import os

import numpy as np

import torch.nn as nn
from torch.optim import Adam

from utils import *
from data.utils import load_cdp_data

DEFAULT_MODEL_PATH = "t2x_nf_model.pt"


def make_grid_masks(h, w, device=None):
    mask_a = torch.tensor([
        [
            [0 if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1) else 1]
            for i in range(w)]
        for j in range(h)]).reshape(1, h, w)

    mask_b = 1 - mask_a

    if device:
        mask_a = mask_a.to(device)
        mask_b = mask_b.to(device)

    return mask_a, mask_b


def default_network():
    return nn.Sequential(
        nn.Conv2d(1, 10, 3, padding=1),
        nn.Conv2d(10, 10, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(10, 2, 3, padding=1),
    )


class AffineCoupling(nn.Module):
    def __init__(self, mask, network_fn):
        super(AffineCoupling, self).__init__()
        self.mask = mask
        self.network = network_fn()

    def forward(self, x):
        x1 = x * self.mask
        st = self.network(x1)
        s, t = st.chunk(2, 1)

        s = (1 - self.mask) * torch.sigmoid(s + 2)
        t = (1 - self.mask) * t

        out = x1 + (x * s + t)
        log_det = torch.sum(torch.log(s + self.mask), dim=[1, 2, 3])

        return out, log_det

    def backward(self, out):
        x1 = self.mask * out
        x2 = out - x1

        st = self.network(x1)
        s, t = st.chunk(2, 1)

        s = (1 - self.mask) * torch.sigmoid(s + 2)
        t = (1 - self.mask) * t

        x2 = (x2 - t) / (s + self.mask)
        log_det = 1 / torch.sum(torch.log(s + self.mask), dim=[1, 2, 3])

        return x1 + x2, log_det

    def zero_init(self):
        for parameter in self.parameters():
            parameter.data.zero_()


class CDPNF(nn.Module):
    def __init__(self, n_steps, network_fn, device, h=684, w=684):
        super(CDPNF, self).__init__()
        self.mask_a, self.mask_b = make_grid_masks(h, w, device)
        self.n_steps = n_steps
        self.layers = nn.ModuleList([
            AffineCoupling(self.mask_a if i % 2 == 0 else self.mask_b, network_fn) for i in range(n_steps)
        ])

        for l in self.layers:
            l.zero_init()

    def forward(self, x):
        out, log_det = x, 0.0
        for layer in self.layers:
            out, ld = layer(out)
            log_det += ld
        return out, log_det

    def backward(self, out):
        # Running backward through all layers starting from the last
        x, log_det = out, 0
        n = len(self.layers)
        for i in range(len(self.layers)):
            x, ld = self.layers[n - 1 - i].backward(x)
            log_det += ld
        return x, log_det


def train(model, optim, train_loader, val_loader, n_epochs, device, result_dir):
    model = model.to(device)
    model.train()

    d_sqrt = np.sqrt(np.prod(train_loader.dataset[0]["template"].shape))

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        train_loss = 0.0
        for batch in train_loader:
            t, x = batch["template"].to(device), batch["originals"][0].to(device)
            out, log_det = model(t)

            log_prob = torch.sum((out - x) ** 2 / d_sqrt, dim=[1, 2, 3])

            optim.zero_grad()
            batch_loss = torch.mean(log_prob - log_det)
            batch_loss.backward()
            optim.step()

            train_loss += batch_loss.item() * len(log_prob) / len(train_loader.dataset)

        val_loss = 0.0
        for batch in val_loader:
            t, x = batch["template"].to(device), batch["originals"][0].to(device)
            out, log_det = model(t)

            log_prob = torch.sum((out - x) ** 2 / d_sqrt, dim=[1, 2, 3])
            batch_loss = torch.mean(log_prob - log_det)

            val_loss += batch_loss.item() * len(log_prob) / len(train_loader.dataset)

        epoch_str = f"Epoch {epoch + 1}/{n_epochs}: " \
                    f"Train loss -> {train_loss:.2f}\t" \
                    f"Val loss -> {val_loss:.2f}"
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(result_dir, DEFAULT_MODEL_PATH))
            epoch_str += " --> Stored best model"
        print(epoch_str)


def test(test_loader, device, result_dir):
    model = torch.load(os.path.join(result_dir, DEFAULT_MODEL_PATH), map_location=device)
    model.eval()

    d_sqrt = np.sqrt(np.prod(test_loader.dataset[0]["template"].shape))

    o_scores, f_scores = [], [[], [], [], []]
    for batch in test_loader:
        t, x = batch["template"].to(device), batch["originals"][0].to(device)
        estimate, _ = model(t)

        o_scores.extend([s.item() for s in torch.mean((estimate - x) ** 2 / d_sqrt, dim=[1, 2, 3])])

        for i, f in enumerate(batch["fakes"]):
            f = f.to(device)
            f_scores[i].extend([s.item() for s in torch.mean((estimate - f) ** 2 / d_sqrt, dim=[1, 2, 3])])

    np.save(os.path.join(result_dir, "o_scores.npy"), np.array(o_scores))
    np.save(os.path.join(result_dir, "f_scores.npy"), np.array(f_scores))


def main():
    # Parameters
    args = parse_args()
    epochs = args[EPOCHS]
    data_dir = args[DATA_DIR]
    result_dir = args[RESULT_DIR]
    tp = args[TP]
    vp = args[VP]
    bs = args[BS]
    lr = args[LR]
    nl = args[NL]
    seed = args[SEED]
    originals = args[ORIGINALS]
    print(args)

    # Setting reproducibility
    set_reproducibility(seed)

    # Creating result dir
    create_dir(result_dir)

    # CDPs height and width
    h, w = 684, 684

    # Program device
    device = get_device()

    # Loading data
    train_loader, val_loader, test_loader, _, _ = load_cdp_data(data_dir, tp, vp, bs, originals=originals)

    # Creating and training model
    model = CDPNF(nl, default_network, device, h, w)
    optim = Adam(model.parameters(), lr=lr)
    train(model, optim, train_loader, val_loader, epochs, device, result_dir)

    test(test_loader, device, result_dir)
    print("Program finished successfully")


if __name__ == '__main__':
    main()
