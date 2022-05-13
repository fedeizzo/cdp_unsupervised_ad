import os
import torch

from utils.utils import Mode

DEFAULT_T2X_MODEL_PATH = "t2x.pt"
DEFAULT_T2XA_MODEL_PATH = "t2xa.pt"
DEFAULT_X2T_MODEL_PATH = "x2t.pt"
DEFAULT_X2TA_MODEL_PATH = "x2ta.pt"

MODE_TO_PATHS = {
    Mode.MODE_T2X: [DEFAULT_T2X_MODEL_PATH],
    Mode.MODE_T2XA: [DEFAULT_T2XA_MODEL_PATH],
    Mode.MODE_X2T: [DEFAULT_X2T_MODEL_PATH],
    Mode.MODE_X2TA: [DEFAULT_X2TA_MODEL_PATH],
    Mode.MODE_BOTH: [DEFAULT_T2X_MODEL_PATH, DEFAULT_X2T_MODEL_PATH],
    Mode.MODE_BOTH_A: [DEFAULT_T2XA_MODEL_PATH, DEFAULT_X2TA_MODEL_PATH]
}


def forward(mode, models, t, x):
    """Method which returns the loss for the given mode using the given models, templates and originals"""
    if mode == Mode.MODE_T2X:
        x_hat = models[0](t)
        return torch.mean((x_hat - x) ** 2)
    elif mode == Mode.MODE_T2XA:
        x_hat, c = models[0](t).chunk(2, 1)
        diff = (x - x_hat)
        return torch.mean(diff ** 2) + torch.mean(((1 - torch.abs(diff)) - c) ** 2)
    elif mode == Mode.MODE_X2T:
        t_hat = models[0](x)
        return torch.mean((t_hat - t) ** 2)
    elif mode == Mode.MODE_X2TA:
        t_hat, c = models[0](x).chunk(2, 1)
        diff = (t - t_hat)
        return torch.mean(diff ** 2) + torch.mean(((1 - torch.abs(diff)) - c) ** 2)
    elif mode == Mode.MODE_BOTH:
        t2x_model, x2t_model = models[0], models[1]
        x_hat = t2x_model(t)
        t_hat = x2t_model(x)

        l_cyc = torch.mean((x2t_model(x_hat) - t) ** 2) + torch.mean((t2x_model(t_hat) - x) ** 2)
        l_standard = torch.mean((x_hat - x) ** 2) + torch.mean((t_hat - t) ** 2)
        return l_cyc + l_standard
    elif mode == Mode.MODE_BOTH_A:
        t2xa_model, x2ta_model = models[0], models[1]
        x_hat, cx = t2xa_model(t).chunk(2, 1)
        t_hat, ct = x2ta_model(x).chunk(2, 1)

        l_cyc = torch.mean((x2ta_model(x_hat).chunk(2, 1)[0] - t) ** 2) + \
                torch.mean((t2xa_model(t_hat).chunk(2, 1)[0] - x) ** 2)

        x_diff, t_diff = x_hat - x, t_hat - t
        l_standard = torch.mean(x_diff ** 2) + torch.mean(((1 - torch.abs(x_diff)) - cx) ** 2) + \
                     torch.mean(t_diff ** 2) + torch.mean(((1 - torch.abs(t_diff)) - ct) ** 2)

        return l_cyc + l_standard
    else:
        raise KeyError(f"Unknown mode {mode}!")


def store_models(mode, models, dest):
    """Stores the models in the result directory."""
    for model, name in zip(models, MODE_TO_PATHS[mode]):
        torch.save(model, os.path.join(dest, name))


def load_models(mode, models_dir, device):
    """Loads the trained models depending on the mode onto the device."""
    models = []
    for name in MODE_TO_PATHS[mode]:
        models.append(torch.load(os.path.join(models_dir, name), map_location=device))
    return models
