import torch

from utils.utils import Mode


def get_anomaly_score(mode, models, t, y):
    """Given the mode, pre-trained model(s), templates and test printed CDPs, returns the anomaly scores per each sample
    based on the provided mode."""
    if mode == Mode.MODE_T2X:
        return mse_af(y, models[0](t))
    if mode == Mode.MODE_X2T:
        return mse_af(t, models[0](y))
    if mode == Mode.MODE_T2XA:
        y_hat, confidence = models[0](t).chunk(2, 1)
        return confidence_mse_af(y, y_hat, confidence)
    if mode == Mode.MODE_X2TA:
        t_hat, confidence = models[0](y).chunk(2, 1)
        return confidence_mse_af(t, t_hat, confidence)
    if mode == Mode.MODE_BOTH:
        t2x_model, x2t_model = models[0], models[1]
        t2x_score = get_anomaly_score(Mode.MODE_T2X, [t2x_model], t, y)
        x2t_score = get_anomaly_score(Mode.MODE_X2T, [x2t_model], t, y)
        return (t2x_score ** 2 + x2t_score ** 2) ** 0.5
    if mode == Mode.MODE_BOTH_A:
        t2xa_model, x2ta_model = models[0], models[1]
        t2xa_score = get_anomaly_score(Mode.MODE_T2XA, [t2xa_model], t, y)
        x2ta_score = get_anomaly_score(Mode.MODE_X2TA, [x2ta_model], t, y)
        return (t2xa_score ** 2 + x2ta_score ** 2) ** 0.5


def mse_af(actual, estimate):
    """Returns the batch-wise MSE between printed codes and model estimation given by the template"""
    return torch.mean((estimate - actual) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


def confidence_mse_af(actual, estimate, confidence):
    """Returns the MSEs (per samples) between printed and estimated images weighted by the confidence tensor"""
    return torch.mean(confidence * (estimate - actual) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()
