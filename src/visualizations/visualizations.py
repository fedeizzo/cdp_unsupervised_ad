from utils.utils import *
from visualizations.utils import *
from data.utils import load_cdp_data
from models.utils import load_models, forward


def viz_t2x(t, x, fakes, x_hat):
    # TODO
    pass


def viz_x2t(t, x, f, t_hat):
    # TODO
    pass


def viz_t2xa(t, x, f, x_hat, c):
    # TODO
    pass


def viz_x2ta(t, x, f, t_hat, c):
    # TODO
    pass


def viz_both(t, x, f, x_hat, t_hat):
    # TODO
    pass


def viz_both_a(t, x, f, x_hat, cx, t_hat, ct):
    # TODO
    pass


def visualize_batches(mode, models, loader, device, n_batches=1):
    for i, batch in enumerate(loader):
        templates = batch["template"].to(device)
        originals = batch["originals"][0].to(device)

        _, *result = forward(mode, models, templates, originals)

        # Displaying output
        for m, fn in zip(AVAILABLE_MODES, [viz_t2x, viz_x2t, viz_t2xa, viz_x2ta, viz_both, viz_both_a]):
            if mode == m:
                fn(templates, originals, batch["fakes"].to(device), *result)

        if i + 1 >= n_batches:
            break


def main():
    # Loading arguments and setting reproducibility
    args = parse_args()
    mode = args[MODE]
    data_dir = args[DATA_DIR]
    result_dir = args[RESULT_DIR]
    originals = args[ORIGINALS]
    seed = args[SEED]
    set_reproducibility(seed)

    # Getting current device
    device = get_device()

    # Loading model
    models = load_models(mode, result_dir, device)

    # Loading data
    _, _, loader, _ = load_cdp_data(data_dir, 0, 0, 1, originals=originals)

    # Visualizing model outputs
    visualize_batches(mode, models, loader, device)


if __name__ == '__main__':
    main()
