from src.visualizations.utils import *
from src.utils.utils import *
from src.data.utils import load_cdp_data
from src.models.utils import load_models, forward


def viz_t2x(t, x, fakes, out_o, out_f, crop_size=50):
    # Obtaining print estimation from template
    x_hat = out_o[0]

    # Show template, original, printed estimation and difference between original and estimation
    show_cdps(
        [t[0], x[0], x_hat[0], torch.abs(x[0] - x_hat[0])],
        ["Template", "Original", "Estimation", "Difference"],
        crop_size
    )

    # Doing the same for fakes
    for idx, f_number in zip(range(4), FAKE_NUMBERS):
        f = fakes[idx][0]
        show_cdps(
            [t[0], f, x_hat[0], torch.abs(f - x_hat[0])],
            ["Template", f"Fake {f_number}", "Estimation", "Difference"],
            crop_size
        )


def viz_x2t(t, x, fakes, out_o, out_f, crop_size=50):
    # Obtaining estimated templates
    t_hat_o = out_o[0]
    t_hat_fs = [of[0] for of in out_f]

    # Showing Original, Template, Template estimation (from original) and difference
    show_cdps(
        [x[0], t[0], t_hat_o[0], torch.abs(t[0] - t_hat_o[0])],
        ["Original", "Template", "Template estimation", "Difference"],
        crop_size
    )

    # Doing the same for all fakes
    for idx, (f_number, t_hat_f) in enumerate(zip(FAKE_NUMBERS, t_hat_fs)):
        show_cdps(
            [fakes[idx][0], t[0], t_hat_f[0], torch.abs(t[0] - t_hat_f[0])],
            [f"Fake {f_number}", "Template", "Template estimation", "Difference"],
            crop_size
        )


def viz_t2xa(t, x, fakes, out_o, out_f, crop_size=50):
    # Obtaining print estimation and confidence from template
    x_hat, c_x = out_o

    # Show template, original, printed estimation, difference between original and estimation and confidence
    show_cdps(
        [t[0], x[0], x_hat[0], torch.abs(x[0] - x_hat[0]), c_x[0]],
        ["Template", "Original", "Estimation", "Difference", "Confidence"],
        crop_size
    )

    # Doing the same for fakes
    for idx, f_number in zip(range(4), FAKE_NUMBERS):
        f = fakes[idx][0]
        c_f = out_f[idx][1]
        show_cdps(
            [t[0], f, x_hat[0], torch.abs(f - x_hat[0]), c_f[0]],
            ["Template", f"Fake {f_number}", "Estimation", "Difference", "Confidence"],
            crop_size
        )


def viz_x2ta(t, x, fakes, out_o, out_f, crop_size=50):
    # Obtaining estimated templates and confidences
    t_hat_o, c_t = out_o

    # Showing Original, Template, Template estimation (from original) and difference
    show_cdps(
        [x[0], t[0], t_hat_o[0], torch.abs(t[0] - t_hat_o[0]), c_t[0]],
        ["Original", "Template", "Template estimation", "Difference", "Confidence"],
        crop_size
    )

    # Doing the same for all fakes
    for idx, (f_number, of) in enumerate(zip(FAKE_NUMBERS, out_f)):
        t_hat_f, c_f = of
        show_cdps(
            [fakes[idx][0], t[0], t_hat_f[0], torch.abs(t[0] - t_hat_f[0]), c_f[0]],
            [f"Fake {f_number}", "Template", "Template estimation", "Difference", "Confidence"],
            crop_size
        )


def viz_both(t, x, fakes, out_o, out_f, crop_size=50):
    # Obtaining estimations of printed original and template (from originals)
    x_hat_o, t_hat_o = out_o

    # Obtaining estimations of printed original and template (from fakes)
    x_hat_fs, t_hat_fs = [], []

    # Collecting fake estimations in lists of tuples
    for x_hat_f, t_hat_f in out_f:
        x_hat_fs.append((x_hat_f,))
        t_hat_fs.append((t_hat_f,))

    # Visualizing T2X and X2T estimations
    viz_t2x(t, x, fakes, (x_hat_o,), x_hat_fs, crop_size)
    viz_x2t(t, x, fakes, (t_hat_o,), t_hat_fs, crop_size)


def viz_both_a(t, x, fakes, out_o, out_f, crop_size=50):
    # Obtaining estimations of printed original and template + confidences (from originals)
    x_hat_o, ctx_o, t_hat_o, cxt_o = out_o

    # Obtaining estimations of printed original and template + confidences (from fakes)
    x_hat_fs, ctx_f, t_hat_fs, cxt_f = [], [], [], []

    # Collecting fake estimations and confidences lists of tuples
    for x_hat_f, cx, t_hat_f, ct in out_f:
        x_hat_fs.append((x_hat_f, cx))
        t_hat_fs.append((t_hat_f, ct))

    # Visualizing T2XA and X2TA estimations
    viz_t2xa(t, x, fakes, (x_hat_o, ctx_o), x_hat_fs, crop_size)
    viz_x2ta(t, x, fakes, (t_hat_o, cxt_o), t_hat_fs, crop_size)


def visualize_batches(mode, models, loader, device, n_batches=1):
    for i, batch in enumerate(loader):
        templates = batch["template"].to(device)
        originals = batch["originals"][0].to(device)
        all_fakes = [f.to(device) for f in batch["fakes"]]

        results_original = forward(mode, models, templates, originals)[1:]
        all_results_fakes = [forward(mode, models, templates, fake)[1:] for fake in all_fakes]

        # Displaying output
        for m, fn in zip(list(Mode), [viz_t2x, viz_t2xa, viz_x2t, viz_x2ta, viz_both, viz_both_a]):
            if mode == m:
                fn(templates, originals, all_fakes, results_original, all_results_fakes)

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
    print(f"Visualizing outputs of model(s) in {result_dir} using mode {mode} and originals {originals} (seed {seed}).")
    visualize_batches(mode, models, loader, device)


if __name__ == '__main__':
    main()
