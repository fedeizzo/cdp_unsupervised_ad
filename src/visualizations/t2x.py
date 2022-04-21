import matplotlib.pyplot as plt


from utils import *
from data.utils import load_cdp_data


def show_cdps(cdps, titles):
    n = len(cdps)
    fig = plt.figure(figsize=(9, 13))

    for i, (cdp, title) in enumerate(zip(cdps, titles)):
        cdp = cdp.detach().cpu().numpy()[0].astype(float)

        sub_plot = fig.add_subplot(1, n, i + 1)
        sub_plot.imshow(cdp)
        sub_plot.set_title(title)

    plt.show()


def visualize_print_simulation(model, data_loader, device, n_batches=1):
    for i, batch in enumerate(data_loader):
        templates = batch["template"]
        originals = batch["originals"][0]
        predictions = model(templates.to(device))

        for t, o, p in zip(templates, originals, predictions):
            show_cdps(
                [t, o, p],
                ["Template", "Original printed", "Estimated printed"]
            )

        if i + 1 >= n_batches:
            break


def visualize_anomaly_locations(model, loader, device, n_batches=1):
    for i, batch in enumerate(loader):
        original = batch["originals"][0]
        estimate = model(batch["template"].to(device))

        for o, e in zip(original, estimate):
            show_cdps([e, o, (e - o) ** 2], ["Estimated", "Original", "Anomaly Map"])

        for fake in batch["fakes"]:
            for idx, (f, e) in enumerate(zip(fake, estimate)):
                show_cdps([e, f, (e - f) ** 2], ["Estimated", "Fake", "Anomaly Map"])

        if i + 1 >= n_batches:
            break


def main():
    # Loading arguments and setting reproducibility
    args = parse_args()
    model_path = args[MODEL]
    data_dir = args[DATA_DIR]
    originals = args[ORIGINALS]
    seed = args[SEED]
    set_reproducibility(seed)

    # Getting current device
    device = get_device()

    # Loading model
    model = torch.load(model_path, map_location=device).to(device)
    model.eval()

    # Loading data
    _, _, loader, _, _ = load_cdp_data(data_dir, 0, 0, 1, originals=originals)

    # Showing printing simulation
    visualize_print_simulation(model, loader, device)

    # Showing anomaly locations
    visualize_anomaly_locations(model, loader, device)


if __name__ == '__main__':
    main()
