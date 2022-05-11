from src.utils import *
from visualizations.utils import show_cdps
from data.utils import load_cdp_data


def visualize_print_simulation(model, data_loader, device, n_batches=1):
    for i, batch in enumerate(data_loader):
        templates = batch["template"]
        originals = batch["originals"][0]
        predictions, doubts = model(templates.to(device)).chunk(2, 1)

        for t, o, p, d in zip(templates, originals, predictions, doubts):
            show_cdps(
                [t, o, p, d],
                ["Template", "Original printed", "Estimated printed", "Doubt"]
            )

        if i + 1 >= n_batches:
            break


def visualize_anomaly_locations(model, loader, device, n_batches=1):
    for i, batch in enumerate(loader):
        original = batch["originals"][0]
        estimate, doubt = model(batch["template"].to(device)).chunk(2, 1)

        for o, e, d in zip(original, estimate, doubt):
            show_cdps([e, o, (e - o) ** 2], ["Estimated", "Original", "Anomaly Map"])

        for fake in batch["fakes"]:
            for idx, (f, e, d) in enumerate(zip(fake, estimate, doubt)):
                show_cdps([e, f, (e - f) ** 2, d], ["Estimated", "Fake", "Anomaly Map", "Doubt"])

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
    _, _, loader, _ = load_cdp_data(data_dir, 0, 0, 1, originals=originals)

    # Showing printing simulation
    visualize_print_simulation(model, loader, device)

    # Showing anomaly locations
    visualize_anomaly_locations(model, loader, device)


if __name__ == '__main__':
    main()
