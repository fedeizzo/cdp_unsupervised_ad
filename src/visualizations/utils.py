import matplotlib.pyplot as plt
import torch


def show_cdps(cdps, titles, crop_size=None, vmin=0, vmax=1):
    """Shows CPDs in one line with the given titles for each CDP"""
    n = len(cdps)
    fig = plt.figure(figsize=(9, 13))

    for i, (cdp, title) in enumerate(zip(cdps, titles)):
        cdp = cdp.detach().cpu().numpy()

        while len(cdp.shape) > 2:
            cdp = cdp[0]

        cdp = cdp.astype(float)

        if crop_size:
            h, w = cdp.shape
            h_min, h_max = h // 2 - crop_size // 2, h // 2 + crop_size // 2
            w_min, w_max = w // 2 - crop_size // 2, w // 2 + crop_size // 2
            cdp = cdp[h_min:h_max, w_min:w_max]

        sub_plot = fig.add_subplot(1, n, i + 1)
        sub_plot.imshow(cdp, cmap="gray", vmin=vmin, vmax=vmax)
        sub_plot.set_title(title)

    plt.show()


def dump_intermediate_images(
    t, x, x_hat, f, anomaly_map_original, anomaly_map_fake, epoch, result_filename
):
    plt.tight_layout()
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    for i, ax, title in zip(
        [t, x, x_hat, f, anomaly_map_original, anomaly_map_fake],
        axs.flatten(),
        [
            "template $t$",
            "original $x$",
            "predicted $\hat{x}$",
            "fake $f$",
            f"anomaly $\hat{{x}}$/$x$ = {int(torch.sum(anomaly_map_original))}",
            f"anomaly $\hat{{x}}$/$f$ = {int(torch.sum(anomaly_map_fake))}",
        ],
    ):
        # here it is taken the first element to remove the batch and channel dimensions
        img = i.cpu().detach()[0, 0].numpy()
        width = max(img.shape[0] // 10, 50)
        center_x = img.shape[0] // 2
        center_y = img.shape[1] // 2
        ax.imshow(
            img[
                center_x - width // 2 : center_x + width // 2,
                center_y - width // 2 : center_y + width // 2,
            ],
            cmap="Greys_r",
            interpolation="none",
        )
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=15)

    fig.suptitle(f"Epoch {epoch}", fontsize=15)
    fig.savefig(result_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
