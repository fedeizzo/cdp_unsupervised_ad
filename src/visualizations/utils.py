import matplotlib.pyplot as plt


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
