import matplotlib.pyplot as plt


def show_cdps(cdps, titles):
    """Shows CPDs in one line with the given titles for each CDP"""
    n = len(cdps)
    fig = plt.figure(figsize=(9, 13))

    for i, (cdp, title) in enumerate(zip(cdps, titles)):
        cdp = cdp.detach().cpu().numpy()[0].astype(float)

        sub_plot = fig.add_subplot(1, n, i + 1)
        sub_plot.imshow(cdp, cmap="gray")
        sub_plot.set_title(title)

    plt.show()
