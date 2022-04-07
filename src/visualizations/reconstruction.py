import os

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, wide_resnet50_2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from anomalib.data.mvtec import MVTec

from data.cdp_dataset import CDPDataset
from models.models import NormalizingFlowModel
from models.utils import get_backbone_resnet
from utils import parse_args, DATA_DIR, MODEL, PRETRAINED, FC, NL, RL, CATEGORY


def get_heatmap(tensor):
    assert tensor.dim() == 3, f"visualize_tensor_heatmap() takes as input a 3-dimensional vector, but a " \
                              f"{tensor.dim()}-dimensional vector was given."
    tensor = tensor.detach()
    heat_map = torch.mean(torch.abs(tensor), dim=[0]).numpy()

    return heat_map


def show_images(images, h=1):
    """Given a list of images with associated titles, shows the images"""
    fig = plt.figure(figsize=(10, 7))
    n = len(images)
    w = int(np.ceil(n / h))
    for i, (img, title) in enumerate(images):
        fig.add_subplot(h, w, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
    plt.show()


def visualize_sample(model, sample, with_manipulation=False):
    # Running original forward
    features, out, _ = model(sample.unsqueeze(0))
    reconstruction = model.reverse(out)
    f, o, r = get_heatmap(features[0]), get_heatmap(out[0]), get_heatmap(reconstruction[0])

    if with_manipulation:
        # Running manipulation forward
        manipulated = torch.clone(sample)
        manipulated[:, 292:392, 292:392] = 1
        features_m, out_m, _ = model(manipulated.unsqueeze(0))
        reconstruction_m = model.reverse(out_m)
        fm, om, rm = get_heatmap(features_m[0]), get_heatmap(out_m[0]), get_heatmap(reconstruction_m[0])

        show_images(
            [(sample[0], "Original"), (f, "Features"), (o, "Mapped to normal"), (r, "Reconstructed features"),
             (manipulated[0], "Manipulated"), (fm, "Features"), (om, "Mapped to normal"),
             (rm, "Reconstructed features")],
            2
        )
    else:
        show_images([(sample[0], "Original"), (f, "Features"), (o, "Mapped to normal"), (r, "Reconstructed features")])


def mvtec_show(data_dir, model, category):
    # Loading data
    transform = A.Compose([A.Normalize(), ToTensorV2()])
    dataset = MVTec(data_dir, category, pre_process=transform, is_train=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Visualizing reconstruction and reconstruction after introduction of anomaly
    for batch in loader:
        images = batch["image"]

        for image in images:
            visualize_sample(model, image)


def cdp_show(data_dir, model):
    # Loading data
    t_dir = os.path.join(data_dir, 'templates')
    x_dirs = [os.path.join(data_dir, 'originals_55')]
    f_dirs = [os.path.join(data_dir, 'fakes_55_55'), os.path.join(data_dir, 'fakes_55_76'),
              os.path.join(data_dir, 'fakes_76_55'), os.path.join(data_dir, 'fakes_76_76')]
    dataset = CDPDataset(t_dir, x_dirs, f_dirs, np.arange(721), load=False)
    loader = DataLoader(dataset, 8)

    # Visualizing test data
    for batch in loader:
        original = batch["originals"][0][0]
        f55_55, f55_76 = batch["fakes"][0][0], batch["fakes"][1][0]
        f76_55, f76_76 = batch["fakes"][2][0], batch["fakes"][3][0]

        visualize_sample(model, original, with_manipulation=True)
        visualize_sample(model, f55_55)
        visualize_sample(model, f55_76)
        visualize_sample(model, f76_55)
        visualize_sample(model, f76_76)

        break


def main():
    # Getting program arguments
    args = parse_args()
    data_dir, model_path = args[DATA_DIR], args[MODEL]
    fc = args[FC]
    rl = args[RL]
    category = args[CATEGORY]
    pretrained = args[PRETRAINED]
    nl = args[NL]

    if not os.path.isdir(data_dir) or not os.path.isfile(model_path):
        raise KeyError("Error: either data dir or model path are invalid. Use --data and --model options.")

    mvtec = "mvtec" in data_dir

    # Loading model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NormalizingFlowModel(
        get_backbone_resnet(wide_resnet50_2,
                            3 if mvtec else 1,
                            1024,
                            fc,
                            pretrained,
                            rl),
        fc, nl)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if mvtec:
        mvtec_show(data_dir, model, category)
    else:
        cdp_show(data_dir, model)


if __name__ == '__main__':
    main()
