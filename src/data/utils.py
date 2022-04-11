import os

from torch.utils.data import DataLoader

from anomalib.data.mvtec import MVTec
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from data.cdp_dataset import get_split


def load_cdp_data(data_dir, tp, bs, return_diff=False, return_stack=False, load=True):
    t_dir = os.path.join(data_dir, 'templates')
    # TODO: Multiple models for each original
    # x_dirs = [os.path.join(data_dir, 'originals_55'), os.path.join(data_dir, 'originals_76')]
    x_dirs = [os.path.join(data_dir, 'originals_55')]
    f_dirs = [os.path.join(data_dir, 'fakes_55_55'), os.path.join(data_dir, 'fakes_55_76'),
              os.path.join(data_dir, 'fakes_76_55'), os.path.join(data_dir, 'fakes_76_76')]

    n_orig, n_fakes = len(x_dirs), len(f_dirs)
    train_set, _, test_set = get_split(t_dir,
                                       x_dirs,
                                       f_dirs,
                                       train_percent=tp,
                                       val_percent=0,
                                       return_diff=return_diff,
                                       return_stack=return_stack,
                                       load=load
                                       )
    train_loader, test_loader = DataLoader(train_set, batch_size=bs, shuffle=True), DataLoader(test_set, batch_size=bs)

    return train_loader, test_loader, n_orig, n_fakes


def load_mvtec_data(data_dir, category, bs):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()])
    train_set = MVTec(data_dir, category, pre_process=transform, is_train=True)
    test_set = MVTec(data_dir, category, pre_process=transform, is_train=False)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    return train_loader, test_loader
