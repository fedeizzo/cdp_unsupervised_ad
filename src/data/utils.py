import os

from torch.utils.data import DataLoader

from data.cdp_dataset import get_split
from data.transforms import NormalizedTensorTransform


def load_cdp_data(data_dir,
                  tp,
                  vp,
                  bs,
                  train_pre_transform=NormalizedTensorTransform(),
                  train_post_transform=None,
                  val_pre_transform=NormalizedTensorTransform(),
                  val_post_transform=None,
                  test_pre_transform=NormalizedTensorTransform(),
                  test_post_transform=None,
                  return_diff=False,
                  return_stack=False,
                  load=True,
                  originals="55"
                  ):
    """Loads CDP data from the given directory, splitting according to the percentages and applying the transforms.
    Only loads the particular type of original selected. Returns the 3 data loaders and the number of fake codes."""
    t_dir = os.path.join(data_dir, 'templates')
    x_dirs = [os.path.join(data_dir, f'originals_{originals}')]
    f_dirs = [os.path.join(data_dir, 'fakes_55_55'), os.path.join(data_dir, 'fakes_55_76'),
              os.path.join(data_dir, 'fakes_76_55'), os.path.join(data_dir, 'fakes_76_76')]

    n_fakes = len(f_dirs)
    train_set, val_set, test_set = get_split(t_dir,
                                             x_dirs,
                                             f_dirs,
                                             train_percent=tp,
                                             val_percent=vp,
                                             train_pre_transform=train_pre_transform,
                                             train_post_transform=train_post_transform,
                                             val_pre_transform=val_pre_transform,
                                             val_post_transform=val_post_transform,
                                             test_pre_transform=test_pre_transform,
                                             test_post_transform=test_post_transform,
                                             return_diff=return_diff,
                                             return_stack=return_stack,
                                             load=load
                                             )
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True) if tp > 0 else None
    val_loader = DataLoader(val_set, batch_size=bs) if vp > 0 else None
    test_loader = DataLoader(test_set, batch_size=bs) if tp + vp < 1 else None

    return train_loader, val_loader, test_loader, n_fakes
