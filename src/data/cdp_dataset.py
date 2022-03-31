# Imports
import os
from copy import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from data.transforms import AllRandomTransforms, NormalizedTensorTransform, ToTensorAll

# Definitions
# Bad indices from dataset A (5x5 codes)
INDICES_BAD_A = [29, 39, 104, 106, 110, 120, 130, 140, 141, 150, 174, 208, 226, 235, 244, 270, 280, 290, 300]


def get_split(t_dir,
              x_dirs,
              f_dirs,
              train_percent=0.4,
              val_percent=0.1,
              train_pre_transform=NormalizedTensorTransform(),
              train_post_transform=AllRandomTransforms(),
              val_pre_transform=NormalizedTensorTransform(),
              val_post_transform=None,
              test_pre_transform=NormalizedTensorTransform(),
              test_post_transform=None,
              bad_indexes=None,
              return_diff=False,
              return_stack=False,
              multi_gpu=False,
              load=True
              ):
    """
    Loads the CDP Dataset, composed of triplets of templates, originals and fakes (t,x,f).
    Randomly splits the dataset according to the training data percentage provided.
    The training set (only) is, by default, augmented with angles rotations, gamma corrections and flips.
    Images which file name is a number in the bad indexes are not loaded.
    """

    # Obtaining all images indices and shuffling them
    all_indices = np.arange(1, len(os.listdir(t_dir)) + 1)

    if bad_indexes:
        all_indices = [idx for idx in all_indices if idx not in INDICES_BAD_A]

    np.random.shuffle(all_indices)
    n = len(all_indices)

    # Getting number of samples per split
    n_train = round(n * train_percent)
    n_val = round(n * val_percent)

    # Creating datasets
    train_set = CDPDataset(
        t_dir=t_dir,
        x_dirs=x_dirs,
        f_dirs=f_dirs,
        indices=all_indices[:n_train],
        pre_transform=train_pre_transform,
        post_transform=train_post_transform,
        return_diff=return_diff,
        return_stack=return_stack,
        multi_gpu=multi_gpu,
        load=load
    )

    val_set = CDPDataset(
        t_dir=t_dir,
        x_dirs=x_dirs,
        f_dirs=f_dirs,
        indices=all_indices[n_train:n_train + n_val],
        pre_transform=val_pre_transform,
        post_transform=val_post_transform,
        return_diff=return_diff,
        return_stack=return_stack,
        multi_gpu=multi_gpu,
        load=load
    )

    test_set = CDPDataset(
        t_dir=t_dir,
        x_dirs=x_dirs,
        f_dirs=f_dirs,
        indices=all_indices[n_train + n_val:],
        pre_transform=test_pre_transform,
        post_transform=test_post_transform,
        return_diff=return_diff,
        return_stack=return_stack,
        multi_gpu=multi_gpu,
        load=load
    )

    return train_set, val_set, test_set


class CDPDataset(Dataset):
    def __init__(self, t_dir, x_dirs, f_dirs, indices, pre_transform=NormalizedTensorTransform(), post_transform=None,
                 return_diff=False, return_stack=False, multi_gpu=False, load=True):
        """
            Copy Detection Pattern (CDP) dataset. Data is loaded in triplets of templates, originals and fakes (t, x, f).

            :param t_dir: Directory containing images of the digital templates
            :param x_dirs: Directories containing images of the originals
            :param f_dirs: Directories containing images of the counterfeits
            :param indices: List of img numbers to be taken (e.g. [1, 7, 8, 13, ...])
            :param pre_transform: Transform to be applied to all images when loaded
            :param post_transform: Transform to be applied just-in-time when an item is retrieved (stochastic)
            :param return_diff: Whether to return (t, x, f) or the differences w.r.t template t -> (t-t, x-t, f-t)
            :param return_stack: Whether to a stacked version with template t -> [[t,t], [t,x], [t,f]]
            :param multi_gpu: If true, some samples are going to be discarded such that DS is divisible by #GPUs
        """
        super(CDPDataset, self).__init__()

        # Local variables
        self.t_dir = t_dir
        self.x_dirs = x_dirs
        self.f_dirs = f_dirs
        self.indices = indices
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.return_diff = return_diff
        self.return_stack = return_stack
        self.multi_gpu = multi_gpu
        self.all_loaded = load

        # Keeping only file names that exists in all folders
        self.file_names = []
        all_dirs = [t_dir, *x_dirs, *f_dirs]
        for fn in os.listdir(t_dir):
            if np.all([os.path.isfile(os.path.join(d, fn)) for d in all_dirs]) and int(fn.split(".")[0]) in indices:
                self.file_names.append(fn)

        if load:
            self.all_images = []
            for idx in range(self.__len__()):
                images = self._idx_to_images(idx)
                self.all_images.append(images)

    def __getitem__(self, item):
        """Returns the template image, followed by the several originals and fakes images.
        The original and fakes images come, in order, from the x_dirs and f_dirs used to initialize this CDPDataset
         """
        if self.all_loaded:
            images = self.all_images[item]
        else:
            images = self._idx_to_images(item)

        if self.post_transform:
            images = self.post_transform(*images)

        if self.return_diff:
            images = [images[0] - img for img in images]
        elif self.return_stack:
            images = [torch.cat((images[0], img)) for img in images]

        f_idx = 1 + len(self.x_dirs)
        return {
            'template': images[0],
            'originals': images[1:f_idx],
            'fakes': images[f_idx:]
        }

    def __len__(self):
        if self.multi_gpu and torch.cuda.device_count() > 0:
            # Avoiding to have GPUs without samples in the batch (causes errors)
            return len(self.file_names) - (len(self.file_names) % torch.cuda.device_count())
        return len(self.file_names)

    def _idx_to_images(self, idx):
        """
        Given an index, returns the template, all originals and all fakes for the same CDP (as a list).
        Also, applies the deterministic pre_transform, if specified, to all the images.
        """
        file_name = self.file_names[idx]
        images = [cv2.imread(os.path.join(self.t_dir, file_name), -1).astype(np.float32)]
        for x_dir in self.x_dirs:
            images.append(cv2.imread(os.path.join(x_dir, file_name), -1).astype(np.float32))

        for f_dir in self.f_dirs:
            images.append(cv2.imread(os.path.join(f_dir, file_name), -1).astype(np.float32))

        if self.pre_transform:
            images = self.pre_transform(*images)

        return images


class CDPSourceLoader(DataLoader):
    """Loads a particular set of images (either templates or one kind of original / fake) from a CDP dataset"""

    def __init__(self, dataset, key, source, device, batch_size=4):
        self.dataset = dataset
        self.key = key
        self.source = source
        self.device = device
        self.batch_size = batch_size
        self.idx = 0

    def __next__(self):
        if len(self.dataset) <= self.idx:
            self.idx = 0
            raise StopIteration

        x = torch.stack([self.dataset[idx][self.key][self.source]
                         for idx in np.arange(self.idx, self.idx + self.batch_size)
                         if idx < len(self.dataset)])
        x = x.to(self.device)

        self.idx += self.batch_size
        return x

    def __iter__(self):
        return self


def main():
    """Study the dataset in terms of pearson correlation in the input space."""
    import matplotlib.pyplot as plt

    t_dir = './../../1x1/templates'
    x_dirs = [
        './../../1x1/originals_55',
        './../../1x1/originals_76'
    ]
    f_dirs = [
        './../../1x1/fakes_55_55',
        './../../1x1/fakes_55_76',
        './../../1x1/fakes_76_55',
        './../../1x1/fakes_76_76'
    ]

    ds = CDPDataset(
        t_dir,
        x_dirs,
        f_dirs,
        np.arange(0, 720),
        pre_transform=NormalizedTensorTransform(),
        load=False
    )

    def std(img):
        return torch.std(img)

    def norm_std(img):
        return torch.std(img) / torch.sum(img < (torch.mean(img)))

    def t_std(img):
        return torch.std(img) / torch.sum(img < 0.1)

    def percent_black(img):
        return torch.sum(img < 0.5) / (np.prod([d for d in img.shape]))

    metric = percent_black
    all_x55, all_x76 = [], []
    all_f55_55, all_f55_76, all_f76_55, all_f76_76 = [], [], [], []
    features = {
        'x55': all_x55, 'x76': all_x76,
        'f55_55': all_f55_55, 'f55_76': all_f55_76, 'f76_55': all_f76_55, 'f76_76': all_f76_76
    }
    for element in ds:
        x55, x76 = element['originals']
        f55_55, f55_76, f76_55, f76_76 = element['fakes']

        for name, img in zip(features.keys(), [x55, x76, f55_55, f55_76, f76_55, f76_76]):
            features[name].append(metric(img))

    fig = plt.figure()
    ax = plt.subplot(111)
    for name in features.keys():
        y = features[name]
        ax.plot(np.arange(len(y)), y, label=name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Sample")
    plt.ylabel(metric.__name__)
    plt.show()


if __name__ == '__main__':
    main()
