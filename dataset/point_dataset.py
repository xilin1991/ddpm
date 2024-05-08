import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn import datasets as sklearn_datasets
from loguru import logger
import matplotlib.pyplot as plt


class PointDataset(Dataset):
    def __init__(self, data_type:str, res_dirs:str, save_img:bool=True):
        super().__init__()
        if data_type == 's':
            curve, _ = sklearn_datasets.make_s_curve(10**4, noise=0.03)
            curve = curve[:, [0, 2]] / 10.0
        elif data_type == 'o':
            curve, _ = sklearn_datasets.make_circles(10**4, noise=0.01, factor=0.4)
            curve = curve / 10.0
        elif data_type == 'b':
            curve, *_ = sklearn_datasets.make_blobs(10**4, centers=2, cluster_std=[0.5, 0.5])
            curve = curve / 100.0
        elif data_type == 'm':
            curve, _ = sklearn_datasets.make_moons(10**4, noise=0.03)
            curve = curve / 10.0
        elif data_type == 'r':
            curve, _ = sklearn_datasets.make_swiss_roll(10**4, noise=0.1)
            curve = curve[:, [0, 2]] / 100.0
        else:
            raise NotImplementedError

        logger.info(f'shape of data: {np.shape(curve)}')

        if save_img:
            data = curve.T
            fig, ax = plt.subplots()
            ax.scatter(*data, color='red', edgecolor='white')
            ax.axis('off')
            plt.savefig(f'{res_dirs}/dataset.png', dpi=600, bbox_inches='tight')
        self.points = torch.from_numpy(curve).float()

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx, :]
