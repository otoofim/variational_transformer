from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, os.path
import torch
from PIL import *
from cityscapesscripts.helpers.labels import labels as city_labels
from torchvision.datasets import Cityscapes
from cityscapesscripts.helpers.labels import labels as city_labels


class CityscapesLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_path, transform_in = None, transform_ou = None, mode = 'train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.dataset = Cityscapes(dataset_path, split = mode, mode = 'fine', target_type = 'semantic')
        self.city_labels = city_labels


        self.transform_in = transform_in
        self.transform_ou = transform_ou

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img = np.array(self.dataset[idx][0])
        seg_mask = np.array(self.dataset[idx][1])

        seg_mask[seg_mask == -1] = 34
        label = np.zeros((seg_mask.shape[0], seg_mask.shape[1], self.get_num_classes()))
        indexs = np.ix_(np.arange(seg_mask.shape[0]), np.arange(seg_mask.shape[1]))
        label[indexs[0], indexs[1], seg_mask] = 1


        if self.transform_in:
            img = self.transform_in(img)
        if self.transform_ou:
            label = self.transform_ou(label)



        return {'image': img, 'label': label}


    def get_num_classes(self):
        return len(self.city_labels)
