# coding: utf-8


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import random

class LaneDataSet(Dataset):
    def __init__(self, dataset):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])

        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))

        label_binary[mask] = 1

        label_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)

        return (img, label_binary, label_img)