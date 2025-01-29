#%%
import os, sys
sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('./'))
import numpy as np
from glob import glob
import random
import torch
import torch.utils.data
import h5py
from data.augmentation import RandomCrop, Compose, one_hot
from torchvision.transforms import v2
# import skimage.io as skio


class RMSloader(torch.utils.data.Dataset):
    def __init__(self, img_path, is_train, src_key, trg_key):
        self.path = img_path
        self.is_train = is_train
        if self.is_train:
            self.augmentation = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(60, interpolation=v2.InterpolationMode.BICUBIC)
            ])
        self.src_key = src_key
        self.trg_key = trg_key
        
        with h5py.File(img_path, 'r') as hf:
            self.src_vol = hf[src_key]['img'][...]
            self.src_mask_vol = hf[self.src_key]['mask'][...]

            self.trg_vol = hf[trg_key]['img'][...]
            self.trg_mask_vol = hf[trg_key]['mask'][...]
        
        self.src_num = self.src_vol.shape[0]
        self.trg_num = self.trg_vol.shape[0]
        assert(self.src_num == self.trg_num)

        self.src_indices = np.arange(self.src_num)
        np.random.shuffle(self.src_indices)
        self.trg_indices = np.arange(self.trg_num)
        np.random.shuffle(self.trg_indices)

    def __len__(self):
        return self.src_num

    def __getitem__(self, item):
       
        src_idx = self.src_indices[item]
        trg_idx = self.trg_indices[item]
        src_img, src_seg = self.src_vol[src_idx, ...], self.src_mask_vol[src_idx, ...]
        trg_img, trg_seg = self.trg_vol[trg_idx, ...], self.trg_mask_vol[trg_idx, ...]

        # skio.imsave('C:/Users/azhylka/Projects/segmentation-renormalized-harmonization/src.png', self.src_key)
        if self.is_train:
            src_img, src_seg = self.augmentation([src_img, src_seg])
            trg_img, trg_seg = self.augmentation([trg_img, trg_seg])

        # skio.imsave('C:/Users/azhylka/Projects/segmentation-renormalized-harmonization/src_aug.png', self.src_key)

        dict = {}
        dict['A'] = np.expand_dims(src_img, axis=0)
        dict['B'] = np.expand_dims(trg_img, axis=0)

        ss = src_seg.astype(np.uint8)  # w, h
        # One hot encoding
        ss_oh = one_hot(ss, ncols=2)
        dict['A_seg'] = ss_oh

        ss = trg_seg.astype(np.uint8)  # w, h
        ss_oh = one_hot(ss, ncols=2)
        dict['B_seg'] = ss_oh
        return dict
