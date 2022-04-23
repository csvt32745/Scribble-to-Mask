from enum import Enum, auto
import abc
import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

class EnumDatasetSource(Enum):
    VM108 = auto()
    VM240 = auto()
    STATIC_IMG = auto()

class EnumDatasetType(Enum):
    VIDEO = auto()
    IMAGE = auto()

class InteractiveVideoMattingDataset(Dataset, abc.ABC):
    def __init__(
        self,
        root, 
        enum_src: EnumDatasetSource,
        enum_type: EnumDatasetType,
        train_or_valid: str = 'train',
        frame_shape = (480, 480),
        sample_length = 5,
    ):
        print(f"Dataset({train_or_valid}): [src = {enum_src.name}], [type = {enum_type.name}]")
        self.root = root
        self.sample_length = sample_length
        self.frame_shape = frame_shape

    def parse_path_vm108(self):
        def parse(self, f, length=None):
            if length is None:
                length = self.sample_length
            samples = []
            for v in f:
                v = v.strip()
                fns = [k for k in sorted(self.frame_corr.keys()) if os.path.dirname(k) == v]
                #fns = sorted(os.listdir(os.path.join(self.data_root, self.FG_FOLDER, v)))
                for i in range(len(fns)):
                    sample = [None] * length
                    c = length // 2
                    sample[c] = fns[i]
                    for j in range(length // 2):
                        sample[c-j-1] = fns[i-j-1] if i-j-1 >= 0 else fns[-(i-j-1)]
                        sample[c+j+1] = fns[i+j+1] if i+j+1 < len(fns) else fns[len(fns)-(i+j+1)-2]
                    samples.append(sample)
            return samples
        return

    def read_image_vm108(self, idx):
        return
    
    @abc.abstractmethod
    def read_data(self, idx):
        pass

    def preprocess_data(self, data)

    def __getitem__(self, idx):
        data = self.read_data(idx)
        aug = self.PREPROCESS_DATA(data)
        return self.PACK_DATA(aug)