import os
from os import path
import time

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
import random

from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

class D646ImageDataset(Dataset):
    FG_FOLDER = 'FG'
    BG_FOLDER = 'BG'
    IMG_FOLDER = 'Image'
    GT_FOLDER = 'GT'
    def __init__(self, root='../dataset_mat/Distinctions646', mode='train'):
        assert mode in ['train', 'test']
        self.root = os.path.join(root, mode.capitalize())
        
        self.im_list = os.listdir(os.path.join(self.root, self.IMG_FOLDER))
        self.dataset_length = len(self.im_list)


        print('%d images found in %s' % (self.dataset_length, root))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        interp_mode = transforms.InterpolationMode.BILINEAR

        self.im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=im_mean),
            transforms.Resize(480, interp_mode),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=im_mean),
            transforms.GaussianBlur((5, 5)),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=0),
            transforms.Resize(480, interp_mode),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=0),
            transforms.GaussianBlur((5, 5)),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        name = self.im_list[idx]
        # start = time.time()
        # img I/O
        im = Image.open(path.join(self.root, self.IMG_FOLDER, name)).convert('RGB')
        fg = Image.open(path.join(self.root, self.FG_FOLDER, name)).convert('RGB')
        bg = Image.open(path.join(self.root, self.BG_FOLDER, name)).convert('RGB')
        gt = Image.open(path.join(self.root, self.GT_FOLDER, name)).convert('L')

        sequence_seed = np.random.randint(2147483647)
        reseed(sequence_seed)
        fg = self.im_dual_transform(fg)
        fg = self.im_lone_transform(fg)
        reseed(sequence_seed)
        bg = self.im_dual_transform(bg)
        bg = self.im_lone_transform(bg)
        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        im = self.im_lone_transform(im)
        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)
        gt_np = np.array(gt)

        if np.random.rand() < 0.5:
        # if True:
            # from_zero - no previous mask
            prev_pred = np.zeros_like(gt_np)
            from_zero = True
        else:
            iou_max = 0.95
            iou_min = 0.4
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            prev_pred = perturb_mask(gt_np, iou_target=iou_target)
            from_zero = False

        # Generate scribbles
        p_srb, n_srb = get_scribble(prev_pred, gt_np, from_zero=from_zero)

        fg = self.final_im_transform(fg)
        im = self.final_im_transform(im)
        bg = self.final_im_transform(bg)
        gt = self.final_gt_transform(gt)

        # p_srb = torch.from_numpy(p_srb)
        # n_srb = torch.from_numpy(n_srb)
        # srb = torch.stack([p_srb, n_srb], 0).float()
        srb = torch.from_numpy(0.5 + 0.5*p_srb - 0.5*n_srb).float().unsqueeze(0)
        prev_pred = self.final_gt_transform(prev_pred)

        # ==== Debug
        # shape = im.shape[-2:]
        # srb = torch.zeros((2, *shape)).float()
        # prev_pred = torch.zeros_like(gt)
        # print("data loader: ", time.time()-start)
        # ====

        info = {}
        info['name'] = name
        # print("d646: ", im.shape, fg.shape, bg.shape, gt.shape)
        data = {
            'rgb': im,
            'fg': fg,
            'bg': bg,
            'gt_mask': gt,
            'prev_pred': prev_pred,
            'srb': srb,
            'info': info
        }
        return data


    def __len__(self):
        return self.dataset_length
