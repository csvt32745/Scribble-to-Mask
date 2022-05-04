import os
from os import path
import time

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

import imgaug.augmenters as iaa
from imgaug import parameters as iap

import cv2
import json
import random
from scipy.ndimage import distance_transform_edt

from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

class D646ImageDataset(Dataset):
    FG_FOLDER = 'FG'
    BG_FOLDER = 'BG'
    IMG_FOLDER = 'Image'
    GT_FOLDER = 'GT'
    def __init__(self, root='../dataset_mat/Distinctions646', mode='train', is_3ch_srb=False):
        assert mode in ['train', 'test']
        self.root = os.path.join(root, mode.capitalize())
        self.is_3ch_srb = is_3ch_srb
        self.im_list = os.listdir(os.path.join(self.root, self.IMG_FOLDER))
        self.dataset_length = len(self.im_list)


        print('%d images found in %s' % (self.dataset_length, root))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.05, 0.05, 0.2),
            transforms.RandomGrayscale(0.05),
        ])

        interp_mode = transforms.InterpolationMode.BILINEAR

        # TODO: Original is Affine -> Resize, dont know if the perf is decreased
        self.im_dual_transform = transforms.Compose([
            transforms.Resize(480, interp_mode),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=im_mean),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=im_mean),
            transforms.RandomApply([transforms.GaussianBlur((13, 13))]),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.Resize(480, interp_mode),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=0),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=0),
            transforms.RandomApply([transforms.GaussianBlur((13, 13))]),
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

        # self.pixel_aug = iaa.Sequential([
        #     iaa.MultiplyHueAndSaturation(mul=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)), # mean, std, low, high
        #     iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)),
        #     iaa.AddToHue(value=iap.TruncatedNormal(0.0, 0.1*100, -0.2*255, 0.2*255)),
        # ])
        self.jpeg_aug = iaa.Sometimes(0.6, iaa.JpegCompression(compression=(70, 99)))
    # TODO: FG BG processing is disabled
    def __getitem__(self, idx):
        name = self.im_list[idx]
        # start = time.time()
        # img I/O
        im = Image.open(path.join(self.root, self.IMG_FOLDER, name)).convert('RGB')
        # fg = Image.open(path.join(self.root, self.FG_FOLDER, name)).convert('RGB')
        # bg = Image.open(path.join(self.root, self.BG_FOLDER, name)).convert('RGB')
        gt = Image.open(path.join(self.root, self.GT_FOLDER, name)).convert('L')

        # fg_aug = self.pixel_aug.to_deterministic()
        jpeg_aug = self.jpeg_aug.to_deterministic()
        sequence_seed = np.random.randint(2147483647)
        # reseed(sequence_seed)
        # fg = self.im_dual_transform(fg)
        # fg = self.im_lone_transform(fg)
        # reseed(sequence_seed)
        # bg = self.im_dual_transform(bg)
        # bg = self.im_lone_transform(bg)
        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        im = self.im_lone_transform(im)
        im = jpeg_aug.augment_image(np.array(im, dtype=np.uint8))
        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)
        gt = jpeg_aug.augment_image(np.array(gt, dtype=np.uint8)[..., None])
        gt_np = gt.squeeze()
        # print(im.shape, gt.shape)
        # if np.random.rand() < 0.5:
        if True:
            # from_zero - no previous mask
            prev_pred = np.zeros_like(gt_np)
            from_zero = True
        else:
            iou_min = 20
            iou_max = 40
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            prev_pred = perturb_mask(gt_np, iou_target=iou_target)
            from_zero = False

        # Generate scribbles
        if self.is_3ch_srb:
            srbs = get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=True)
            # srb_dists = torch.stack([torch.from_numpy(distance_transform_edt(1-x)) for x in srbs], 0).float()/min(srbs[0].shape)
            srb = torch.stack([torch.from_numpy(x) for x in srbs], 0).float()
            # srb = torch.zeros((3, *(gt_np.shape)))
        else:
            srbs = get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=False)
            srb = torch.from_numpy(0.5 + 0.5*srbs[0] - 0.5*srbs[1]).float().unsqueeze(0)
            # srb = torch.zeros((1, *(gt_np.shape)))

        fg = -1
        bg = -1
        # fg = self.final_im_transform(fg)
        # bg = self.final_im_transform(bg)
        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)
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
            # 'srb_dist': srb_dists,
            'info': info
        }
        return data


    def __len__(self):
        return self.dataset_length
