from functools import lru_cache
import os
from os import path
import time

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm

import imgaug.augmenters as iaa
from imgaug import parameters as iap

import cv2
import json
import random
from scipy.ndimage import distance_transform_edt

from dataset.custom_transform import CustomTransform, CustomTestTransform
from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble, get_region_gt, get_deform_regions
from dataset.reseed import reseed


class D646ImageDataset(Dataset):
    FG_FOLDER = 'FG'
    BG_FOLDER = 'BG'
    IMG_FOLDER = 'Image'
    GT_FOLDER = 'GT'
    def __init__(self, root='../dataset_sc/Distinctions646', mode='train', 
        is_3ch_srb=False, stage=3, shape=512, lru_cache_size=-1,
    ):
        assert mode in ['train', 'test']
        assert stage in range(1, 4)
        self.is_test = mode == 'test'
        self.root = os.path.join(root, mode.capitalize())
        self.is_3ch_srb = is_3ch_srb
        self.im_list = os.listdir(os.path.join(self.root, self.IMG_FOLDER))
        self.dataset_length = len(self.im_list)


        print('%d images found in %s' % (self.dataset_length, root))
        self.custom_transform = CustomTransform(shape=shape) if mode == 'train' else CustomTestTransform(shape=shape)

        self.get_sribbles = {
            1: self._get_trimap,
            2: self._get_deform_trimap,
            3: self._get_scribbles,
        }[stage]

        if mode == 'train' and lru_cache_size > 0:
            print("D646 lru cache size: %d" % lru_cache_size)
            self.read_fg_gt = lru_cache(lru_cache_size)(self._read_fg_gt)
        else:
            print("D646 lru cache is disabled")
            self.read_fg_gt = self._read_fg_gt
    
    def _get_trimap(self, prev_pred, gt_np, from_zero):
        return [get_region_gt(gt_np, self.is_3ch_srb)]*2

    def _get_deform_trimap(self, prev_pred, gt_np, from_zero):
        gts = get_region_gt(gt_np, self.is_3ch_srb, tran_size_min=10, tran_size_max=50)
        return gts, get_deform_regions(gts)

    def _get_scribbles(self, prev_pred, gt_np, from_zero):
        return get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=self.is_3ch_srb)

    def _read_fg_gt(self, name):
        fg = Image.open(path.join(self.root, self.FG_FOLDER, name)).convert('RGB').copy()
        gt = Image.open(path.join(self.root, self.GT_FOLDER, name)).convert('L').copy()
        return fg, gt

    # TODO: FG BG processing is disabled
    def __getitem__(self, idx):
        name = self.im_list[idx]
        # start = time.time()
        # img I/O
        if self.is_test:
            # fg, gt = self.read_fg_gt(name)
            im = Image.open(path.join(self.root, self.IMG_FOLDER, name)).convert('RGB')
            gt = Image.open(path.join(self.root, self.GT_FOLDER, name)).convert('L')
            im, gt = self.custom_transform.apply(im, gt)
        else:
            fg_name = name[:-4].rsplit('_', maxsplit=1)
            fg_name[1] = '0'
            fg, gt = self.read_fg_gt(fg_name[0]+'_'+fg_name[1]+name[-4:])
            bg = Image.open(path.join(self.root, self.BG_FOLDER, name)).convert('RGB')

            im, gt = self.custom_transform.applyFBG(fg, bg, gt)
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
        trimap, srbs = self.get_sribbles(prev_pred, gt_np, from_zero=from_zero)
        trimap = torch.from_numpy(np.stack(trimap, 0)).float()
        # srb = torch.stack([torch.from_numpy(x) for x in srbs], 0).float()
        srb = torch.from_numpy(np.stack(srbs, 0)).float()
        # if self.is_3ch_srb:
        #     srbs = get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=True)
        #     # srb_dists = torch.stack([torch.from_numpy(distance_transform_edt(1-x)) for x in srbs], 0).float()/min(srbs[0].shape)
        #     srb = torch.stack([torch.from_numpy(x) for x in srbs], 0).float()
        #     # srb = torch.zeros((3, *(gt_np.shape)))
        # else:
        #     srbs = get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=False)
        #     srb = torch.from_numpy(0.5 + 0.5*srbs[0] - 0.5*srbs[1]).float().unsqueeze(0)
        #     # srb = torch.zeros((1, *(gt_np.shape)))

        fg = -1
        bg = -1
        # fg = self.final_im_transform(fg)
        # bg = self.final_im_transform(bg)
        # im = self.final_im_transform(im)
        # gt = self.final_gt_transform(gt)
        # prev_pred = self.final_gt_transform(prev_pred)

        [im], [gt, prev_pred] = self.custom_transform.apply_final_transform(
            [im], [gt, prev_pred]
        )
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
            'prev_mask': prev_pred,
            'srb': srb,
            'trimap': trimap,
            # 'srb_dist': srb_dists,
            'info': info
        }
        return data


    def __len__(self):
        return self.dataset_length
