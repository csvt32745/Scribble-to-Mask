import logging
import os
import json
import random
import sys
import time

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

import torchvision
from imgaug import parameters as iap
import numpy as np

from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

# from utils.utils import coords_grid, grid_sampler


class VideoMattingDataset(torch.utils.data.Dataset):
    VIDEO_SHAPE = (1080, 1920)
    FLOW_QUANTIZATION_SCALE = 100
    FG_FOLDER = 'FG_done'
    BG_FOLDER = 'BG_done'
    FLOW_FOLDER = 'flow_png'
    def __init__(self, data_root, image_shape, plus1, mode='train', \
                 use_subset=False, no_flow=True, precomputed_val=None, \
                 sample_length=5):
        self.no_flow = no_flow
        self.mode = mode
        self.precomputed_val = precomputed_val
        self.sample_length = sample_length
        self.is_single_img = sample_length == 1
        assert self.mode in ['train', 'val']
        if self.precomputed_val is not None:
            assert self.mode == 'val'
        self.data_root = data_root
        if plus1:
            self.image_shape = [image_shape[0]+1, image_shape[1]+1]
        else:
            self.image_shape = list(image_shape)
        setname = '{}_videos_subset.txt' if use_subset else '{}_videos.txt'
        setname = setname.format(self.mode)

        with open(os.path.join(self.data_root, 'frame_corr.json'), 'r') as f:
            self.frame_corr = json.load(f)
        with open(os.path.join(self.data_root, setname), 'r') as f:
            self.samples = self.parse(f)
        #self.samples = self.samples[:240]
        self.dataset_length = len(self.samples)

    def __len__(self):
        return self.dataset_length

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
                    # reflect padding if out of indices
                    sample[c-j-1] = fns[i-j-1] if i-j-1 >= 0 else fns[-(i-j-1)]
                    sample[c+j+1] = fns[i+j+1] if i+j+1 < len(fns) else fns[len(fns)-(i+j+1)-2]
                samples.append(sample)
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.mode == 'train' and random.random() > 0.5:
            sample = sample[::-1]
        length = len(sample)
        fg, bg, a = [None] * length, [None] * length, [None] * length
        dn = os.path.dirname(sample[0])

        data_root = self.data_root if self.precomputed_val is None else self.precomputed_val

        # img I/O
        for i in range(length):
            _f = cv2.imread(os.path.join(data_root, self.FG_FOLDER, sample[i]), cv2.IMREAD_UNCHANGED)
            bgp = os.path.join(data_root, self.BG_FOLDER, self.frame_corr[sample[i]])
            if not os.path.exists(bgp):
                bgp = os.path.splitext(bgp)[0]+'.png'
            bg[i] = np.float32(cv2.imread(bgp, cv2.IMREAD_COLOR))
            fg[i] = np.float32(_f[..., :-1])
            a[i] = np.float32(_f[..., -1:])
            assert bg[i].shape[:2] == fg[i].shape[:2]

        # optical flow I/O
        if not self.no_flow:
            wb, wf = [None] * length, [None] * length
            fns = []
            for i in range(length):
                fns.append(os.path.splitext(os.path.basename(sample[i]))[0])
            for i in range(2, length-2):
                wf[i] = _flow_read(fns[i], fns[i+1])
                wb[i] = _flow_read(fns[i], fns[i-1])
            wf[1] = _flow_read(fns[1], fns[2])
            wb[-2] = _flow_read(fns[-2], fns[-3])
        else:
            wf = None
            wb = None
            
        # augmentation
        # if self.mode == 'train':
        fg_aug = self.pixel_aug.to_deterministic()
        bg_aug = self.pixel_aug.to_deterministic()
        jpeg_aug = self.jpeg_aug.to_deterministic()
        fg, bg, a, wb, wf = self.shape_aug(fg, bg, a, wb, wf)
        for i in range(length):
            fg[i] = fg_aug.augment_image(np.uint8(fg[i].permute(1, 2, 0).numpy()))
            fg[i] = jpeg_aug.augment_image(fg[i])
            # fg[i] = torch.from_numpy(fg[i]).permute(2, 0, 1).float()
            # fg[i] = self.to_tensor(fg[i])
            bg[i] = bg_aug.augment_image(np.uint8(bg[i].permute(1, 2, 0).numpy()))
            # bg[i] = torch.from_numpy(bg[i]).permute(2, 0, 1).float()
            # bg[i] = self.to_tensor(bg[i])
            # a[i] = self.to_tensor(a[i])
        # else:
        #     if self.precomputed_val is not None:
        #         # img_padding_value = [103.53, 116.28, 123.675] # BGR
        #         img_padding_value = [0.406, 0.456, 0.485] # BGR
        #         for i in range(length):
        #             fg[i] = self.possible_pad(torch.from_numpy(fg[i]).permute(2, 0, 1), img_padding_value)
        #             bg[i] = self.possible_pad(torch.from_numpy(bg[i]).permute(2, 0, 1), img_padding_value)
        #             a[i] = self.possible_pad(torch.from_numpy(a[i]).permute(2, 0, 1))
        #         if not self.no_flow:
        #             for i in range(2, length-2):
        #                 wb[i] = self.possible_pad(wb[i].permute(2, 0, 1), np.nan)
        #                 wf[i] = self.possible_pad(wf[i].permute(2, 0, 1), np.nan)
        #             wb[-2] = self.possible_pad(wb[-2].permute(2, 0, 1), np.nan)
        #             wf[1] = self.possible_pad(wf[1].permute(2, 0, 1), np.nan)
        #     else:
        #         for i in range(length):
        #             fg[i] = self.img_crop_and_resize(fg[i], 0, 0).squeeze(0)
        #             bg[i] = self.img_crop_and_resize(bg[i], 0, 0).squeeze(0)
        #             a[i] = self.img_crop_and_resize(a[i], 0, 0).squeeze(0)
        #         if not self.no_flow:
        #             for i in range(2, length-2):
        #                 wb[i] = self.flow_crop_and_resize(wb[i], 0, 0).squeeze(0)
        #                 wf[i] = self.flow_crop_and_resize(wf[i], 0, 0).squeeze(0)
        #             wb[-2] = self.flow_crop_and_resize(wb[-2], 0, 0).squeeze(0)
        #             wf[1] = self.flow_crop_and_resize(wf[1], 0, 0).squeeze(0)
        #     if not self.no_flow:
        #         for i in range(length):
        #             if wb[i] is None:
        #                 wb[i] = torch.ones_like(wb[length // 2]) * torch.tensor(np.nan)
        #             if wf[i] is None:
        #                 wf[i] = torch.ones_like(wf[length // 2]) * torch.tensor(np.nan)

        fg = torch.stack(fg).flip(1).float()/255.
        bg = torch.stack(bg).flip(1).float()/255.
        a = torch.stack(a).float()/255.
        # if not self.no_flow:
        #     wb = torch.stack(wb).float()
        #     wf = torch.stack(wf).float()
        #     # Everything here is [S, N, H, W]
        #     return fg, bg, a, wb, wf#, torch.tensor(idx)
        
        # Generate previous noisy/blank mask
        gt_np = np.array(a)
        if np.random.rand() < 0.33:
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

        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)

        p_srb = torch.from_numpy(p_srb)
        n_srb = torch.from_numpy(n_srb)
        srb = torch.stack([p_srb, n_srb], 0).float()
        prev_pred = self.final_gt_transform(prev_pred)

        info = {}
        info['name'] = ''

        # Class label version of GT
        # cls_gt = (gt>0.5).long().squeeze(0)

        data = {
            'rgb': im,
            'gt_mask': gt,
            # 'cls_gt': cls_gt,
            'prev_pred': prev_pred,
            'srb': srb,
            'info': info
        }

        return data