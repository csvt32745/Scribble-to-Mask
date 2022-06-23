import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
import random
from scipy.ndimage import distance_transform_edt

from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed

class VM108ImageDataset(Dataset):
    FG_FOLDER = 'FG_done'
    BG_FOLDER = 'BG_done'
    def __init__(self, root, mode='train', is_3ch_srb=False):
        self.root = root
        assert mode in ['train', 'val']
        self.is_3ch_srb = is_3ch_srb
        with open(os.path.join(self.root, 'frame_corr.json'), 'r') as f:
            self.frame_corr = json.load(f)
        with open(os.path.join(self.root, f'{mode}_videos.txt'), 'r') as f:
            self.im_list = self.parse_frames(f)
        #self.samples = self.samples[:240]
        self.dataset_length = len(self.im_list)


        print('%d images found in %s' % (self.dataset_length, root))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        interp_mode = transforms.InterpolationMode.BILINEAR

        # TODO: Original is Affine -> Resize, dont know if the perf is decreased
        self.im_dual_transform = transforms.Compose([
            transforms.Resize(480, interp_mode),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=im_mean),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=im_mean),
            transforms.GaussianBlur((13, 13)),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.Resize(480, interp_mode),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=0),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=0),
            transforms.GaussianBlur((13, 13)),
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

    def parse_frames(self, frame_list):
        samples = []
        for v in frame_list:
            samples += [k for k in sorted(self.frame_corr.keys()) if os.path.dirname(k) == v.strip()]
        return samples
    
    def __getitem__(self, idx):
        name = self.im_list[idx]
        # dn = os.path.dirname(name)

        # img I/O
        bgp = path.join(self.root, self.BG_FOLDER, self.frame_corr[name])
        if not os.path.exists(bgp):
            bgp = os.path.splitext(bgp)[0]+'.png'
        bg = Image.open(bgp).convert('RGB')
        # bg = np.float32(cv2.imread(bgp, cv2.IMREAD_COLOR))
        fg_gt = cv2.imread(path.join(self.root, self.FG_FOLDER, name), cv2.IMREAD_UNCHANGED)
        fg = fg_gt[..., -2::-1] # [B, G, R, A] -> [R, G, B])
        gt = fg_gt[..., -1]
        # assert bg.shape[:2] == fg.shape[:2]

        sequence_seed = np.random.randint(2147483647)
        reseed(sequence_seed)
        fg = self.im_dual_transform(Image.fromarray(fg).convert('RGB'))
        fg = self.im_lone_transform(fg)
        # reseed(sequence_seed)
        bg = self.im_dual_transform(bg)
        bg = self.im_lone_transform(bg)
        reseed(sequence_seed)
        gt = self.gt_dual_transform(Image.fromarray(gt))
        gt_np = np.array(gt)

        #TODO
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

        fg = self.final_im_transform(fg)
        bg = self.final_im_transform(bg)
        gt = self.final_gt_transform(gt)
        full_img = fg*gt + bg*(1-gt)
        fg = -1
        bg = -1

        prev_pred = self.final_gt_transform(prev_pred)

        info = {}
        info['name'] = name

        # print("vm108: ", full_img.shape, fg.shape, bg.shape, gt.shape)
        data = {
            'rgb': full_img,
            'fg': fg,
            'bg': bg,
            'gt_mask': gt,
            'prev_mask': prev_pred,
            'srb': srb,
            # 'srb_dist': srb_dists,
            'info': info
        }
        return data


    def __len__(self):
        return self.dataset_length
