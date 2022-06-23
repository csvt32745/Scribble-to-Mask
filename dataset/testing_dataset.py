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


class StaticTransformDataset(Dataset):
    """
    Apply random transform on static images.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, root, gt_root):
        self.root = root

        self.im_list = os.listdir(self.root)

        print('%d images found in %s' % (len(self.im_list), root))

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
        im = Image.open(path.join(self.root, name)).convert('RGB')
        gt = Image.open(path.join(self.root, name.rsplit('.', maxsplit=1)[0]+'.png')).convert('L')
        gt_np = np.array(gt)

        # if np.random.rand() < 0.33:
        if True:
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
        trimap, srbs = get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=True)
        trimap = torch.from_numpy(np.stack(trimap, 0)).float()
        srb = torch.from_numpy(np.stack(srbs, 0)).float()

        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)
        prev_pred = self.final_gt_transform(prev_pred)

        info = {}
        info['name'] = name
        # print("d646: ", im.shape, fg.shape, bg.shape, gt.shape)
        data = {
            'rgb': im,
            'fg': -1,
            'bg': -1,
            'gt_mask': gt,
            'prev_pred': prev_pred,
            'srb': srb,
            'trimap': trimap,
            # 'srb_dist': srb_dists,
            'info': info
        }
        return data

    def __len__(self):
        return len(self.im_list)
