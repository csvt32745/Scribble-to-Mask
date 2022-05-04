import os
from os import path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO

from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed


class COCODataset(Dataset):
    """
    COCO Instance Segmetnation Dataset
    """
    def __init__(self,
        root_img='../dataset_mat/coco/train2017',
        root_anno='../dataset_mat/coco/annotations/instances_train2017.json',
        min_mask_ratio = 0.2
    ):
        self.min_mask_ratio = min_mask_ratio
        self.root  = root_img
        self.coco = COCO(root_anno)
        # self.coco = test
        self.img_keys = self.filter_data(list(self.coco.imgs.keys()))
        # self.im_list = [img.split('.')[0] for img in os.listdir(self.root)]
        
        # print('%d images found in %s' % (len(self.im_list), root))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        interp_mode = transforms.InterpolationMode.BILINEAR
        interp_mode_gt = transforms.InterpolationMode.NEAREST

        # TODO: Original is Affine -> Resize, dont know if the perf is decreased
        self.im_dual_transform = transforms.Compose([
            transforms.Resize(480, interp_mode),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=im_mean),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=im_mean),
            # transforms.RandomResizedCrop((480, 480))
            # transforms.GaussianBlur((5, 5)),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.Resize(480, interp_mode_gt),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode_gt, fill=0),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=0),
            # transforms.GaussianBlur((5, 5)),
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

    def filter_data(self, keys):
        def is_usable(id):
            # anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=id))
            anns = self.coco.imgToAnns[id]
            md = self.coco.imgs[id]
            img_area = md['height']*md['width']
            mask_ratio = np.array([ann['area'] for ann in anns])/img_area
            
            # filter the single pattern
            usable_mask = mask_ratio > 1e-3
            anns = self.coco.imgToAnns[id] = [ann for i, ann in enumerate(anns) if usable_mask[i]]
            mask_ratio = mask_ratio[usable_mask]

            if len(anns) > 1 and mask_ratio.sum() > self.min_mask_ratio:
                # store ratio info
                for i, ann in enumerate(anns):
                    ann['ratio'] = mask_ratio[i]
                return True
            return False
        return list(filter(is_usable, keys))

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_metadata = self.coco.imgs[img_id]
        
        name = img_metadata['file_name']
        im = Image.open(path.join(self.root, name)).convert('RGB')
        
        # polygon -> instance masks -> random select and merge the subset of instance masks
        anns = self.coco.imgToAnns[img_id]
        anns = np.random.permutation(anns)
        ann_masks = np.array([self.coco.annToMask(a) for a in anns], dtype=bool)
        # print(ann_masks)
        areas = np.cumsum([ann['ratio'] for ann in anns])

        # 
        num_instances = min(ann_masks.shape[0], 255) # preserve a value for non-instance
        rd = np.random.randint(1, max(2, num_instances//2))
        # Linear search since num_instances is small (mostly < 30)
        while rd < num_instances:
            if areas[rd] >= self.min_mask_ratio:
                break
            rd += 1
            
        # Merge instance into values [1, 2, ..., rd+1, ..., num_instances], 0 for non-instance
        # Then recover instance by ann_mask_merge-1 < rd (0 becomes 255 since uint8)
        ann_mask_merge = np.zeros(ann_masks.shape[1:], dtype=np.uint8)
        for i in range(num_instances):
            ann_mask_merge[ann_masks[i]] = i+1
        
        # Data augmentation
        sequence_seed = np.random.randint(2147483647)
        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        im = self.im_lone_transform(im)
        reseed(sequence_seed)
        ann_mask_merge = np.array(self.gt_dual_transform(Image.fromarray(ann_mask_merge)), dtype=np.uint8)
        
        # Reconver instance masks
        # print(num_instances, np.unique(ann_mask_merge.reshape(-1)).shape)
        ann_mask_merge -= 1
        # print(ann_mask_merge.dtype, ann_mask_merge.max())
        gt = (ann_mask_merge < rd).astype(np.uint8)*255
        # Instance-wise sample regions
        sample_regions = [
            [(ann_mask_merge==i).astype(np.uint8) for i in range(0, rd)],
            [(ann_mask_merge==i).astype(np.uint8) for i in range(rd, num_instances)] \
                + [(ann_mask_merge==255).astype(np.uint8)]
        ]

        # if from_zero:= (np.random.rand() < 0.5:):
        if from_zero:= True:
            # from_zero - no previous mask
            prev_pred = np.zeros_like(gt)
        else:
            iou_min = 20
            iou_max = 40
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            prev_pred = perturb_mask(gt, iou_target=iou_target)

        # Generate scribbles
        # if self.is_3ch_srb:
        srbs = get_scribble(
            prev_pred, gt, from_zero=from_zero, 
            is_transition_included=False,
            is_point_center_only=True,
            sample_regions=sample_regions,)
        # print([np.array(x).shape for x in sample_regions])
        # print([x.shape for x in srbs])
        # srb_dists = torch.stack([torch.from_numpy(distance_transform_edt(1-x)) for x in srbs], 0).float()/min(srbs[0].shape)
        srb = torch.stack([torch.from_numpy(x) for x in srbs] + [torch.zeros(gt.shape)], 0).float()
        # srb = np.concatenate([srbs, np.zeros_like(srbs[0])], axis=0)
        # srb = torch.zeros((3, *(gt_np.shape)))

        # else:
        #     srbs = get_scribble(prev_pred, gt_np, from_zero=from_zero, is_transition_included=False)
        #     srb = torch.from_numpy(0.5 + 0.5*srbs[0] - 0.5*srbs[1]).float().unsqueeze(0)
        #     # srb = torch.zeros((1, *(gt_np.shape)))

        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)
        prev_pred = self.final_gt_transform(prev_pred)

        info = {}
        info['name'] = name
        # print("coco: ", im.shape, fg.shape, bg.shape, gt.shape)
        data = {
            'rgb': im,
            'gt_mask': gt,
            'prev_pred': prev_pred,
            'srb': srb,
            # 'srb_dist': srb_dists,
            'info': info
        }
        return data


    def __len__(self):
        return len(self.img_keys)
