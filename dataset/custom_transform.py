from tkinter import Variable
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np

import imgaug.augmenters as iaa
from imgaug import parameters as iap

from dataset.range_transform import *
from dataset.reseed import reseed
    

class RandomGaussianNoise:
    def __init__(self, mean=0, variance=(0, 32)):
        self.mean = mean
        self.variance = variance
    
    def __call__(self, img):
        img = np.array(img)
        var = np.random.uniform(*self.variance)
        h, w, c = img.shape
        noise = np.random.normal(loc=self.mean, scale=var, size=(h, w, 1))
        return Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8)).convert('RGB')

class CustomTransform:
    def __init__(self, shape=512):
        self.createTransforms(shape)

    def createTransforms(self, shape):
        
        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.05, 0.05, 0.1),
            transforms.RandomGrayscale(0.05),
        ])

        random_blur = lambda pil_img: pil_img.filter(ImageFilter.BoxBlur(np.random.randint(10, 40)))
        self.random_blur = transforms.RandomApply([random_blur], p=0.25)
        
        interp_mode = transforms.InterpolationMode.BILINEAR
        # TODO: Original is Affine -> Resize, dont know if the perf is decreased
        self.im_dual_transform = transforms.Compose([
            # transforms.Resize(shape, interp_mode),
            # transforms.RandomCrop((shape, shape), pad_if_needed=True, fill=im_mean),
            transforms.RandomResizedCrop(shape, scale=(0.3, 1.0)),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=im_mean),
            transforms.RandomApply([transforms.GaussianBlur((13, 13))], 0.25),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            # transforms.Resize(shape, interp_mode),
            # transforms.RandomCrop((shape, shape), pad_if_needed=True, fill=0),
            transforms.RandomResizedCrop(shape, scale=(0.3, 1.0)),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=0),
            transforms.RandomApply([transforms.GaussianBlur((13, 13))], 0.25),
            transforms.RandomHorizontalFlip(),
        ])

        self.jpeg_aug = iaa.Sometimes(0.5, iaa.JpegCompression(compression=(50, 90)))

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.RandomApply([RandomGaussianNoise(0, (0, 16))]),
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def applyMultiFGs(self, fgs, bg, gts):
        sequence_seed = np.random.randint(2147483647)
        jpeg_aug = self.jpeg_aug.to_deterministic()
        
        for i in range(len(fgs)):
            reseed(sequence_seed)
            fg = self.im_dual_transform(fgs[i])
            fg = self.im_lone_transform(fg)
            fgs[i] = np.array(fg)
            reseed(sequence_seed)
            gts[i] = np.array(self.gt_dual_transform(gts[i]))

        # reseed(sequence_seed)
        bg = self.im_dual_transform(bg)
        bg = self.im_lone_transform(bg)
        bg = self.random_blur(bg)

        # reseed(sequence_seed)
        # im = self.im_dual_transform(im)
        # im = self.im_lone_transform(im)

        fg = fg[0]
        gt = gt[0]

        # np.random.randint()


        gt = np.array(gt[0], dtype=np.uint8)[..., None]
        alpha = gt / 255.

        im = (np.array(fg)*alpha + np.array(bg)*(1-alpha)).astype(np.uint8)
        im = jpeg_aug.augment_image(np.array(im, dtype=np.uint8))
        gt = jpeg_aug.augment_image(gt)
        return im, gt

    def applyFBG(self, fg, bg, gt, is_return_pil=False):
        sequence_seed = np.random.randint(2147483647)
        jpeg_aug = self.jpeg_aug.to_deterministic()
        
        reseed(sequence_seed)
        fg = self.im_dual_transform(fg)
        fg = self.im_lone_transform(fg)

        # reseed(sequence_seed)
        bg = self.im_dual_transform(bg)
        bg = self.im_lone_transform(bg)
        bg = self.random_blur(bg)

        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)

        # reseed(sequence_seed)
        # im = self.im_dual_transform(im)
        # im = self.im_lone_transform(im)

        gt = np.array(gt, dtype=np.uint8)[..., None]
        alpha = gt / 255.
        im = (np.array(fg)*alpha + np.array(bg)*(1-alpha)).astype(np.uint8)
        im = jpeg_aug.augment_image(np.array(im, dtype=np.uint8))
        gt = jpeg_aug.augment_image(gt)
        
        if is_return_pil:
            im = Image.fromarray(im)
            gt = Image.fromarray(gt)
        return im, gt

    def apply(self, img, gt, is_return_pil=False):
        sequence_seed = np.random.randint(2147483647)
        jpeg_aug = self.jpeg_aug.to_deterministic()
        
        reseed(sequence_seed)
        img = self.im_dual_transform(img)
        img = self.im_lone_transform(img)

        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)

        im = jpeg_aug.augment_image(np.array(im, dtype=np.uint8))
        gt = jpeg_aug.augment_image(np.array(gt, dtype=np.uint8))
        
        if is_return_pil:
            im = Image.fromarray(im)
            gt = Image.fromarray(gt)
        return im, gt

    def apply_final_transform(self, imgs, gts):
        return \
            [self.final_im_transform(img) for img in imgs], \
            [self.final_gt_transform(gt) for gt in gts]

class CustomTransformSegmentation(CustomTransform):
    def createTransforms(self, shape):
        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.05, 0.05, 0.1),
            transforms.RandomGrayscale(0.05),
        ])
        
        interp_mode = transforms.InterpolationMode.BILINEAR
        interp_mode_gt = transforms.InterpolationMode.NEAREST

        # TODO: Original is Affine -> Resize, dont know if the perf is decreased
        self.im_dual_transform = transforms.Compose([
            transforms.Resize(shape, interp_mode),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode, fill=im_mean),
            transforms.RandomCrop((shape, shape), pad_if_needed=True, fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.Resize(shape, interp_mode_gt),
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, interpolation=interp_mode_gt, fill=0),
            transforms.RandomCrop((shape, shape), pad_if_needed=True, fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            RandomGaussianNoise(0, (0, 24)),
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def apply(self, img, gt, is_return_pil=False):
        sequence_seed = np.random.randint(2147483647)
        
        reseed(sequence_seed)
        img = self.im_dual_transform(img)
        img = self.im_lone_transform(img)

        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)

        if is_return_pil:
            im = im
            gt = gt
        return np.array(im, dtype=np.uint8), np.array(gt, dtype=np.uint8)


class CustomTestTransform(CustomTransform):
    def createTransforms(self, shape):

        interp_mode = transforms.InterpolationMode.BILINEAR
        interp_mode_gt = transforms.InterpolationMode.BILINEAR

        # TODO: Original is Affine -> Resize, dont know if the perf is decreased
        self.im_dual_transform = transforms.Compose([
            transforms.Resize(shape, interp_mode),
            transforms.RandomCrop((shape, shape), pad_if_needed=True, fill=im_mean),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.Resize(shape, interp_mode_gt),
            transforms.RandomCrop((shape, shape), pad_if_needed=True, fill=0),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def apply(self, im, gt, is_return_pil=False):
        sequence_seed = np.random.randint(2147483647)
        
        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)

        if is_return_pil:
            return im, gt
        return np.array(im, dtype=np.uint8), np.array(gt, dtype=np.uint8)
    
    def applyFBG(self, fg, bg, gt, is_return_pil=False):
        sequence_seed = np.random.randint(2147483647)
        
        alpha = np.array(gt, dtype=np.uint8)[..., None] / 255.
        im = (np.array(fg)*alpha + np.array(bg)*(1-alpha)).astype(np.uint8)
        
        reseed(sequence_seed)
        im = self.im_dual_transform(Image.fromarray(im))

        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)
        
        if is_return_pil:
            return im, gt

        return np.array(im, dtype=np.uint8), np.array(gt, dtype=np.uint8)

            
    def apply_final_transform(self, imgs, gts):
        return \
            [self.final_im_transform(img) for img in imgs], \
            [self.final_gt_transform(gt) for gt in gts]