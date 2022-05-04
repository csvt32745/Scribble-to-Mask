import os
from os import path
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
import mediapy as media
from enum import IntEnum, auto
from scipy.ndimage import distance_transform_edt

# from model.network import deeplabv3plus_resnet50 as S2M
from MODNet.modnet import MODNet as S2M
from MODNet.modnet_orig import MODNet as S2M_old
from model.aggregate import aggregate_wbg_channel as aggregate
from dataset.range_transform import im_normalization
from util.tensor_util import pad_divide_by

class ScribbleMode(IntEnum):
    POS = auto()
    NEG = auto()
    TRAN = auto()

class InteractiveManager:
    def __init__(self, model, image, mask, srb_3ch, pad_size=32, display_ratio=1.):
        self.model = model
        self.srb_3ch = srb_3ch
        self.image = im_normalization(TF.to_tensor(image)).unsqueeze(0).cuda()
        self.mask = TF.to_tensor(mask).unsqueeze(0).cuda()
        self.pad_size = pad_size
        self.display_ratio = display_ratio
        h, w = self.image.shape[-2:]
        self.img_size = self.image.shape[-2:]
        self.norm = min(h, w)
        self.image, self.pad = pad_divide_by(self.image, self.pad_size)
        print(self.image.shape)
        self.mask, _ = pad_divide_by(self.mask, self.pad_size)
        self.last_mask = None

        # Positive and negative scribbles
        self.p_srb = np.zeros((h, w), dtype=np.uint8)
        self.n_srb = np.zeros((h, w), dtype=np.uint8)
        self.t_srb = np.zeros((h, w), dtype=np.uint8) if self.srb_3ch else None
        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.positive_mode = ScribbleMode.POS
        self.cur_srb = self.p_srb
        self.need_update = True
        self.mode_dict = {
            ScribbleMode.POS: self.p_srb, 
            ScribbleMode.NEG: self.n_srb,
            ScribbleMode.TRAN: self.t_srb
        }
        self.srb_default_size = 3
        self.srb_size = self.srb_default_size

    def mouse_down(self, ex, ey):
        ex = int(ex / self.display_ratio)
        ey = int(ey / self.display_ratio)
        if ex >= self.img_size[1] or ey >= self.img_size[0]:
            return
        self.last_ex = ex
        self.last_ey = ey
        self.pressed = True
        cv2.circle(self.cur_srb, (ex, ey), radius=self.srb_size, color=(1), thickness=-1)
        self.need_update = True

    def mouse_move(self, ex, ey):
        ex = int(ex / self.display_ratio)
        ey = int(ey / self.display_ratio)
        if ex >= self.img_size[1] or ey >= self.img_size[0]:
            return
        if not self.pressed:
            self.last_ex = ex
            self.last_ey = ey
            return
        cv2.line(self.cur_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=self.srb_size)
        self.last_ex = ex
        self.last_ey = ey
        self.need_update = True

    def mouse_up(self):
        self.pressed = False

    def run_s2m(self):
        # Convert scribbles to tensors
        if self.srb_3ch:
            # Rs = [torch.from_numpy(distance_transform_edt(1-srb)).float() for srb in [self.p_srb, self.n_srb, self.t_srb]]
            # Rs = torch.stack(Rs, 0).unsqueeze(0).cuda() / self.norm
            # print(Rs.max())
            Rs = [torch.from_numpy(srb).unsqueeze(0).unsqueeze(0).float().cuda() for srb in [self.p_srb, self.n_srb, self.t_srb]]
            Rs = torch.cat(Rs, 1)
        else:
            Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float().cuda()
            Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float().cuda()
            Rs = 0.5 + 0.5*Rsp - 0.5*Rsn
        Rs, _ = pad_divide_by(Rs, self.pad_size)

        # Use the network to do stuff
        # inputs = torch.cat([self.image, Rs], 1)
        inputs = torch.cat([self.image, self.mask, Rs], 1)
        # _, mask = aggregate((net(inputs)))
        if args.old:
            _, mask = aggregate(net(inputs)[3]) # TODO: sigmoid
        else:
            mask = net(inputs)[3]

        # We don't overwrite current mask until commit
        self.last_mask = mask
        np_mask = (mask.detach().cpu().numpy()[0,0] * 255).astype(np.uint8)

        if self.pad[2]+self.pad[3] > 0:
            np_mask = np_mask[self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            np_mask = np_mask[:,self.pad[0]:-self.pad[1]]

        return np_mask

    def commit(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        if self.srb_3ch:
            self.t_srb.fill(0)
        if self.last_mask is not None:
            self.mask = self.last_mask

    def clean_up(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        if self.srb_3ch:
            self.t_srb.fill(0)
        self.mask.zero_()
        self.last_mask = None

    def next_mode(self):
        self.positive_mode = list(ScribbleMode)[(self.positive_mode+1) % ScribbleMode.__len__()]

    def next_scribble_mode(self):
        self.next_mode()
        if self.positive_mode == ScribbleMode.TRAN and not self.srb_3ch:
            self.next_mode()
        self.cur_srb = self.mode_dict[self.positive_mode]
        print('Entering [ %s ] scribble mode.' % self.positive_mode.name)

def mouse_callback(event, x, y, *args):
    if event == cv2.EVENT_LBUTTONDOWN:
        manager.mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        manager.mouse_up()
    elif event == cv2.EVENT_MBUTTONDOWN:
        manager.next_scribble_mode()
    # Draw
    if event == cv2.EVENT_MOUSEMOVE:
        manager.mouse_move(x, y)

def get_color(mode):
    if mode == ScribbleMode.POS:
        return [64, 255, 64]
    elif mode == ScribbleMode.NEG:
        return [255, 64, 64]
    return [64, 64, 255]

def comp_image(image, mask, p_srb, n_srb, t_srb=None):
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:,:,2] = 1
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    comp = (image*0.5 + color_mask*mask*0.5).astype(np.uint8)
    comp[p_srb>0.5, :] = np.array([0, 255, 0], dtype=np.uint8)
    comp[n_srb>0.5, :] = np.array([255, 0, 0], dtype=np.uint8)
    if t_srb is not None:
        comp[t_srb>0.5, :] = np.array([0, 0, 255], dtype=np.uint8)

    return comp

parser = ArgumentParser()
parser.add_argument('--image', default='ust_cat.jpg')
parser.add_argument('--model', default='saves/s2m.pth')
parser.add_argument('--mask', default=None)
parser.add_argument('--prev_mask', default=False, action='store_true')
parser.add_argument('--srb_3ch', default=False, action='store_true')
parser.add_argument('--backbone', type=str, default='mobilenetv3_large_100')
parser.add_argument('--pad_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=1280)
parser.add_argument('--display_size', type=int, default=1280)
parser.add_argument('--old', default=False, action='store_true')
args = parser.parse_args()

if 'AIM-500' in (s := args.image.split('/')):
    s[-2] = "mask"
    s[-1] = s[-1][:-3] + "png"
    args.mask = os.path.join('/' if args.image[0]=='/' else '', *s)
    print("Autoload AIM-500 Ground Truth: ", args.mask)

# network stuff
# net = S2M(6 if args.srb_3ch else 5)
if args.old:
    net = S2M_old(7 if args.srb_3ch else 5)
else:
    net = S2M(7 if args.srb_3ch else 5, backbone_arch=args.backbone)
# net = S2M()
net.load_state_dict(torch.load(args.model))
net = net.cuda().eval()
torch.set_grad_enabled(False)

# Reading stuff
image = cv2.imread(args.image, cv2.IMREAD_COLOR)
h, w = image.shape[:2]
is_img_wide =  w > 2*h
    
ratio = args.img_size / max(h, w)
image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
# image = cv2.resize(image, (1024, 1024))
h, w = image.shape[:2]
display_ratio = args.display_size / max(h, w)

new_bg = np.array([0, 255, 0]).reshape(1, 1, 3)

if show_mask_aside := (args.mask is not None):
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    show_mask = cv2.resize(mask, dsize=None, fx=display_ratio, fy=display_ratio, interpolation=cv2.INTER_CUBIC)
    alpha = show_mask[..., None]/255.
    show_comp = (alpha*cv2.resize(image, dsize=(show_mask.shape[1], show_mask.shape[0]), interpolation=cv2.INTER_CUBIC) \
        + (1-alpha)*new_bg).astype(np.uint8)
    show_mask = np.tile(show_mask[..., None], 3)

manager = InteractiveManager(
    net, image, mask if args.prev_mask else np.zeros((h, w), dtype=np.uint8), 
    args.srb_3ch, args.pad_size, display_ratio=display_ratio)

# OpenCV setup
cv2.namedWindow('S2M demo')
cv2.setMouseCallback('S2M demo', mouse_callback)

print('Usage: python interactive.py --image <image> --model <model> [Optional: --mask initial_mask]')
print('This GUI is rudimentary; the network is naively designed.')
print('Mouse Left - Draw scribbles')
print('Mouse middle key - Switch positive/negative')
print('Key f - Commit changes, clear scribbles')
print('Key r - Clear everything')
print('Key d - Switch between overlay/mask view')
print('Key s - Save masks into a temporary output folder (./output/)')
print('Key q - Increase scribble size')
print('Key e - Decrease scribble size')
print('Key w - Default scribble size')

class DisplayMode(IntEnum):
    Comparance = auto()
    Mask = auto()
    FG = auto()

display_comp = DisplayMode.Comparance
while 1:
    if manager.need_update:
        np_mask = manager.run_s2m()[..., None]
        if display_comp == DisplayMode.Comparance:
            display = comp_image(image, np_mask, manager.p_srb, manager.n_srb, manager.t_srb)
        elif display_comp == DisplayMode.Mask:
            # display = np_mask
            display = np.tile(np_mask, 3)
        else:# display_comp == DisplayMode.FG:
            alpha = (np_mask/255.)
            display = (alpha*image + (1-alpha)*new_bg).astype(np.uint8)
            # display = np.stack([distance_transform_edt(1-srb) for srb in [manager.p_srb, manager.n_srb, manager.t_srb]], axis=-1)/manager.norm
            # display = distance_transform_edt(1-manager.n_srb)/manager.norm
            
        manager.need_update = False

    final_display = display.copy()
    cv2.circle(final_display, (manager.last_ex, manager.last_ey), radius=manager.srb_size, color=get_color(manager.positive_mode), thickness=-1)
    final_display = cv2.resize(final_display, None, fx = display_ratio, fy = display_ratio)
    if show_mask_aside:
        # print(final_display.shape, show_mask.shape, show_comp.shape)
        final_display = np.concatenate([
            final_display, 
            (show_mask if display_comp == DisplayMode.Mask else show_comp)
            ],
            axis=(0 if is_img_wide else 1))
    cv2.imshow('S2M demo', final_display)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('f'):
        manager.commit()
        manager.need_update = True
    elif k == ord('s'):
        print('saved')
        os.makedirs('output', exist_ok=True)
        cv2.imwrite('output/%s' % path.basename(args.mask), mask)
    elif k == ord('d'):
        display_comp = (display_comp+1) % DisplayMode.__len__()
        manager.need_update = True
    elif k == ord('r'):
        manager.clean_up()
        manager.need_update = True
    elif k == ord('q'):
        if manager.srb_size+5 <= 50:
            manager.srb_size += 5
        print(f'Scribble size = [ {manager.srb_size} ]')
    elif k == ord('e'):
        if manager.srb_size-5 >= manager.srb_default_size:
            manager.srb_size -= 5
        print(f'Scribble size = [ {manager.srb_size} ]')
    elif k == ord('w'):
        manager.srb_size = manager.srb_default_size
        print(f'Scribble size = [ {manager.srb_size} ]')
    elif k == 27:
        break

cv2.destroyAllWindows()
