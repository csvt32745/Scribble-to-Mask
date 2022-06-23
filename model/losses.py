import torch
import torch.nn.functional as F
import torch.nn as nn
from util.tensor_util import compute_tensor_iu
import kornia as K

def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)
    # return 'iou/iou', (values['pha_l1'])/(values['pha_grad'])
iou_hooks_to_be_used = [
    # lambda val: ('pha/l1', val['pha_l1']),
    # lambda val: ('pha/laplacian', val['pha_laplacian']),
    # lambda val: ('fgr/l1', val['fgr_l1']),
]

class SegLossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        # self.bce = nn.BCEWithLogitsLoss
        self.lapla_loss = LapLoss(max_levels=4).cuda()
        self.morph_kernel = torch.ones(9, 9).cuda()
        self.semantic_blur = nn.Sequential(
            # K.filters.GaussianBlur2d((3, 3), (1, 1)),
            Interpolate(1/16),
            # nn.AvgPool2d((16, 16)),
            # nn.AvgPool2d((2, 2)),
        )
        self.blur = K.filters.GaussianBlur2d((3, 3), (1.4, 1.4))
        self.spatial_grad = K.filters.SpatialGradient()

    def compute(self, data, it):
        # mask = data['mask']
        gt_mask = data['gt_mask']
        trimap = data['trimap']
        srb = data['srb']
        losses = {}

        # TODO: Test wo this
        if 'mask_semantic' in data:
            losses.update(self.modnet_loss(data['mask_semantic'], gt_mask, srb, trimap))

        losses['seg_bce_weight_srb'] = self._bce_weighted_by_srb(data['mask_seg'], gt_mask, srb[:, :2]*10+trimap[:, :2])
        losses['seg_total_loss'] = sum(losses.values())
        
        return losses
        
    def modnet_loss(self, mask_sem, gt_mask, srb, trimap):
        return {
            'mask_semantic': self._semantic_loss(mask_sem, gt_mask, srb, trimap),
            # 'mask_boundary': self._boundary_detail_loss(mask_bound, gt_mask, srb, trimap)
        }

    def _semantic_loss(self, mask, gt_mask, srb, trimap):
        # print(mask.shape, gt.shape)
        gt = self.semantic_blur(gt_mask)
        weight = self.semantic_blur(srb[:, :2]*10+trimap[:, :2])
        # return 0.5*(F.mse_loss(torch.sigmoid(mask), gt))
        return 0.5*(F.mse_loss(torch.sigmoid(mask), gt) + self._bce_weighted_by_srb(mask, gt, weight)) # FG & BG only

    def _bce_weighted_by_srb(self, logits, gt, srb):
        # if srb.size(1) == 1:
        #     weight = ((srb > 0.99) | (srb < 0.01)).float()
        #     # weight = self.blur(weight)
        # else:
        weight = torch.sum(srb, dim=1, keepdim=True)
            # weight = torch.sum(2-srb, dim=1, keepdim=True) # max ~= 1.4
        
        # return F.binary_cross_entropy_with_logits(logits, gt, 1+weight*lamb)
        return F.binary_cross_entropy_with_logits(logits, gt, self.blur(weight))

class Interpolate(nn.Module):
    def __init__(self, scale_ratio, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.scale_ratio = scale_ratio
    
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_ratio, mode=self.mode)

class MatLossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        # self.bce = nn.BCEWithLogitsLoss
        self.lapla_loss = LapLoss(max_levels=4).cuda()
        self.morph_kernel = torch.ones(9, 9).cuda()
        self.semantic_blur = nn.Sequential(
            # K.filters.GaussianBlur2d((3, 3), (1, 1)),
            Interpolate(1/16),
            # nn.AvgPool2d((16, 16)),
            # nn.AvgPool2d((2, 2)),
        )
        self.blur = K.filters.GaussianBlur2d((3, 3), (1.4, 1.4))
        self.spatial_grad = K.filters.SpatialGradient()

    def compute(self, data, it):
        mask = data['mask']
        gt_mask = data['gt_mask']
        trimap = data['trimap']
        losses = self.matting_loss(mask, gt_mask)#, data['fg'], data['bg'], data['rgb'])
        
        # TODO: Test wo this
        if 'mask_semantic' in data:
            losses.update(self.modnet_loss(data['mask_semantic'], data['mask_boundary'], gt_mask, data['srb'], trimap))

        # losses['bce_weight_srb'] = self._bce_weighted_by_srb(data['logits'], gt_mask, data['srb'])
        losses['total_loss'] = sum(losses.values())
        
        return losses
        
    def modnet_loss(self, mask_sem, mask_bound, gt_mask, srb, trimap):
        return {
            'mask_semantic': self._semantic_loss(mask_sem, gt_mask, srb, trimap),
            'mask_boundary': self._boundary_detail_loss(mask_bound, gt_mask, srb, trimap)
        }

    def _boundary_detail_loss(self, mask, gt_mask, srb, trimap):
        # bound_mask = K.morphology.dilation(((gt_mask < 0.99) & (gt_mask > 0.01)).float(), self.morph_kernel)
        bound_mask = trimap[:, 2:] + srb[:, 2:]
        # bound_mask = K.morphology.dilation(gt_mask, self.morph_kernel) - K.morphology.erosion(gt_mask, self.morph_kernel)
        return 5*L1_mask(mask, gt_mask, bound_mask)

    def _semantic_loss(self, mask, gt_mask, srb, trimap):
        # print(mask.shape, gt.shape)
        gt = self.semantic_blur(gt_mask)
        weight = self.semantic_blur(srb[:, :2]*10+trimap[:, :2])
        # return 0.5*(F.mse_loss(torch.sigmoid(mask), gt))
        return 0.5*(F.mse_loss(torch.sigmoid(mask), gt) + self._bce_weighted_by_srb(mask, gt, weight)) # FG & BG only

    def matting_loss(self, pred_pha, true_pha, fg=None, bg=None ,img=None):
        """
        Args:
            pred_fgr: Shape(B, T, 3, H, W)
            pred_pha: Shape(B, T, 1, H, W)
            true_fgr: Shape(B, T, 3, H, W)
            true_pha: Shape(B, T, 1, H, W)
        """
        loss = dict()
        # Alpha losses
        # loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
        loss['pha_l1_l2'] = L1L2_split_loss(pred_pha, true_pha)
        # loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
        # loss['pha_grad'] = F.l1_loss(K.filters.sobel(pred_pha), K.filters.sobel(true_pha))
        
        # TODO: TEST wo these =========================
        loss['pha_laplacian'] = self.lapla_loss(pred_pha, true_pha)*10
        # loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
        #                                    true_pha[:, 1:] - true_pha[:, :-1]) * 5

        pred_grad = self.spatial_grad(pred_pha)
        true_grad = self.spatial_grad(true_pha)
        loss['pha_grad'] = F.l1_loss(pred_grad, true_grad)*10
        # loss['pha_grad_punish'] = 0.001 * torch.abs(pred_grad).mean()
        # ================================================

        # Foreground losses
        # true_msk = true_pha.gt(0)
        # pred_fgr = pred_fgr * true_msk
        # true_fgr = true_fgr * true_msk
        # composited = fg*pred_pha + bg*(1-pred_pha)
        # loss['fgr_l1'] = F.l1_loss(composited, img)


        # loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
        #                                    true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
        # Total
        # loss['total'] = loss['pha_l1'] + loss['pha_coherence'] + loss['pha_laplacian'] \
        #               + loss['fgr_l1'] + loss['fgr_coherence']
        # loss['total_loss'] = loss['pha_l1_l2'] + loss['pha_laplacian'] + loss['fgr_l1'] + + loss['pha_grad']
        # loss['total_loss'] = loss['pha_l1'] + loss['pha_grad']
        return loss

    def _bce_weighted_by_srb(self, logits, gt, srb):
        # if srb.size(1) == 1:
        #     weight = ((srb > 0.99) | (srb < 0.01)).float()
        #     # weight = self.blur(weight)
        # else:
        weight = torch.sum(srb, dim=1, keepdim=True)
            # weight = torch.sum(2-srb, dim=1, keepdim=True) # max ~= 1.4
        
        # return F.binary_cross_entropy_with_logits(logits, gt, 1+weight*lamb)
        return F.binary_cross_entropy_with_logits(logits, gt, self.blur(weight))

# ----------------------------------------------------------------------------- Laplacian Loss

def L1L2_split_loss(x, y, epsilon=1.001e-5):
    mask = ((y > 1-epsilon) | (y < epsilon)).float() # FG & BG
    dif = x - y
    l1 = torch.abs(dif)
    l2 = torch.square(dif)
    b,c,h,w = mask.shape
    
    mask_sum = mask.sum()
    res = torch.sum(l2 * mask)/(mask_sum+1e-5) + 5*torch.sum(l1 * (1-mask))/(b*h*w-mask_sum+1e-5)
    return res
    

def L1_mask(x, y, mask=None, epsilon=1.001e-5, normalize=True):
    res = torch.abs(x - y)
    b,c,h,w = y.shape
    if mask is not None:
        res = res * mask
        if normalize:
            _safe = torch.sum((mask > epsilon).float()).clamp(epsilon, b*c*h*w+1)
            return torch.sum(res) / _safe
        else:
            return torch.sum(res)
    if normalize:
        return torch.mean(res)
    else:
        return torch.sum(res)

'''
Borrowed from https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8
It directly follows OpenCV's image pyramid implementation pyrDown() and pyrUp().
Reference: https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff
'''
class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                    [4., 16., 24., 16., 4.],
                    [6., 24., 36., 24., 6.],
                    [4., 16., 24., 16., 4.],
                    [1., 4., 6., 4., 1.]])
        kernel /= 256.
        self.register_buffer('KERNEL', kernel.float())
        # self.L1 = nn.L1Loss()

    def downsample(self, x):
        # rejecting even rows and columns
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        # Padding zeros interleaved in x (similar to unpooling where indices are always at top-left corner)
        # Original code only works when x.shape[2] == x.shape[3] because it uses the wrong indice order
        # after the first permute
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = cc.permute(0,1,3,2)
        return self.conv_gauss(x_up, 4*self.KERNEL.repeat(x.shape[1], 1, 1, 1))

    def conv_gauss(self, img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        out = F.conv2d(img, kernel, groups=img.shape[1])
        return out

    def laplacian_pyramid(self, img):
        current = img
        pyr = []
        for level in range(self.max_levels):
            filtered = self.conv_gauss(current, \
                self.KERNEL.repeat(img.shape[1], 1, 1, 1))
            down = self.downsample(filtered)
            up = self.upsample(down)
            diff = current-up
            pyr.append(diff)
            current = down
        return pyr

    
    def forward(self, img, tgt, mask=None, normalize=True):
        pyr_input  = self.laplacian_pyramid(img)
        pyr_target = self.laplacian_pyramid(tgt)
        loss = sum((2 ** level) * L1_mask(ab[0], ab[1], mask=mask, normalize=False) \
                    for level, ab in enumerate(zip(pyr_input, pyr_target)))
        if normalize:
            b,c,h,w = tgt.shape
            if mask is not None:
                _safe = torch.sum((mask > 1e-6).float()).clamp(1e-6, b*c*h*w+1)
            else:
                _safe = b*c*h*w
            return loss / _safe
        return loss

