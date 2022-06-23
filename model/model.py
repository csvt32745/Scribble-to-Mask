import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

# from model.network import deeplabv3plus_resnet50
from MODNet.modnet_new import MODNet
# from MODNet.modnet_fuse import MODNet
# from MODNet.modnet_split import MODNet
from MODNet.modnet_seg import MODNet as MODNet_seg
# from MODNet.modnet_split2 import MODNet
# from matteformer.generators import Generator_MatteFormer as MatteFormer
# from networks.fba import FBAMatting as FBA
# from matte_models.Index.net import IndexMatting
from matte_models.GCA.generators import GCA

from model.aggregate import aggregate_wbg_channel as aggregate
from model.losses import MatLossComputer, SegLossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs


class S2MModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.local_rank = local_rank
        self.trimap_srb = not self.para['srb_3ch']
        # self.S2M = nn.parallel.DistributedDataParallel(
        #     nn.SyncBatchNorm.convert_sync_batchnorm(
        #         # deeplabv3plus_resnet50(num_classes=1, output_stride=16, pretrained_backbone=False)
        #         MODNet(in_channels = (5 if self.trimap_srb else 7), backbone_arch=para['backbone'], backbone_pretrained=para['pretrained_backbone']), # 3+1+3 or 3+1+1
        #         # MODNet(in_channels = (5 if self.trimap_srb else 6)), # 3+1+3 or 3+1+1
        #     ).cuda(),
        #     device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)

        self.S2M = MODNet(
            in_channels = (5 if self.trimap_srb else 7), 
            backbone_arch=para['backbone'], backbone_pretrained=para['pretrained_backbone']
            ).cuda() # 3+1+3 or 3+1+1
        
        # self.S2M = IndexMatting(in_channels = (5 if self.trimap_srb else 7)).cuda()
        # self.S2M = GCA().cuda()

        # self.S2M = MODNet_seg(
        #     in_channels = (5 if self.trimap_srb else 7), 
        #     backbone_arch=para['backbone'], backbone_pretrained=para['pretrained_backbone']
        #     ).cuda() # 3+1+3 or 3+1+1
        

        # self.S2M = FBA(
        #     in_channels = (5 if self.trimap_srb else 7), 
        #     backbone=para['backbone']).cuda()

        # self.S2M = MatteFormer().cuda()

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.para.save(os.path.join(os.path.dirname(save_path), 'config.json'))

        if logger is not None:
            self.last_time = time.time()
        # self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.train_integrator = Integrator(self.logger, distributed=False)
        # self.train_integrator.add_hook(iou_hooks_to_be_used)
        self.mat_loss_computer = MatLossComputer(para)
        self.seg_loss_computer = SegLossComputer(para)
        self.train()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.S2M.parameters()), lr=para['lr'])
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        self.make_lr_schedular()
        # Logging info
        self.report_interval = 50
        self.save_im_interval = 800
        self.save_model_interval = 10000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
        
        self.lt = 0
        self.loss = 0.

    def make_lr_schedular(self):
        ''' for reseting schedular '''
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-7, verbose=True, cooldown=50)

    def time_stamp(self, text=""):
        return
        print(text,time.time()-self.lt)
        self.lt = time.time()

    def do_pass(self, data, it=0, segmentation_mode=False):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)
        self.time_stamp('start')

        for k, v in data.items():
            # if type(v) != list and type(v) != dict and type(v) != int:
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda(non_blocking=True)

        out = {}
        self.time_stamp('to cuda')

        # ===== MODNet
        if isinstance(self.S2M, MODNet):
            mask_semantic, mask_boundary, logits, mask = self.S2M(
                data['rgb'], torch.cat([data['prev_mask'], data['srb']], 1), 
                inference=False) # TODO: sigmoid
            out['mask_semantic'] = mask_semantic
            out['mask_boundary'] = mask_boundary
        # ===== MODNet with split seg
        elif isinstance(self.S2M, MODNet_seg):
            mask_semantic, mask_boundary, logits, mask, mask_seg = self.S2M(
                data['rgb'], torch.cat([data['prev_mask'], data['srb']], 1), 
                inference=False) # TODO: sigmoid
            out['mask_semantic'] = mask_semantic
            out['mask_seg'] = mask_seg
        # ====== 
        else:
            # mask = self.S2M(data['rgb'], torch.cat([data['prev_mask'], data['srb']], 1)) #TODO: Other model
            mask = self.S2M(torch.cat([data['rgb'], data['srb']], 1)) #TODO: GCA
        # mask = torch.cat(list(ret.values()), 0)
        # data['srb'] = Rs.repeat((3, 1, 1, 1))
        # data['gt_mask'] = data['gt_mask'].repeat((3, 1, 1, 1))
        
        # logits, mask = aggregate(prob)

        # out['logits'] = logits
        # out['mask'] = mask # TODO: no sigmoid
        out['mask'] = logits

        self.time_stamp('do model')
        
        if self._do_log or self._is_train:
            losses = (self.seg_loss_computer if segmentation_mode else self.mat_loss_computer)\
                .compute({**data, **out}, it)
            # losses = self.mat_loss_computer\
            #     .compute({**data, **out}, it)

            # Logging
            if self._do_log:
                self.integrator.add_dict(losses)
                if self._is_train:
                    if it % self.save_im_interval == 0 and it != 0:
                        if self.logger is not None:
                            if segmentation_mode and isinstance(self.S2M, MODNet_seg):
                                out['mask'] = torch.sigmoid(out['mask_seg'])
                            images = {**data, **out}
                            size = (384, 384)
                            self.logger.log_cv2('train/pairs', pool_pairs(images, size=size, trimap_srb=self.trimap_srb), it)

        if self._is_train:
            if (it) % self.report_interval == 0 and it != 0:
                # Plateau lr_scheduler
                self.scheduler.step(self.loss/self.report_interval)
                self.loss = 0.

                if self.logger is not None:
                    cur_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log_scalar('train/lr', cur_lr, it)
                    # self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                    self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                self.last_time = time.time()
                self.train_integrator.finalize('train', it)
                self.train_integrator.reset_except_hooks()

            if it % self.save_model_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save(it)

            self.time_stamp('log')

            # Backward pass
            self.optimizer.zero_grad() 
            loss = losses['total_loss']
            loss.backward() 
            self.loss += loss.detach().cpu().item()
            self.optimizer.step()
            # self.scheduler.step()

            self.time_stamp('step')
    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        # os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        # torch.save(self.S2M.module.state_dict(), model_path)
        torch.save(self.S2M.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = {
            'it': it,
            'network': self.S2M.state_dict(),
            # 'network': self.S2M.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        # self.S2M.module.load_state_dict(network)
        self.S2M.load_state_dict(network)
        # self.optimizer.load_state_dict(optimizer)
        # self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        map_location = 'cuda:%d' % self.local_rank
        self.S2M.load_state_dict(torch.load(path, map_location={'cuda:0': map_location}))
        # self.S2M.module.load_state_dict(torch.load(path, map_location={'cuda:0': map_location}))
        print('Network weight loaded:', path)

    def load_deeplab(self, path):
        map_location = 'cuda:%d' % self.local_rank

        cur_dict = self.S2M.state_dict()
        # cur_dict = self.S2M.module.state_dict()
        src_dict = torch.load(path, map_location={'cuda:0': map_location})['model_state']

        for k in list(src_dict.keys()):
            if type(src_dict[k]) is not int:
                if src_dict[k].shape != cur_dict[k].shape:
                    print('Reloading: ', k)
                    if 'bias' in k:
                        # Reseting the class prob bias
                        src_dict[k] = torch.zeros_like((src_dict[k][0:1]))
                    elif src_dict[k].shape[1] != 3:
                        # Reseting the class prob weight
                        src_dict[k] = torch.zeros_like((src_dict[k][0:1]))
                        nn.init.orthogonal_(src_dict[k])
                    else:
                        # Adding the mask and scribbles channel
                        pads = torch.zeros((64,3,7,7), device=src_dict[k].device)
                        nn.init.orthogonal_(pads)
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.S2M.load_state_dict(src_dict)
        # self.S2M.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.S2M.train()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.S2M.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.S2M.eval()
        return self

