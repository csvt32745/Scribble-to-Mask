from os import path
import datetime
import time

import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.model import S2MModel
from dataset.static_dataset import StaticTransformDataset
from dataset.vm108_dataset import VM108ImageDataset
from dataset.d646_dataset import D646ImageDataset
from dataset.coco_dataset import COCODataset
# from dataset.lvis import LVIS
# from dataset.lvis_dataset import LVISTransformDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters


"""
Initial setup
"""
torch.backends.cudnn.benchmark = True

# Init distributed environment
# distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

# print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()

# local_rank = torch.distributed.get_rank()
# world_size = torch.distributed.get_world_size()
# torch.cuda.set_device(local_rank)

# print('I am rank %d in the world of %d!' % (local_rank, world_size))

"""
Model related
"""
# if local_rank == 0:
#     # Logging
#     if para['id'].lower() != 'null':
#         long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
#     else:
#         long_id = None
#     logger = TensorboardLogger(para['id'], long_id)
#     logger.log_string('hyperpara', str(para))

#     # Construct rank 0 model
#     model = S2MModel(para, logger=logger, 
#                     save_path=path.join('saves', long_id, long_id) if long_id is not None else None, 
#                     local_rank=local_rank, world_size=world_size).train()
# else:
#     # Construct models of other ranks
#     model = S2MModel(para, local_rank=local_rank, world_size=world_size).train()
# Logging

if para['id'].lower() != 'null':
    long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
else:
    long_id = None

logger = TensorboardLogger(para['id'], long_id)
logger.log_string('hyperpara', str(para))

# Construct rank 0 model
model = S2MModel(para, logger=logger, 
                save_path=path.join('saves', long_id, long_id) if long_id is not None else None, 
).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
else:
    total_iter = 0
    
if para['load_deeplab'] is not None:
    model.load_deeplab(para['load_deeplab'])

if para['load_network'] is not None:
    model.load_network(para['load_network'])

"""
Dataloader related
"""
def worker_init_fn(worker_id): 
    # return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id)

def construct_loader(train_dataset):
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank, shuffle=True)
    # train_loader = DataLoader(train_dataset, para['batch_size'], sampler=train_sampler, num_workers=8,
    train_loader = DataLoader(train_dataset, para['batch_size'], num_workers=16, shuffle=True,
                            drop_last=True, pin_memory=True)
                            # worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    return None, train_loader
    # return train_sampler, train_loader

"""
Dataset related
"""
# # Construct dataset
# static_root = path.expanduser(para['static_root'])
# fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
# duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
# duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
# ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)
# big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
# hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)

# static_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset] + [big_dataset, hrsod_dataset]*5)

# # LVIS
# lvis_root = para['lvis_root']
# LVIS_set = LVIS(path.join(lvis_root, 'lvis_v1_train.json'))
# LVIS_dataset = LVISTransformDataset(path.join(lvis_root, 'train2017'), LVIS_set)

# Matte
# train_dataset = StaticTransformDataset(para['matte_root'])
d646_dataset = D646ImageDataset(is_3ch_srb=para['srb_3ch'])
# vm108_dataset = VM108ImageDataset(para['matte_root'], is_3ch_srb=para['srb_3ch'])
# train_dataset = ConcatDataset([vm108_dataset]+[d646_dataset])
train_dataset = d646_dataset

train_sampler, train_loader = construct_loader(train_dataset)
print('Total dataset size: ', len(train_dataset))
total_epoch = math.ceil(para['iterations']/len(train_loader))
if para['seg_epoch'] > 0:
    coco_dataset = COCODataset()
    _, seg_loader = construct_loader(coco_dataset)
    num_seg_iter = len(train_loader)//2
    print('Segmentation dataset size: ', len(coco_dataset))
print('Number of training epochs (the last epoch might not complete): ', total_epoch)

# Need this to select random bases in different workers
# np.random.seed(np.random.randint(2**30-1) + local_rank*100)

try:
    for e in range(total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))
        
        # Crucial for randomness! 
        # train_sampler.set_epoch(e)
        # Train loop
        model.train()
        if e < para['seg_epoch']:
            seg_iter = 0
            print('Segmentation Training')
            for data in seg_loader:
                model.do_pass(data, total_iter, segmentation_mode=True)
                total_iter += 1
                seg_iter += 1
                # if seg_iter >= num_seg_iter or total_iter >= para['iterations']:
                if total_iter >= para['iterations']:
                    break
        else:
            print('Matting Training')
            # start = time.time()
            for data in train_loader:
                # print("data loader: ", time.time()-start)
                model.do_pass(data, total_iter)
                total_iter += 1
                # start = time.time()
            if total_iter >= para['iterations']:
                # last training must be matte
                break

finally:
    if not para['debug'] and model.logger is not None and total_iter>1000:
        model.save(total_iter)
    # Clean up
    # distributed.destroy_process_group()
