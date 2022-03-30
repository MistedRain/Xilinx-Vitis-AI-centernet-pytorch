import os
import sys
import numpy as np
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from utils.dataloader import CenternetDataset, centernet_dataset_collate
from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50

division = '------------------'
def focal_loss(pred, target):
    pred = pred.permute(0,2,3,1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

if __name__ == '__main__':

    float_model = 'float_model'
    quant_mode = 'test'
    batchsize = 1
    quant_model = 'quantized_model'
    model_name = 'centernet.pth'

    

    #CPU or GPU
    if(torch.cuda.device_count()>0):
        print('CUDA available')
        device = torch.device('cuda:0')
    else:
        print('No CUDA available')
        device = torch.device('cpu')
    
    #load model
    num_classes = 20 #for voc
    model = CenterNet_Resnet50(num_classes, pretrain=False).to(device)
    model.load_state_dict(torch.load('float_model/centernet.pth',map_location='cpu'))


    #where to use?
    optimize = 1 
    

    rand_in = torch.randn([batchsize,3,512,512])#image size
    quantizer = torch_quantizer(quant_mode,model,(rand_in),output_dir=quant_model)
    quantized_model = quantizer.quant_model  


    #data loader
    input_shape = (512,512,3)
    annotation_path = '2007_test.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    test_dataset = CenternetDataset(lines[0:100], input_shape, num_classes, False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batchsize, 
                            shuffle=False,num_workers=8,pin_memory=True, 
                                drop_last=True, collate_fn=centernet_dataset_collate)
    print('ok1dataloader')
    '''
    for data,target in enumerate(test_loader):
        with torch.no_grad():
            target = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in target]
    batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
                    hm, wh, offset  = net(batch_images)
                    c_loss          = focal_loss(hm, batch_hms)
                    wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                    off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)

                    loss            = c_loss + wh_loss + off_loss

                    val_loss        += loss.item()
    '''

    #evaluate
    quantized_model.eval()
    total_c_loss = 0

    with torch.no_grad():
        for data, target in enumerate(test_loader):
            target = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in target]     
            batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = target
            hm, wh, offset = quantized_model(batch_images)
            c_loss = focal_loss(hm, batch_hms)
            total_c_loss += c_loss.item()
            print(data,total_c_loss)

    print('\nTest set: Accuracy: {}/{}\n'.format(total_c_loss, len(test_loader.dataset)))

    #export config
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)


    
