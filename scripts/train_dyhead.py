import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from easydict import EasyDict as edict

import detectron2
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import random

import mmdet
from mmdet.models.detectors.atss import ATSS

def get_img_metas(x):
    '''
    x: x = next(iter(dataloader))
    '''
    res = []
    for i in x:
        meta = {
            'pad_shape': i['instances']._image_size + tuple([3]),
            'img_shape': i['instances']._image_size + tuple([3])
        }
        # pad shape is the shape of the image after applying some padding to it
        res.append(meta)
    return res


def main():
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))


    neck=[
            dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_output',
                num_outs=5), 
            dict(type='SleepyDyHead', in_channels=256, out_channels=256, num_layers=6)
        ]

    bbox_head=dict(
            type='ATSSHead',
            num_classes=80,
            in_channels=256,
            stacked_convs=0,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))

    # need to use edict, normal dict doesnt work cuz mmdetection is dogshit
    train_cfg=edict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False)

    test_cfg=edict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)

    
    # model/data/optim init
    device = torch.device('cuda:0')
    model = ATSS(backbone, neck, bbox_head, train_cfg, test_cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    
    register_coco_instances(
        "my_coco2017_train", 
        {}, 
        # "json_annotation.json", 
        '/Users/sleepy/Documents/Github/mmdetection/datasets/coco/annotations/instances_train2017.json',
        # "path/to/image/dir"
        '/Users/sleepy/Documents/Github/mmdetection/datasets/coco/images/train2017/'
    )
    dataset = get_detection_dataset_dicts('my_coco2017_train')
    augs = [
        T.RandomBrightness(0.9, 1.1),
        T.RandomFlip(prob=0.5),
        T.Resize((640, 640))
    ]
    sampler = detectron2.data.samplers.TrainingSampler(shuffle=True, size=len(dataset))
    trainloader = build_detection_train_loader(total_batch_size=5, 
                                               mapper=detectron2.data.DatasetMapper(is_train=True, 
                                                                                    augmentations=augs,
                                                                                    image_format='RGB'),
                                               dataset=dataset, 
                                               sampler=sampler
                                              )
    # with the trainloader, its literally an infinite stream or training images that are randomly sampled
    # so we need to use iterations rather than epochs for training this way?
    # how do we do validation then?

    # if mapper == None, then the bounding boxes are in xywh
    # but what if mapper == DatasetMapper?
    
    cls_loss, bbox_loss, centerness_loss = [], [], []
    checkpoint = 1000

    for iteration, batch in enumerate(trainloader):
        # batch data into single tensors/lists and send to device
        img_batch = [i['image'] for i in batch]
        img_batch = torch.stack(img_batch, dim=0).to(device) / 255
        img_metas = get_img_metas(batch)
        gt_bboxes = [i['instances'].get_fields()['gt_boxes'].tensor.to(device) for i in batch]
        gt_labels = [i['instances'].get_fields()['gt_classes'].to(device) for i in batch]

        optimizer.zero_grad()
        loss = model(img_batch, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

        # how are the losses scaled bruh
        total_loss = sum(loss['loss_cls']) + sum(loss['loss_bbox']) + sum(loss['loss_centerness'])
        total_loss.backward()
        optimizer.step()

        # metrics
        cls_loss.append((sum(loss['loss_cls'])).detach().cpu().numpy())
        bbox_loss.append((sum(loss['loss_bbox'])).detach().cpu().numpy())
        centerness_loss.append((sum(loss['loss_centerness'])).detach().cpu().numpy())
        print(f'CLS Loss: {np.mean(cls_loss)}, BBOX Loss: {np.mean(bbox_loss)}, CENTERNESS Loss: {np.mean(centerness_loss)}')

        if iteration % checkpoint == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 'model_checkpoint_' + str(iteration) + '.pth')
            

if __name__ == '__main__':
    main()