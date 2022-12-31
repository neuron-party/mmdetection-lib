import cv2
import numpy as np
import torch


def visualize_dataloader_sample(dataloader):
    '''
    takes a dataloader(output) of the build_detectection_train_loader and visualizes images and their labels
    make sure that shuffle=True
    '''
    x = next(iter(dataloader))
    img = x[0]['image'].permute(1, 2, 0).detach().cpu().numpy().copy()
    
    for i in x[0]['instances'].get_fields()['gt_boxes']:
        img = cv2.rectangle(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0))
        
    return img