import sys
sys.path.insert(1,'../repositories_used/face-parsing.PyTorch/')

from logger import setup_logger
from model import BiSeNet

import torch

import os
# import os.path as osp
import numpy as np
# from PIL import Image
import torchvision.transforms as transforms
import cv2

part_dict= {0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 
    9: 'ear_r', 10: 'nose', 11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck', 15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}
    
part_dict_inverse = {'background': 0, 'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4, 'r_eye': 5, 'eye_g': 6, 'l_ear': 7, 'r_ear': 8,
'ear_r': 9, 'nose': 10, 'mouth': 11, 'u_lip': 12, 'l_lip': 13, 'neck': 14, 'neck_l': 15, 'cloth': 16, 'hair': 17, 'hat': 18}

part_color_dict = {'background': [255, 255, 255], 'skin': [255, 85, 0], 'l_brow': [255, 170, 0], 'r_brow': [255, 0, 85], 'l_eye': [255, 0, 170], 
'r_eye': [0, 255, 0], 'eye_g': [85, 255, 0], 'l_ear': [170, 255, 0], 'r_ear': [0, 255, 85], 'ear_r': [0, 255, 170], 'nose': [0, 0, 255], 
'mouth': [85, 0, 255], 'u_lip': [170, 0, 255], 'l_lip': [0, 85, 255], 'neck': [0, 170, 255], 'neck_l': [255, 255, 0], 'cloth': [255, 255, 85], 
'hair': [255, 255, 170], 'hat': [255, 0, 255]}


def get_face_mask(img, img_parsed, mask_id=['skin','nose']):
    
    # mask_img = np.ones(img.shape, dtype=img.dtype)*255
    mask_img = np.zeros(img.shape, dtype=img.dtype)
    for i in range(0,len(mask_id)):       
        mask_img[np.all(img_parsed == part_color_dict[mask_id[i]], axis=-1)] = img[np.all(img_parsed == part_color_dict[mask_id[i]], axis=-1)]

    return mask_img


# from model import BiSeNet
def load_parsing_model(ckpt_path, n_classes=19):
    
    net = BiSeNet(n_classes=n_classes)
    if torch.cuda.is_available():
        print('------------------------------------')
        print('Loading parsing model on cuda')
        print('------------------------------------')
        net.cuda()
        # save_pth = osp.join('res/cp', cp)
        net.load_state_dict(torch.load(ckpt_path))
    else:
        print('------------------------------------')
        print('Loading parsing model on cpu')
        print('------------------------------------')
        net.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    net.eval()
    return net
    
def vis_parsing_maps(im, parsing_anno, stride, return_overlay=True):
    # Colors for all 20 parts
    # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
    #               [255, 0, 85], [255, 0, 170],[0, 255, 0], 
    #               [85, 255, 0], [170, 255, 0],[0, 255, 85], 
    #               [0, 255, 170],[0, 0, 255], [85, 0, 255], 
    #               [170, 0, 255], [0, 85, 255], [0, 170, 255],
    #               [255, 255, 0], [255, 255, 85], [255, 255, 170],
    #               [255, 0, 255], [255, 85, 255], [255, 170, 255],
    #               [0, 255, 255], [85, 255, 255], [170, 255, 255]]



    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    # print('max parse classes:',num_of_class)
    
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_color_dict[part_dict[pi]]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    if return_overlay==True:
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    
        return vis_parsing_anno_color, vis_im
    else:
        return vis_parsing_anno_color

to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def parse_img(image, net, resize=None):
    with torch.no_grad():
        if resize!=None:
            image = cv2.resize(image,resize)
        
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        if torch.cuda.is_available():
            img = img.cuda()
            
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        vis_img = vis_parsing_maps(image, parsing, stride=1)
    
    return vis_img