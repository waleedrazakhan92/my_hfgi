import sys
sys.path.append('../')
from repositories_used.DPR.utils.utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
from scipy import special

from repositories_used.DPR.model.defineHourglass_1024_gray_skip_matchFeature import *
from math import atan2, degrees

def load_light_model(ckpt_path_light):

    # load model
    my_network_512 = HourglassNet(16)
    my_network = HourglassNet_1024(my_network_512, 16)
    # if USE_GPU==True:
    if torch.cuda.is_available():
        print('------------------------------------')
        print('Loading DPR on cuda')
        print('------------------------------------')
        my_network.load_state_dict(torch.load(ckpt_path_light))
        my_network.cuda()
    else:
        print('------------------------------------')
        print('Loading DPR on cpu')
        print('------------------------------------')
        my_network.load_state_dict(torch.load(ckpt_path_light,map_location=torch.device('cpu')))
    
    my_network.train(False)
    return my_network
    
def infer_light_change(img, my_network, sh):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    #-----------------------------------------------------------------

    row, col, _ = img.shape
    img = cv2.resize(img, (1024, 1024))
    Lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    if torch.cuda.is_available():
        inputL = Variable(torch.from_numpy(inputL).cuda())
    else:
        inputL = Variable(torch.from_numpy(inputL))
    
    sh = sh[0:9]
    sh = sh * 0.7

    # rendering half-sphere
    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid
    

    #----------------------------------------------
    #  rendering images using the network
    #----------------------------------------------
    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    if torch.cuda.is_available():
        sh = Variable(torch.from_numpy(sh).cuda())
    else:
        sh = Variable(torch.from_numpy(sh))
    outputImg, _, outputSH, _  = my_network(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1,2,0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg*255.0).astype(np.uint8)
    Lab[:,:,0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
    resultLab = cv2.resize(resultLab, (col, row))
    
    return resultLab, outputSH.cpu().data.numpy(), shading



def make_shade_img(sh):
    '''
    this script converts the shading array to its corresponding shading image
    '''
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = sh[0:9]
    sh = sh * 0.7

    # rendering half-sphere
    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid
    return shading


def createSH(theta_deg, phi_deg):
    '''
    this script creates a shading array on the basis of theta and phi values
    the theta and phi values need to be in degrees
    '''
    theta = theta_deg*np.pi/180
    phi = phi_deg*np.pi/180
    
    a = np.real(special.sph_harm(0,0,theta,phi))
    b = np.real(special.sph_harm(-1,1,theta,phi))
    c = np.real(special.sph_harm(0,1,theta,phi))
    d = np.real(special.sph_harm(1,1,theta,phi))
    e = np.real(special.sph_harm(-2,2,theta,phi))
    f = np.real(special.sph_harm(-1,2,theta,phi))
    g = np.real(special.sph_harm(0,2,theta,phi))
    h = np.real(special.sph_harm(1,2,theta,phi))
    i = np.real(special.sph_harm(2,2,theta,phi))

    shade_array = np.array([a,b,c,d,e,f,g,h,i])
    return shade_array
    
    
light_dirs_dict = {'front':'light_directions/rotate_light_2/rotate_light_25.txt',
                  'left':'light_directions/rotate_light_2/rotate_light_60.txt',
                  'right':'light_directions/rotate_light_2/rotate_light_00.txt',
                  'up':'light_directions/rotate_light_1/rotate_light_18.txt',
                  'down':'light_directions/rotate_light_1/rotate_light_56.txt'}

def change_lighting(light_network, img, light_dir, sh_path=None, intensity=0.85, theta=0, phi=0):

    
    if light_dir=='custom':
        try:
            sh = createSH(theta,phi)
            sh = sh*intensity
        except:
            print('Please input valid input shade!')
            return

    elif light_dir in light_dirs_dict.keys():
        path_lighting = light_dirs_dict[light_dir]
        sh = np.loadtxt(path_lighting)*intensity
    elif light_dir=='set_path':
        assert sh_path!=None
        sh = np.loadtxt(sh_path)*intensity
    else:
        print('Please select valid lighting direction from', light_dirs_dict.keys())
        print('or set it to (set_path) and give the shade path in sh_path parameter')
        return
    

    result_img, detected_sh, in_shading = infer_light_change(img, light_network, sh)  
    # detected_lighting_img = make_shade_img(detected_sh)
    return result_img, detected_sh, sh#, in_shading


def get_max_region(shade_img,n_regions=10,n_max=4,mean_thresh=200):
    assert n_regions>n_max
    
    angles = np.linspace(0 * np.pi, 2 * np.pi, n_regions)
    r = 128
    
    xs = np.array(r * np.cos(angles)+128, dtype=np.int)
    ys = np.array(r * np.sin(angles)+128, dtype=np.int)

    all_angles = []
    all_means = []
    for i in range(0,len(xs)-1):
        mask_img = np.zeros((shade_img.shape),dtype=shade_img.dtype)
        points = np.array( [[128,128], [xs[i],ys[i]], [xs[i+1],ys[i+1]]] )    
        mask_poly = cv2.fillPoly(mask_img, pts=[points], color=(255, 255, 255))
        
        area_mean = np.mean(shade_img[mask_poly!=0])
        all_means.append(area_mean)

        
    max_areas = np.flip(np.argsort(all_means))[:n_max]
    for max_idx in max_areas:
        mask_img = np.zeros((shade_img.shape),dtype=shade_img.dtype)
        points = np.array([[128,128], [xs[max_idx],ys[max_idx]], [xs[max_idx+1],ys[max_idx+1]]])    
        mask_poly = cv2.fillPoly(mask_img, pts=[points], color=(255, 255, 255))
    
        cpy_shade_img = shade_img.copy() 
        cpy_shade_img[mask_poly!=0]=255
        
        # max_idx = np.argmax(all_means)
        pts_mid = (round(np.mean((xs[max_idx],xs[max_idx+1]))), 
               round(np.mean((ys[max_idx],ys[max_idx+1]))))
        
        all_angles.append(find_angle(pts_mid,(128,128)))

    ## checks if difference between mean line of max and min regions has a difference greater than threshold
    min_areas = np.argsort(all_means)[:n_max]
    if abs(all_means[max_areas[0]]-all_means[min_areas[0]])>mean_thresh:
        light_dir_flag = True
    else:
        light_dir_flag = False        
    
  
    return light_dir_flag, all_angles

def find_angle(p1, centre):
    height = 256
    width = 256
    x1, y1 = p1
    y1 = height - y1
    x2, y2 = centre
    y2 = height - y2
    x3, y3 = width, y2
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return round((deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)),2)