import numpy as np
from skimage import io, color
import math

from utils.feature_extraction_utils import get_cropped_part_delta
from utils.ita_regions import *



import cv2
import blend_modes
import os

mask_id=['skin', 'nose','l_brow','r_brow']
selected_regions = ['chin','cheek_l','cheek_r','forehead_l','forehead_r']

def save_and_name_swatches(path_write,img_name,swatches):
    swatch_paths = {}
    for key in swatches.keys():
        sw_name = img_name+'_'+key+'.jpg'
        sw_path = os.path.join(path_write,sw_name)
        cv2.imwrite(sw_path,cv2.cvtColor(swatches[key],cv2.COLOR_RGB2BGR))
        
        swatch_paths[key] = sw_path
    
    return swatch_paths

def blend_and_return_swatches(face_mask,img_landmarks,b_mode,opacity=0.5):
    blend_mode = blend_modes.__dict__[b_mode]
    bl_img = blend_img(face_mask, face_mask, blend_mode=blend_mode,opacity=opacity)
    # bl_img = blend_img(bl_img, face_mask, blend_mode=blend_mode)


    itas_1, ita_mean_1,ita_labels_1,swatches_1 = find_cropped_itas(face_mask, img_landmarks, 
                                                      selected_regions,return_overlay=False)
    itas_2, ita_mean_2,ita_labels_2,swatches_2 = find_cropped_itas(bl_img, img_landmarks, 
                                                      selected_regions,return_overlay=False)
    itas_1['average_ita'] = ita_mean_1
    itas_2['average_ita'] = ita_mean_2
    return (itas_1,ita_labels_1,swatches_1),(itas_2,ita_labels_2,swatches_2),bl_img


def find_lab_one(img):
    # normalizes rgb values [0-1]
    rgb_val = find_rgb_one(img)
    L,A,B = color.rgb2lab(np.array(rgb_val)/255)
    return [round(L,2), round(A,2), round(B,2)]
        
        
def find_lab_dict(swatches):
    lab_dict = {}
    for key in swatches.keys():
        img = swatches[key]
        
        lab_dict[key] = find_lab_one(img)
    
    return lab_dict

def find_rgb_one(img):   
    rgb_mean = np.mean(img[np.where(np.all(img!=[0,0,0], axis=-1))],axis=0)
    R = rgb_mean[0]
    G = rgb_mean[1]
    B = rgb_mean[2]  
    return [round(R,2),round(G,2),round(B,2)]

def find_rgb_dict(swatches):
    lab_dict = {}
    for key in swatches.keys():
        img = swatches[key]
        
        lab_dict[key] = find_rgb_one(img)
    
    return lab_dict
    
def find_cropped_itas(face_mask, img_landmarks, selected_regions,return_overlay=False):
    
    if return_overlay==True:
        cropped_regions,ovrly = get_ita_regions(face_mask, img_landmarks,
                                                regions=selected_regions,return_overlay=return_overlay)
    else:
        cropped_regions = get_ita_regions(face_mask, img_landmarks,
                                                regions=selected_regions,return_overlay=return_overlay)
    
    itas, ita_mean, itas_labels = get_all_itas(cropped_regions, selected_regions)
    
    if return_overlay==True:
        return itas,ita_mean,itas_labels,cropped_regions,ovrly
    else:
        return itas,ita_mean,itas_labels,cropped_regions
    
def display_itas(img,selected_regions,itas,ita_mean,blend_mode):
    
    start_pos = (0,20)
    pos_step = 25
    img = cv2.putText(img, blend_mode , start_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, [255,255,255], 1, cv2.LINE_AA)
    
    pos = (start_pos[0],start_pos[1]+pos_step)
    for ita_num in range(0,len(selected_regions)):
        pos_txt = (pos[0], pos[1]+ita_num*pos_step)      
        text = selected_regions[ita_num]+': ' + str(itas[selected_regions[ita_num]])
        
        img = cv2.putText(img, text ,pos_txt, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, [255,255,255], 1, cv2.LINE_AA)
        
    
    ita_num = ita_num+1
    pos = (pos[0], pos[1]+ita_num*pos_step)
    text = 'Avg: '+str(ita_mean)
    
    img = cv2.putText(img, text ,pos, cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, [255,255,255], 1, cv2.LINE_AA)
    
    return img

def get_all_itas(cropped_regions, selected_regions):
    
    ita_vals = {}
    ita_labs = {}
    for i in range(0,len(selected_regions)):
        try:
            ita = round(ITA(cropped_regions[selected_regions[i]]))
            
        except:
            ita = 0
            
        ita_label = ITA_label(ita)    
        ita_vals[selected_regions[i]] = ita#ita_label

        ita_labs[selected_regions[i]] = ita_label

    mean_ita = round(sum(ita_vals.values())/len(ita_vals.values()))
    return ita_vals, mean_ita, ita_labs
    
def blend_img(bg_img, fg_img, opacity=0.5,blend_mode=blend_modes.addition):
    background_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)  # RGBA image
    background_img = np.array(background_img).astype(float)  # Inputs to blend_modes need to be numpy arrays.

    # Import foreground image
    foreground_img = cv2.cvtColor(fg_img, cv2.COLOR_RGB2RGBA)  # RGBA image
    foreground_img = np.array(foreground_img).astype(float)  # Inputs to blend_modes need to be numpy arrays.

    # blended_img = blend_modes.addition(background_img, foreground_img, opacity)
    blended_img = blend_mode(background_img, foreground_img, opacity)
    blended_img = np.array(blended_img)
    blended_img = np.uint8(blended_img)  
    
    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
    
    return blended_img

def get_ita_regions(img, img_landmarks, regions=['chin'],square=True,return_overlay=False):
    
    if return_overlay==True:
    	img_overlay = img.copy()

    cropped_regions = {}
    for i in range(0,len(regions)):
        face_part = get_cropped_part_delta(img, img_landmarks, 
                                           ita_regions_dict[regions[i]], ita_deltas[regions[i]])
        
        
        # if a face is sideways then some region might not appear
        try:         
            if square==True:
                indices = np.where(face_part!=0)
                x_min = indices[1].min()
                y_min = indices[0].min()
                x_max = indices[1].max()
                y_max = indices[0].max()

                face_part = img[y_min:y_max,x_min:x_max]
        except:
            pass

        cropped_regions[regions[i]] = face_part
        
        if return_overlay==True:
        	img_overlay[y_min:y_max,x_min:x_max]=255
    	
    if return_overlay==True:
    	return cropped_regions,img_overlay

    else:
    	return cropped_regions

def ycbcr(image):
    """
    Applies a YCbCr mask to an input image and 
    saves the resultant image

    Inputs:
        image - (str) RBG image file path

    Outputs:
        mask - (str) YCbCr image file path
    """

    # Read RGB image
    #RGB = io.imread(image)
    RGB = image

    # Subset dimensions
    R = RGB[:, :, 0]
    G = RGB[:, :, 1]
    B = RGB[:, :, 2]

    # Reduce to skin range
    R = np.where(R < 95, 0, R)
    G = np.where(G < 40, 0, G)
    B = np.where(B < 20, 0, B)

    R = np.where(R < G, 0, R)
    R = np.where(R < B, 0, R)
    R = np.where(abs(R - G) < 15, 0, R)

    R = np.where(G == 0, 0, R)
    R = np.where(B == 0, 0, R)

    B = np.where(R == 0, 0, B)
    B = np.where(G == 0, 0, B)

    G = np.where(R == 0, 0, G)
    G = np.where(B == 0, 0, G)

    # Stack into RGB
    RGB = np.stack([R, G, B], axis = 2)

    # Convert to YCBCR color-space
    YCBCR = color.rgb2ycbcr(RGB)

    # Subset dimensions
    Y = YCBCR[:, :, 0]
    Cb = YCBCR[:, :, 1]
    Cr = YCBCR[:, :, 2]

    # Subset to skin range
    Y = np.where(Y < 80, 0, Y)  
    Cb = np.where(Cb < 85, 0, Cb)
    Cr = np.where(Cr < 135, 0, Cr)

    Cr = np.where(Cr >= (1.5862*Cb) + 20, 0, Cr)
    Cr = np.where(Cr <= (0.3448*Cb) + 76.2069, 0, Cr)
    Cr = np.where(Cr <= (-4.5652*Cb) + 234.5652, 0, Cr)
    Cr = np.where(Cr >= (-1.15*Cb) + 301.75, 0, Cr)
    Cr = np.where(Cr >= (-2.2857*Cb) + 432.85, 0, Cr)

    Y = np.where(Cb == 0, 0, Y)
    Y = np.where(Cr == 0, 0, Y)

    Cb = np.where(Y == 0, 0, Cb)
    Cb = np.where(Cr == 0, 0, Cb)

    Cr = np.where(Y == 0, 0, Cr)
    Cr = np.where(Cb == 0, 0, Cr)

    # Stack into skin region
    skinRegion = np.stack([Y, Cb, Cr], axis = 2)
    skinRegion = np.where(skinRegion != 0, 255, 0)
    skinRegion = skinRegion.astype(dtype = "uint8")

    # Apply mask to original RGB image
    mask = np.array(RGB)
    mask = np.where(skinRegion != 0, mask, 0)

    #new_filepath = 'ycbcr/{}'.format(image)
    #Image.save(new_filepath)

    return mask 

def ITA(image):

    # Convert to CIE-LAB color space

    CIELAB = np.array(color.rgb2lab(image))
    
    # Get L and B (subset to +- 1 std from mean)
    L = CIELAB[:, :, 0]
    L = np.where(L != 0, L, np.nan)
    std, mean = np.nanstd(L), np.nanmean(L)
    L = np.where(L >= mean - std, L, np.nan)
    L = np.where(L <= mean + std, L, np.nan)

    B = CIELAB[:, :, 2]
    B = np.where(B != 0, B, np.nan)
    std, mean = np.nanstd(B), np.nanmean(B)
    B = np.where(B >= mean - std, B, np.nan)
    B = np.where(B <= mean + std, B, np.nan)

    # Calculate ITA
    ita_val = math.atan2(np.nanmean(L) - 50, np.nanmean(B)) * (180 / np.pi)

    return round(ita_val,2)

def ITA_label(ITA, method='kinyanjui'):
    """
    Maps an input ITA to a fitzpatrick label given
    a choice method

    Inputs:
        ITA - (float) individual typology angle
        method - (str) 'kinyanjui' or None

    OutputsL
        (int) fitzpatrick type 1-6
    """

    # Use thresholds from kinyanjui et. al.
    if method == 'kinyanjui':
        if ITA > 55:
            return 1
        elif ITA > 41:
            return 2
        elif ITA > 28:
            return 3
        elif ITA > 19:
            return 4
        elif ITA > 10:
            return 5
        elif ITA <= 10:
            return 6
        else:
            return None
    
    # Use empirical thresholds
    else:
        if ITA >= 45:
            return 1
        elif ITA > 28:
            return 2
        elif ITA > 17:
            return 3
        elif ITA > 5:
            return 4
        elif ITA > -20:
            return 5
        elif ITA <= -20:
            return 6
        else:
            return None
    

def run_blending_and_itas(img_path, img, img_landmarks, face_mask, args):

    img_name = img_path.split('/')[-1]
    sp_name,image_ext = os.path.splitext(img_name)

    all_itas_dict = {}
    all_labs_dict = {}
    all_rgbs_dict = {}
    all_ita_labels_dict = {}
    all_swatch_paths = {}

    for b_idx in range(0,len(args.blend_modes)):
        b_mode = args.blend_modes[b_idx]

        (itas_original, ita_labels_original, swatches_original),(itas_blended, ita_labels_blended, swatches_blended),bl_img = \
        blend_and_return_swatches(face_mask, img_landmarks, b_mode, args.opacity)

        #if save_swatches==True:
        if b_idx==0:
            swatches_original['face'] = face_mask
            swatch_paths_original = save_and_name_swatches(args.path_results,sp_name+'_input',swatches_original)
            all_swatch_paths['face'] = swatch_paths_original

        swatches_blended['mask'] = bl_img
        swatch_paths_blended = save_and_name_swatches(args.path_results,sp_name+'_mask_'+b_mode,swatches_blended)
        all_swatch_paths['swatch_'+b_mode] = swatch_paths_blended

        if b_idx==0:
            labs_original = find_lab_dict(swatches_original)
            labs_original['lab_full_mask'] =  find_lab_one(face_mask)
            
            rgbs_original = find_rgb_dict(swatches_original)
            rgbs_original['rgb_full_mask'] = find_rgb_one(face_mask)
    
            all_itas_dict['face'] = itas_original
            all_labs_dict['face'] = labs_original
            all_rgbs_dict['face'] = rgbs_original
            all_ita_labels_dict['face'] = ita_labels_original


        labs_blended = find_lab_dict(swatches_blended) 
        labs_blended['lab_full_mask'] =  find_lab_one(bl_img)
        
        rgbs_blended = find_rgb_dict(swatches_blended)
        rgbs_blended['rgb_full_mask'] = find_rgb_one(bl_img)

        all_itas_dict['swatch_'+b_mode] = itas_blended
        all_labs_dict['swatch_'+b_mode] = labs_blended
        all_rgbs_dict['swatch_'+b_mode] = rgbs_blended
        all_ita_labels_dict['swatch_'+b_mode] = ita_labels_blended

    json_data_ita = {
        'image_path' : img_path,
        'image_name': img_name,
        # 'light_angle':light_angles,
        # 'light_angle_flag':light_dir_flag,
        'light_source': 'NA',
        'light_type':'NA',
        'ITA_values':all_itas_dict,
        'ITA_labels':all_ita_labels_dict,
        'LAB_values':all_labs_dict,
        'RGB_values':all_rgbs_dict,
        'swatch_paths':all_swatch_paths
    }
    return json_data_ita
    