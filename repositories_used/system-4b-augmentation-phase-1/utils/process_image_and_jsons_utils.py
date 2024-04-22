import sys
sys.path.append('..')

import cv2
# from glob import glob
import os
from utils.load_and_preprocess_utils import load_img
import numpy as np

# face recognition
# import dlib

# landmarks
from utils.landmark_utils import find_landmarks#, draw_landmarks
import PIL

## blend img script

# parsing and ita
from utils.face_parsing_utils import parse_img
from utils.face_parsing_utils import parse_img,part_color_dict
# from utils.ita_utils import *
from utils.ita_utils import run_blending_and_itas
from utils.face_parsing_utils import get_face_mask

# attributes
from utils.attribute_utils import calculate_face_angle,compare_face_dims,light_direction_check
from utils.checks_v2 import check_face_angles

# for landmark check on glasses
from utils.attribute_utils import check_glasses_parsing

# for getting mask color for bg change
from utils.feature_extraction_utils import get_one_shade

# bg utils
from utils.bg_removal_utils import remove_bg
from PIL import Image

# iris and hair color
from utils.feature_extraction_utils import get_cropped_part

# beauty score
from utils.beauty_score_utils import calculate_beauty_score


# from protobuf_to_dict import protobuf_to_dict
# from utils.measurement_utils import draw_measurement_lines,calculate_landmark_measurements

from utils.measurement_utils import find_and_save_measurements

from utils.misc_params import make_final_json

from utils.attribute_utils import knockout_checks

import mediapipe as mp

from utils.load_and_preprocess_utils import save_json

from utils.face_recognition_utils import find_matches   

from utils.face_shape_utils import detect_face_shape

import shutil

from utils.attribute_utils import check_face_landmark_occlusion_parsing

def process_everything(img_path, args,blending_args,face_matching_args,
                       parsing_model,light_network,modnet,beauty_model,
                       predictor,detector,face_rec_model):
    
    img = load_img(img_path)
    img_landmarks = find_landmarks(img)
    img_name = img_path.split('/')[-1]
    sp_name,image_ext = os.path.splitext(img_name)
    
    ## parsing and resizing by just copying the values and not the interpolation
    img_parsed = parse_img(img,parsing_model,(512,512))[0]
    # #img_parsed = cv2.resize(img_parsed,img.shape[::-1][1:])
    img_parsed = cv2.resize(img_parsed,img.shape[::-1][1:], None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    
    # face attributes 
    x_y_angle_face = calculate_face_angle(img,img_landmarks)
    # face angle v2
    face_angle_flag, face_angle_flags_dict = check_face_angles(img, img_landmarks,
                                                          down_thresh=0.2,up_thresh=0.85,
                                                            right_thresh=0.3,left_thresh=0.65)    
    ## face_height = compare_face_dims(img,img_landmarks)
    # face shape
    face_shape = detect_face_shape(img,img_landmarks,img_parsed)

    # more knockout checks
    occlusion_dict = check_face_landmark_occlusion_parsing(img_parsed, img_landmarks)

    # knockouts
    lighting_json = light_direction_check(img,light_network,args.num_pieces,args.num_directions,mean_thresh=args.mean_thresh)
    knockout_dict = knockout_checks(img,img_parsed,img_landmarks,x_y_angle_face,lighting_json,
                                face_angle_flag,face_angle_flags_dict,occlusion_dict)
    
    # check if image passes all the criteria
    if len(knockout_dict)==0:  
        # for ITA
        face_mask = get_face_mask(img, img_parsed)

        blending_json = run_blending_and_itas(img_path, img, img_landmarks, face_mask, blending_args)
        blending_json['light_angle_flag'] = lighting_json['light_angle_flag']
        blending_json['light_angle'] = lighting_json['light_angle']
        
        measurements_json = find_and_save_measurements(img_path,img,img_landmarks,
                                                       path_output_img=args.path_results,
                                                       path_output_json=args.path_results,
                                                      img_parsed=img_parsed)

        face_height = measurements_json['landmark_distances']['face_height']
        face_height = round(np.floor(face_height))

        measurements_json['face_shape'] = face_shape

        # bg removal
        mask_color = np.uint8(np.mean(face_mask[np.where(face_mask!=[0,0,0])[:2]], axis=0))

        if len(args.bg_color)==1:
            bg_color = get_one_shade(mask_color,n_shades=10,shade_strength=args.bg_color[0])
        else:
            bg_color = args.bg_color

        bg_img,img_rgba = remove_bg(Image.fromarray(img),modnet,bg_color)

        # rgb values for different parts
        irises = get_cropped_part(img,img_landmarks, list(mp.solutions.face_mesh_connections.FACEMESH_IRISES))
        iris_rgb = np.mean(irises[np.where(np.all(irises!=[0,0,0], axis=-1))],axis=0).astype(int)

        hair_mask = get_face_mask(img,img_parsed,['hair'])
        hair_rgb = np.mean(hair_mask[np.where(np.all(hair_mask!=[0,0,0], axis=-1))],axis=0).astype(int)

        # beauty score and face matching
        beauty_score = calculate_beauty_score(img,beauty_model)

        matched_imgs_paths = find_matches(PIL.Image.fromarray(img), 
                                        face_matching_args.all_encodings, face_matching_args.encoding_paths, 
                                        predictor, detector, face_rec_model, 
                                        tolerance=face_matching_args.threshold, max_imgs=face_matching_args.max_imgs,
                                        find_nearest=face_matching_args.find_nearest_imgs)

        ## so far limited to only one matching face
        matched_img = load_img(matched_imgs_paths[0])
        matching_beauty_score = calculate_beauty_score(matched_img,beauty_model)

        # remove and change image background
        bg_img_path = os.path.join(args.path_results,sp_name+'_bg.jpg')
        cv2.imwrite(bg_img_path, cv2.cvtColor(bg_img,cv2.COLOR_RGB2BGR))

        rgba_img_path = os.path.join(args.path_results,sp_name+'_transparent.png')
        cv2.imwrite(rgba_img_path, cv2.cvtColor(img_rgba,cv2.COLOR_RGB2BGRA))

        # create and save json
        final_json = make_final_json(blending_json,measurements_json,args)        

        final_json['features']['beauty_score'] = beauty_score
        final_json['features']['eye_color'] = iris_rgb.tolist()
        final_json['features']['hair_color'] = hair_rgb.tolist()
        final_json['features']['face_resemblence'] = matched_imgs_paths[0]
        final_json['features']['face_resemblence_beauty_score'] = matching_beauty_score
        # final_json['features']['face_path_bg_changed'] = bg_img_path
        # final_json['features']['face_path_bg_removed'] = rgba_img_path

        final_json['measurements']['face_distance'] = face_height

        save_json(sp_name, final_json, args.path_results)

    else:

        if lighting_json['light_angle_flag']==True or lighting_json['overexposed']==True or lighting_json['underexposed']==True:
            ## save the rejected json and image for lighting knockouts
            save_json(sp_name, knockout_dict, args.path_rejected_light)
            shutil.copy(img_path, args.path_rejected_light)
        else:
            ## save the rejected json and image for other knockouts
            save_json(sp_name, knockout_dict, args.path_rejected_other)
            shutil.copy(img_path, args.path_rejected_other)


