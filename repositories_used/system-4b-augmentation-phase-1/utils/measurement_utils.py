from scipy.spatial import distance
import cv2
from utils.drawing_spec_utils import mp_drawing
from utils.face_attribute_ids import idx_dict,idx_dict_color
from utils.landmark_utils import add_landmarks_id
from utils.face_attribute_ids import eye_contours
from utils.feature_extraction_utils import get_cropped_part,make_segmentation_overlay
from utils.drawing_spec_utils import *
import os
from utils.makeup_regions import forehead_landmarks
import numpy as np

def convert_landmark_to_px(landmark,img_landmarks,image_cols,image_rows):
    landmark = img_landmarks.multi_face_landmarks[0].landmark[landmark]
    landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    return landmark_px

## for extended foregead landmarks
def idx_2_pix_extended(landmark, img_landmarks, img_parsed, image_cols, image_rows):
    
    if landmark in forehead_landmarks.keys():
        landmark = forehead_landmarks[landmark]
    
        landmark_px = convert_landmark_to_px(landmark,img_landmarks,image_cols,image_rows)
        y_idx_px = min(np.where(np.all(img_parsed[:,landmark_px[0],] == [255,85,0], axis=-1))[0])
        landmark_px = (landmark_px[0],y_idx_px)
    else:
        landmark_px = convert_landmark_to_px(landmark,img_landmarks,image_cols,image_rows)
        
    return landmark_px

def draw_measurement_lines(img, results,img_parsed=None,dont_draw=['face_height','face_height_mp']):
    image = img.copy()
    for key in idx_dict.keys():
        if key in dont_draw:
            pass
        else:
            image = draw_landmark_lines(image, results, idx_dict[key],
                                              line_color=idx_dict_color[key], img_parsed=img_parsed)
        
    return image

def calculate_landmark_measurements(image, results, img_parsed=None):
    dist_dict = {}
    for key in idx_dict.keys():
        dist = calculate_landmark_distance(image, results, idx_dict[key], img_parsed=img_parsed)
        dist_dict[key] = dist

    return dist_dict

def measure_dist(st_px, end_px):
    dist = distance.euclidean(st_px,end_px)
    return round(dist,2)

def draw_landmark_lines(image, img_landmarks, part_indexes, line_color=[255,0,0], thickness=2, img_parsed=None):
    img_h,img_w = image.shape[:2]
    
    for kp_start,kp_finish in part_indexes:
        kp_px = idx_2_pix_extended(kp_start, img_landmarks, img_parsed, img_w,img_h)
        kp_px_end = idx_2_pix_extended(kp_finish, img_landmarks, img_parsed, img_w,img_h)
        
        cv2.line(image, kp_px, kp_px_end, line_color, thickness)
        cv2.circle(image, kp_px, 3, line_color, 1)
        cv2.circle(image, kp_px_end, 3, line_color, 1)
        
    return image

def calculate_landmark_distance(image, img_landmarks, part_indexes,img_parsed=None):
    img_h,img_w = image.shape[:2]
    
    for kp_start,kp_finish in part_indexes:
        kp_px = idx_2_pix_extended(kp_start, img_landmarks, img_parsed, img_w,img_h)
        kp_px_end = idx_2_pix_extended(kp_finish, img_landmarks, img_parsed, img_w,img_h)

        dist = measure_dist(kp_px, kp_px_end)
            
    return dist

def find_and_save_measurements(img_path, img, img_landmarks, path_output_img, path_output_json, 
                            save_annotated_images_flag=True,save_features_flag=True,
                            save_mapping_flag=True,save_measurements_flag=True, 
                               img_parsed=None):
    
    img_name = img_path.split('/')[-1]
    sp_name,image_ext = os.path.splitext(img_name)
    
    landmarks_with_ids = add_landmarks_id(img_landmarks)

    if save_annotated_images_flag==True:
        img_height,img_width,_ = img.shape 
        new_height = int(round(img_height*annotated_image_rescale))
        new_width = int(round(img_width*annotated_image_rescale))


        cpy_img = img.copy()
        cpy_img = cv2.resize(cpy_img, (new_width,new_height))
        mp_drawing.draw_landmarks(
                    image=cpy_img,
                    landmark_list=img_landmarks.multi_face_landmarks[0],
                    connections=eye_contours,
                    landmark_drawing_spec=drawing_spec_landmarks,
                    connection_drawing_spec=drawing_spec)

        cpy_img = cv2.cvtColor(cpy_img, cv2.COLOR_RGB2BGR)
        img_path_annotation = os.path.join(path_output_img, sp_name+'_annotated.jpg')
        cv2.imwrite(img_path_annotation, cpy_img)
    else:
        img_path_annotation = 'NA'

    if save_features_flag==True:
        img_left_eye = get_cropped_part(img, img_landmarks, mp_face_mesh.FACEMESH_LEFT_EYE)
        img_right_eye = get_cropped_part(img, img_landmarks, mp_face_mesh.FACEMESH_RIGHT_EYE)
        img_lips = get_cropped_part(img, img_landmarks, mp_face_mesh.FACEMESH_LIPS)
        img_segment = make_segmentation_overlay(img, img_landmarks, overlay_intensity=0.3)

        img_left_eye = cv2.cvtColor(img_left_eye, cv2.COLOR_RGB2BGR)
        img_right_eye = cv2.cvtColor(img_right_eye, cv2.COLOR_RGB2BGR)
        img_lips = cv2.cvtColor(img_lips, cv2.COLOR_RGB2BGR)
        img_segment = cv2.cvtColor(img_segment, cv2.COLOR_RGB2BGR)

        img_path_lips = os.path.join(path_output_img, sp_name+'_lips.jpg')
        img_path_left_eye = os.path.join(path_output_img, sp_name+'_left_eye.jpg')
        img_path_right_eye = os.path.join(path_output_img, sp_name+'_right_eye.jpg')
        img_path_segment = os.path.join(path_output_img, sp_name+'_segment.jpg')
        
        cv2.imwrite(img_path_lips, img_lips)
        cv2.imwrite(img_path_left_eye, img_left_eye)
        cv2.imwrite(img_path_right_eye, img_right_eye)
        cv2.imwrite(img_path_segment, img_segment)
    else:
        img_path_lips = 'NA'
        img_path_left_eye = 'NA' 
        img_path_right_eye = 'NA' 
        img_path_segment = 'NA'

    if save_mapping_flag==True:
        img_lines = draw_measurement_lines(img, img_landmarks,img_parsed=img_parsed)
        img_lines = cv2.cvtColor(img_lines, cv2.COLOR_RGB2BGR)
        img_path_lines = os.path.join(path_output_img, sp_name+'_measurement_lines.jpg')
        cv2.imwrite(img_path_lines, img_lines)
    else:
        img_path_lines = 'NA' 

    if save_measurements_flag==True:
        measurement_dict = calculate_landmark_measurements(img, img_landmarks, img_parsed=img_parsed)
    else:
        measurement_dict = 'NA'

    paths_dict = {
        'image_path_annotation': img_path_annotation,
        'image_path_lips': img_path_lips,
        'image_path_left_eye': img_path_left_eye,
        'image_path_right_eye': img_path_right_eye,
        'image_path_lines':img_path_lines,
        'image_path_segment':img_path_segment,
    }

    json_data = {
        'annotation_paths':paths_dict,
        'landmark_distances':measurement_dict,
        'landmarks':landmarks_with_ids
    }

    return json_data