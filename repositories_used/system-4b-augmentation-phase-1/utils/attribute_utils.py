import numpy as np
import cv2

from utils.change_lighting_utils import get_max_region, infer_light_change, make_shade_img
from utils.change_lighting_utils import change_lighting #, createSH

from utils.face_parsing_utils import part_color_dict

from utils.feature_extraction_utils import make_mask_delta
from utils.drawing_spec_utils import mp_face_mesh

from utils.mp_drawing_utils import _normalized_to_pixel_coordinates
FACE_BOX_LANDMARKS=[10,152,234,454] # head, chin, left_most, right_most


# -----------------------------------------------
# Eye Glasses
# -----------------------------------------------
LEFT_EYE_LANDMARKS =  [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
RIGHT_EYE_LANDMARKS = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]


# -----------------------------------------------
# For Occlusion and knockout
# -----------------------------------------------
FOREHEAD_AREA = [(104,67),(67,109),(109,10),(10,338),(338,297),(297,333),(333,293),(293,334),
                (334,105),(105,104)]

LEFT_EYEBROW_LANDMARKS_V2 = [(276,283),(283,282),(282,295),(295,285),(285,336),
                        (336,296),(296,334),(334,293),(293,300),(300,276)]


RIGHT_EYEBROW_LANDMARKS_V2 = [(46,53),(53,52),(52,65),(65,55),(55,107),
                        (107,66),(66,105),(105,63),(63,70),(70,46)]


# -----------------------------------------------
# Occlusion and knockout
# -----------------------------------------------

def get_part_coordinates_px_one(img, img_landmarks, part_indexes):

    img_h,img_w = img.shape[:2]
    coordinate_pairs = []
    for i in part_indexes:
        keypoint = img_landmarks.multi_face_landmarks[0].landmark[i]
        # end_x_y = img_landmarks.face_landmarks[0].landmark[kp_end]
        keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                    img_w, img_h)
        
        coordinate_pairs.append(keypoint_px)
        
    return np.array(coordinate_pairs)


def part_color_count(img_parsed,color):
    '''
    counts the number of pixels of a give face part
    '''
    indexes = np.where(np.all(img_parsed==color, axis=-1))
    # indexes are x,y tuple so len(x) = len(y) i.e len(indexes[0])=len(indexes[1])
    pixel_count = len(indexes[0])
    
    return pixel_count
    
def part_absent_flag(img_parsed, part_name, pixel_thresh=5):
    '''
    checks if a face part is absent in the image or not 
    based on the number of pixels counted for that part in the parsed image 
    '''
    pixel_count = part_color_count(img_parsed, part_color_dict[part_name])
    
    if pixel_count>pixel_thresh:
        absent_flag = False
    else:
        absent_flag = True
        
    
    return absent_flag

def part_present_flag(img_parsed, part_name, pixel_thresh=5):
    '''
    checks if a face part is present in the image or not 
    based on the number of pixels counted for that part in the parsed image 
    '''
    return not(part_absent_flag(img_parsed, part_name, pixel_thresh))

def check_face_parts_absence(img_parsed,parts_list,pixel_thresh=5):
    
    absence_flags = {}
    for part_name in parts_list:
        absence_flags[part_name] = part_absent_flag(img_parsed, part_name, pixel_thresh)
        
    return absence_flags

def check_face_parts_presence(img_parsed,parts_list,pixel_thresh=5):
    
    presence_flags = {}
    for part_name in parts_list:
        presence_flags[part_name] = part_present_flag(img_parsed, part_name, pixel_thresh)
    
    return presence_flags
    

def check_bad_landmarks_parsing(img_parsed, img_landmarks, part_landmarks, allowed_parts,
                               fill_mask_flag=False):    
    
    img_h, img_w = img_parsed.shape[:2] 
    mask = make_mask_delta(img_parsed,img_landmarks,part_landmarks,None,
                       line_thickness=2,fill_mask_flag=fill_mask_flag)

    indices = np.where(mask!=0)
    colors_found = np.unique(img_parsed[indices],axis=0)
    
    bad_labels = []
    for color in colors_found:
        pix_label = list(part_color_dict.keys())[list(part_color_dict.values()).index(list(color))]
        
        if pix_label not in allowed_parts:
            bad_labels.append(pix_label)
    
    if len(bad_labels)>0:
        bad_landmark_flag = True
    else:
        bad_landmark_flag = False

    return bad_landmark_flag, bad_labels


def check_face_landmark_occlusion_parsing(img_parsed,img_landmarks):
    
    allowed_left_brow = ['background','l_brow','r_brow','skin']
    bad_l_brow_flag, bad_labels_l_brow = check_bad_landmarks_parsing(img_parsed, img_landmarks, 
                                                            LEFT_EYEBROW_LANDMARKS_V2, allowed_left_brow,
                                                            fill_mask_flag=True)
    
    allowed_right_brow = ['background','l_brow','r_brow','skin']
    bad_r_brow_flag, bad_labels_r_brow = check_bad_landmarks_parsing(img_parsed, img_landmarks, 
                                                            RIGHT_EYEBROW_LANDMARKS_V2, allowed_right_brow,
                                                            fill_mask_flag=True)
    
    allowed_forehead = ['background','l_brow','r_brow','skin']
    bad_forehead_flag, bad_labels_forehead = check_bad_landmarks_parsing(img_parsed, img_landmarks, 
                                                            FOREHEAD_AREA, allowed_forehead,
                                                            fill_mask_flag=True)
    
    allowed_left_eye = ['background','l_eye','r_eye','skin']
    bad_l_eye_flag, bad_labels_l_eye = check_bad_landmarks_parsing(img_parsed, img_landmarks, 
                                                            mp_face_mesh.FACEMESH_LEFT_EYE, allowed_left_eye,
                                                            fill_mask_flag=True)
    
    allowed_right_eye = ['background','l_eye','r_eye','skin']
    bad_r_eye_flag, bad_labels_r_eye = check_bad_landmarks_parsing(img_parsed, img_landmarks, 
                                                            mp_face_mesh.FACEMESH_RIGHT_EYE, allowed_right_eye,
                                                            fill_mask_flag=True)
    
    allowed_lips = ['background','u_lip','l_lip','skin','mouth'] 
    bad_lips_flag, bad_labels_lips = check_bad_landmarks_parsing(img_parsed, img_landmarks, 
                                                            mp_face_mesh.FACEMESH_LIPS, 
                                                            allowed_lips)
    
    
    
    
    # eyebrows
    if (('cloth' in bad_labels_l_brow) or ('cloth' in bad_labels_r_brow) or 
        ('hat' in bad_labels_l_brow) or ('hat' in bad_labels_r_brow)):
        cloth_brow_flag = True
    else:
        cloth_brow_flag = False
        
    # hair on brow
    if (('hair' in bad_labels_l_brow) or ('hair' in bad_labels_r_brow)):
        hair_brow_flag = True
    else:
        hair_brow_flag = False
            
    # hair on forehead
    if 'hair' in bad_labels_forehead:
        hair_forehead_flag = True
    else:
        hair_forehead_flag = False
        
    
    # cloth on lips
    if (('cloth' in bad_labels_lips) or ('hat' in bad_labels_lips)):
        cloth_face_flag = True    
    else:
        cloth_face_flag = False
        
    
    parts_list_absent = ['nose','u_lip','l_lip','l_eye','r_eye','l_brow','r_brow']
    part_absence_flags = check_face_parts_absence(img_parsed,parts_list_absent,pixel_thresh=5)
    
    parts_list_present = ['hat']
    part_presence_flags = check_face_parts_presence(img_parsed,parts_list_present,pixel_thresh=5)
    
    
    face_occlusion_flags = {#'glasses_flag':glasses_flag,
                            'cloth_brow_flag':cloth_brow_flag,
                            'hair_brow_flag':hair_brow_flag,
                            #'hair_face_flag':hair_face_flag,
                            'cloth_face_flag':cloth_face_flag,
                            #'hat_flag':hat_flag,
                            #'no_lips_flag':no_lips_flag,
                            #'head_cut_flag':head_cut_flag
        
                            'hair_forehead_flag':hair_forehead_flag
                           }
    
    
    absent_parts = []
    for key in part_absence_flags.keys():
        if part_absence_flags[key]==True:
            absent_parts.append(key)
    
    present_parts = []
    for key in part_presence_flags.keys():
        if part_presence_flags[key]==True:
            present_parts.append(key)
    

    face_occlusion_flags['absence_flags'] = absent_parts
    face_occlusion_flags['presence_flags'] = present_parts
    
    return face_occlusion_flags


# -----------------------------------------------
# For Glasses
# -----------------------------------------------

def check_bad_landmarks(anno_map, img_landmarks, part_landmarks, allowed_parts):    
    img_h, img_w = anno_map.shape[:2]
    
    bad_labels = []
    for i in part_landmarks:
        landmark_px = _normalized_to_pixel_coordinates(img_landmarks.multi_face_landmarks[0].landmark[i].x, 
                                                                img_landmarks.multi_face_landmarks[0].landmark[i].y,
                                                                img_w, img_h)
        anno_pix = anno_map[landmark_px[1],landmark_px[0]]
        
        pix_label = list(part_color_dict.keys())[list(part_color_dict.values()).index(list(anno_pix))]

        if pix_label not in allowed_parts:
            bad_labels.append(pix_label)
            
    return bad_labels


def check_glasses_parsing(anno_map, img_landmarks):

    allowed_eye_parts = ['background','l_eye','r_eye','skin']
    bad_landmarks_left_eye = check_bad_landmarks(anno_map, img_landmarks, 
                                         LEFT_EYE_LANDMARKS, allowed_eye_parts)

    bad_landmarks_right_eye = check_bad_landmarks(anno_map, img_landmarks, 
                                         RIGHT_EYE_LANDMARKS, allowed_eye_parts)
    
    if (('eye_g' in bad_landmarks_left_eye) or ('eye_g' in bad_landmarks_right_eye)):
        return True
    
    return False


# -----------------------------------------------
# light Direction old
# -----------------------------------------------

# def calculate_quadrant_mean(quadrant):
#     '''
#     take the mean of the values that are not 0 to estimate the shade 
#     because 0 is out of the circle
#     '''
#     q_mean = quadrant[np.where(quadrant!=0)].mean()
#     return int(q_mean)

# def light_direction_check_old(detected_shading_img, mean_thresh=70):#, under_exp_thresh=100, over_exp_thresh=230):
    
#     top_left_quadrant = detected_shading_img[:128, :128]
#     top_left_mean = calculate_quadrant_mean(top_left_quadrant)

#     bottom_left_quadrant = detected_shading_img[128:, :128]
#     bottom_left_mean = calculate_quadrant_mean(bottom_left_quadrant)

#     top_right_quadrant = detected_shading_img[:128, 128:]
#     top_right_mean = calculate_quadrant_mean(top_right_quadrant)

#     bottom_right_quadrant = detected_shading_img[128:, 128:]
#     bottom_right_mean = calculate_quadrant_mean(bottom_right_quadrant)
    
    
#     left_half_mean = int(np.mean((top_left_mean, bottom_left_mean)))
#     right_half_mean = int(np.mean((top_right_mean, bottom_right_mean)))

#     top_half_mean = int(np.mean((top_left_mean, top_right_mean)))
#     bottom_half_mean = int(np.mean((bottom_left_mean, bottom_right_mean)))


#     mean_dict = {}
#     mean_dict['top_left'] = top_left_mean
#     mean_dict['bottom_left'] = bottom_left_mean
#     mean_dict['top_right'] = top_right_mean
#     mean_dict['bottom_right'] = bottom_right_mean
    

#     mean_dict_half = {}
#     mean_dict_half['left_half'] = left_half_mean
#     mean_dict_half['right_half'] = right_half_mean
#     mean_dict_half['top_half'] = top_half_mean
#     mean_dict_half['bottom_half'] = bottom_half_mean
    
    

#     max_mean_half = np.max(list(mean_dict_half.values()))
#     max_half = list(mean_dict_half.keys())[list(mean_dict_half.values()).index(max_mean_half)]
    
#     for key_mean in list(mean_dict_half.keys()):
#         if (mean_dict_half[max_half] - mean_dict_half[key_mean]) > mean_thresh:
#             light_dir = max_half
#         else:
#             light_dir=False
        
#     if light_dir==False:
#         max_mean = np.max(list(mean_dict.values()))
#         max_quad = list(mean_dict.keys())[list(mean_dict.values()).index(max_mean)]

#         for key_mean in list(mean_dict.keys()):
#             if (mean_dict[max_quad]- mean_dict[key_mean]) > mean_thresh:
#                 light_dir = max_quad
#             else:
#                 light_dir = False


    

#     return light_dir,mean_dict,mean_dict_half#max_quad,mean_dict#(light_dir,under_exposed,over_exposed), (left_half_mean, right_half_mean)


# -----------------------------------------------
# Face ratio
# -----------------------------------------------

def compare_face_dims(img, results, dims=(0.35,0.5)):
    
    img_h, img_w = img.shape[:2]
    
    coordinate_pairs = get_face_box_coordinates(img, results)

    x_min = coordinate_pairs[:,0].min()
    x_max = coordinate_pairs[:,0].max()
    y_min = coordinate_pairs[:,1].min()
    y_max = coordinate_pairs[:,1].max()
    
    face_height = abs(y_max-y_min)
    face_width = abs(x_max-x_min)
    
    ratio_height = face_height/img_h
        
    if ratio_height>=dims[0] and ratio_height<=dims[1]:
        height_check = True
        height_min = False
        height_max = False    
    else:
        height_check = False
        
    if ratio_height<dims[0]:
        height_min = True
        height_max = False
    elif ratio_height>dims[1]:
        height_min = False
        height_max = True        
    
    return ratio_height#height_check,(height_min, height_max), ratio_height

def get_face_box_coordinates(img, results):
    img_h,img_w = img.shape[:2]
    coordis = []
    for i in FACE_BOX_LANDMARKS:
        keypoint = results.multi_face_landmarks[0].landmark[i]
        keypoint_px =  _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, img_w, img_h)

        coordis.append(keypoint_px)
        
    return np.array(coordis)


# -----------------------------------------------
# Face angle
# -----------------------------------------------    
def calculate_face_angle(img, img_landmarks):
    # https://www.youtube.com/watch?v=-toNMaS4SeQ
    
    idx_list = [33, 263, 1, 61, 291, 199]
    face_2d = []
    face_3d = []
    img_h, img_w = img.shape[:2]

    for idx in  idx_list:
        lm = img_landmarks.multi_face_landmarks[0].landmark[idx]
        if idx_list==1:
            nose_2d = (lm.x * img_w, lm.y * img_h)
            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

        x, y = int(lm.x * img_w), int(lm.y * img_h)

        # Get the 2D Coordinates
        face_2d.append([x, y])

        # # Get the 3D Coordinates
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                        [0, focal_length, img_w / 2],
                        [0, 0, 1]])

    # The Distance Matrix
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x_angle = angles[0] * 360
    y_angle = angles[1] * 360
    return (x_angle,y_angle)


# -----------------------------------------------
# Check for light direction
# -----------------------------------------------
def light_direction_check(img,light_network,num_pieces,num_directions,mean_thresh=200,over_exposed=200,under_exposed=100):
    lighting_json = {}

    # light direction detection
    sh_path = '../repositories_used/DPR/light_directions/rotate_light_2/rotate_light_25.txt'
    relit_img, detected_sh, input_sh  = change_lighting(light_network, img, 'set_path', sh_path=sh_path)
    detected_sh_img = make_shade_img(detected_sh)
    light_angle_flag, light_angles = get_max_region(detected_sh_img,num_pieces,num_directions,mean_thresh=mean_thresh)
    
    lighting_json['light_angle_flag'] = light_angle_flag
    lighting_json['light_angle'] = light_angles

    indices = np.where(detected_sh_img!=0)
    shade_mean = round(np.mean(detected_sh_img[indices]))
    
    lighting_json['mean_intensity'] = shade_mean

    if shade_mean>=over_exposed:
        lighting_json['overexposed'] = True
    else:
        lighting_json['overexposed'] = False

    if shade_mean<=under_exposed:
        lighting_json['underexposed'] = True
    else:
        lighting_json['underexposed'] = False


    return lighting_json


def knockout_checks(img,img_parsed,img_landmarks,x_y_angle_face,lighting_json,face_angle_flag,face_angle_flags_dict,occlusion_dict):
    knockout_dict = {}
    
    # check for no faces in picture
    if img_landmarks.multi_face_landmarks==None:
        knockout_dict['no_face'] = True
        
    # check for multiple faces in picture
    if len(img_landmarks.multi_face_landmarks)>1:
        knockout_dict['multiple_faces'] = True
    
    # check for glasses in picture
    glasses_flag = check_glasses_parsing(img_parsed,img_landmarks)
    if glasses_flag==True:
        knockout_dict['glasses'] = True
    
    x_angle_thresh = 10
    y_angle_thresh = 10
    # check for face angle
    if x_y_angle_face[0]>x_angle_thresh or x_y_angle_face[1]>y_angle_thresh:
        knockout_dict['face_angle'] = True
        knockout_dict['face_angle_x_y'] = x_y_angle_face
    
    if face_angle_flag==True:
        knockout_dict['face_angle_v2'] = True
        knockout_dict['face_angle_v2_direction'] = face_angle_flags_dict

    # check for light direction
    if lighting_json['light_angle_flag']==True:
        knockout_dict['light_angle_flag'] = lighting_json['light_angle_flag']
        knockout_dict['light_angle'] = lighting_json['light_angle']

    if lighting_json['overexposed']==True:
        knockout_dict['light_overexposed'] = True
        knockout_dict['mean_intensity'] = lighting_json['mean_intensity']
    
    if lighting_json['underexposed']==True:
        knockout_dict['light_underexposed'] = True
        knockout_dict['mean_intensity'] = lighting_json['mean_intensity']

    # check for face artifacts and occlusions
    bad_flags = []
    for key in occlusion_dict.keys():
        if occlusion_dict[key]==True and (key not in ['absence_flags','presence_flags']):
            bad_flags.append(key)

    if len(bad_flags)>0:
        knockout_dict['occlusions'] = bad_flags

    if len(occlusion_dict['absence_flags'])>0:
        knockout_dict['missing_face_parts'] = occlusion_dict['absence_flags']

    if len(occlusion_dict['presence_flags'])>0:
        knockout_dict['unwanted_parts'] = occlusion_dict['presence_flags']
    

    return knockout_dict