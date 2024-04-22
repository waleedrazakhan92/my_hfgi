from utils.mp_drawing_utils import _normalized_to_pixel_coordinates
import numpy as np

LEFT_EYE_CENTER = [473]
RIGHT_EYE_CENTER = [468]
NOSE_CENTER_LANDMARK = [1]
FACE_LEFT_RIGHT_LANDMARKS = [447, 227]

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

def eye_ear_nose_lines(img, results):
    img_h, img_w = img.shape[:2]
    
    left_right_eyes = get_part_coordinates_px_one(img,results,[LEFT_EYE_CENTER[0],RIGHT_EYE_CENTER[0]])    
    face_left_right = get_part_coordinates_px_one(img,results,FACE_LEFT_RIGHT_LANDMARKS)
    nose_px = get_part_coordinates_px_one(img,results,NOSE_CENTER_LANDMARK).squeeze()
    
    return left_right_eyes, face_left_right, nose_px

def means_updown(left_right_eyes, face_left_right, nose_px):
    '''
    for calculating up/down face angle
    '''
    
    eyes_x_mean = int(left_right_eyes[:,0].mean())
    eyes_y_mean = int(left_right_eyes[:,1].mean())
    
    face_x_mean = int(face_left_right[:,0].mean())
    face_y_mean = int(face_left_right[:,1].mean())    
    
    nose_x_mean = int(nose_px[0])
    nose_y_mean = int(nose_px[1])
    
    return (eyes_x_mean,eyes_y_mean), (face_x_mean,face_y_mean), (nose_x_mean,nose_y_mean)


def face_angle_updown(eye_mean, face_mean, nose_mean, down_thresh=0.3, up_thresh=0.6):
    mean_line = int(np.mean((eye_mean[1],nose_mean[1])))
    difference = abs(eye_mean[1]-nose_mean[1])
    
    dir_down = False
    dir_up = False
    
    face_line_dist = (face_mean[1]-eye_mean[1])/difference
    face_line_dist = round(face_line_dist,2)
    
    if face_line_dist<down_thresh:
        dir_down = True
    elif face_line_dist>up_thresh:
        dir_up = True
    
        
    return dir_up,dir_down, face_line_dist

def means_leftright(left_right_eyes, face_left_right, nose_px):
    '''
    for calculating left/right face angle
    '''
    
    eyes_x_min = int(left_right_eyes[:,0].min())
    eyes_x_max = int(left_right_eyes[:,0].max())
    
    #face_x_min = int(face_left_right[:,0].min())
    #face_x_max = int(face_left_right[:,0].max())    
    
    nose_x_min = int(nose_px[0])
    # nose_y_mean = int(nose_px[1])
        
    return eyes_x_min, eyes_x_max, nose_x_min

def face_angle_right_left(eyes_x_min, eyes_x_max, nose_x_min, right_thresh=0.3, left_thresh=0.6):
    mean_line = int(np.mean((eyes_x_min, eyes_x_max)))
    difference = abs(eyes_x_min-eyes_x_max)
    
    dir_left = False
    dir_right = False
    
    nose_line_dist = (nose_x_min-eyes_x_min)/difference
    nose_line_dist = round(nose_line_dist,2)
    
    if nose_line_dist<right_thresh:
        dir_right = True
    elif nose_line_dist>left_thresh:
        dir_left = True
        
    return dir_right,dir_left, nose_line_dist


# def check_face_angles(img, results,down_thresh=0.25,up_thresh=0.6,
#                         right_thresh=0.3,left_thresh=0.6):
def check_face_angles(img, results,down_thresh=0.2,up_thresh=0.85,
                            right_thresh=0.3,left_thresh=0.65):
    
    left_right_eyes, face_left_right, nose_px = eye_ear_nose_lines(img, results)
    
    ## face direction up/down
    eye_mean, face_mean, nose_mean = means_updown(left_right_eyes, face_left_right, nose_px) 
    dir_up,dir_down,face_line_dist = face_angle_updown(eye_mean, face_mean, nose_mean,
                                                  down_thresh=down_thresh, up_thresh=up_thresh)
    
    ## face direction right/left
    eyes_x_min, eyes_x_max, nose_x_min = means_leftright(left_right_eyes, face_left_right, nose_px)
    dir_right,dir_left, nose_line_dist = face_angle_right_left(eyes_x_min, eyes_x_max, nose_x_min, 
                          right_thresh=right_thresh, left_thresh=left_thresh)
    
    
    face_angle_flags_dict = {}
    face_angle_flags_dict['UP'] = dir_up
    face_angle_flags_dict['DOWN'] = dir_down
    face_angle_flags_dict['LEFT'] = dir_left
    face_angle_flags_dict['RIGHT'] = dir_right
    
    face_angle_flag = True if True in face_angle_flags_dict.values() else False
    
    return face_angle_flag, face_angle_flags_dict