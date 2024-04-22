from utils.measurement_utils import calculate_landmark_measurements

from math import atan2, degrees
import numpy as np
from utils.measurement_utils import calculate_landmark_distance,convert_landmark_to_px,idx_2_pix_extended

def find_angle_3pts(p1,p2, centre,height,width):

    x1, y1 = p1
    y1 = height - y1
    
    x2, y2 = centre
    y2 = height - y2

    x3,y3 = p2
    y3 = height - y3
    
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    
    final_angle = round((deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)),2)
    
    if final_angle>180:
        final_angle = 360-final_angle
        
    return round(final_angle)

def detect_landmark_angle(img_landmarks,landmark_pts,img_parsed=None,image_cols=1024, image_rows=1024):
    '''
    finds angles between three landmarks
    landmark_pts is a list of three landmark points. point1, center point, point2
    '''
    if np.all(img_parsed)==None:
        p1 = convert_landmark_to_px(landmark_pts[0],img_landmarks,image_cols,image_rows)
        center = convert_landmark_to_px(landmark_pts[1],img_landmarks,image_cols,image_rows)
        p2 = convert_landmark_to_px(landmark_pts[2],img_landmarks,image_cols,image_rows)
    else:
        p1 = idx_2_pix_extended(landmark_pts[0],img_landmarks,img_parsed,image_cols,image_rows)
        center = idx_2_pix_extended(landmark_pts[1],img_landmarks,img_parsed,image_cols,image_rows)
        p2 = idx_2_pix_extended(landmark_pts[2],img_landmarks,img_parsed,image_cols,image_rows)
    
    landmark_angle = find_angle_3pts(p1,p2,center,image_rows,image_cols)
    return landmark_angle

def detect_chin_shape_angle(img_landmarks,landmark_pts,thresh_angle=118,image_cols=1024, image_rows=1024):
    '''
    checks if the angle between three points is less than some angle then chin is pointy
    '''
    chin_angle = detect_landmark_angle(img_landmarks,landmark_pts,
                                      image_cols=image_cols, image_rows=image_rows)
    
    if chin_angle<thresh_angle:
        shape = 'pointy'
    else:
        shape = 'round'
    
    # print('jaw_angle:',jaw_angle)
    return chin_angle,shape
    
def detect_forehead_shape_angle(img_landmarks,landmark_pts,img_parsed,thresh_angle=180,image_cols=1024, image_rows=1024):
    forehead_angle = detect_landmark_angle(img_landmarks,landmark_pts,img_parsed,
                                      thresh_angle=thresh_angle,image_cols=image_cols, image_rows=image_rows)
    
    if forehead_angle<thresh_angle:
        shape = 'heart'
    else:
        shape = 'round'
    
    return shape


def check_shapes_and_distances(ratios_dict,thresh_dict,checks_flags):
    dist = 0
    for k in thresh_dict.keys():
        # if checks_flags[k]==False:
        if k=='chin_angle':
            ## dividing(normalizing) "chin_angle" by 180 because its a larger value and would otherwise 
            ## effect the distance very much
            dist = round(dist + abs(ratios_dict[k]-thresh_dict[k])/180,3)
            # dist = round(dist + abs(ratios_dict[k]-thresh_dict[k]),3)
        else:
            dist = round(dist + abs(ratios_dict[k]-np.mean(thresh_dict[k])),3)
            
    if False in checks_flags.values():
        return (False,dist)
    else:
        return (True,dist)

def shape_check_round(ratios_dict):
    thresh_dict = {}
    thresh_dict['tragi_jaw'] = 1.25
    thresh_dict['tragi_height'] = 0.9
    thresh_dict['chin_angle'] = 120
    thresh_dict['forehead_jaw'] = 0.93
    
    checks_flags = {}
    # checks_flags['chin_shape']=True if ratios_dict['chin_shape']=='round' else False
    checks_flags['tragi_jaw']=True if ratios_dict['tragi_jaw']>=thresh_dict['tragi_jaw'] else False    
    checks_flags['tragi_height']=True if ratios_dict['tragi_height']>=thresh_dict['tragi_height'] else False
    checks_flags['chin_angle']=True if ratios_dict['chin_angle']>=thresh_dict['chin_angle'] else False
    checks_flags['forehead_jaw']=True if ratios_dict['forehead_jaw']>=thresh_dict['forehead_jaw'] else False
    
    return check_shapes_and_distances(ratios_dict,thresh_dict,checks_flags)
    
def shape_check_square(ratios_dict):
    thresh_dict = {}
    thresh_dict['tragi_jaw'] = 1.25
    thresh_dict['tragi_height'] = (0.8,0.9)#0.67
    thresh_dict['chin_angle'] = 125
    thresh_dict['forehead_jaw'] = 0.93
    
    checks_flags = {} 
    # checks_flags['chin_shape']=True if ratios_dict['chin_shape']=='round' else False
    checks_flags['tragi_jaw']=True if ratios_dict['tragi_jaw']<=thresh_dict['tragi_jaw'] else False    
    checks_flags['tragi_height']=True if (ratios_dict['tragi_height']>=thresh_dict['tragi_height'][0]
                                         and ratios_dict['tragi_height']<thresh_dict['tragi_height'][1]
                                         ) else False
    checks_flags['chin_angle']=True if ratios_dict['chin_angle']>=thresh_dict['chin_angle'] else False
    checks_flags['forehead_jaw']=True if ratios_dict['forehead_jaw']<thresh_dict['forehead_jaw'] else False
    
    return check_shapes_and_distances(ratios_dict,thresh_dict,checks_flags)

def shape_check_oval(ratios_dict):
    
    thresh_dict = {}
    thresh_dict['tragi_jaw'] = 1.23
    thresh_dict['tragi_height'] = (0.77,0.82)
    thresh_dict['chin_angle'] = 124
    thresh_dict['forehead_jaw'] = 0.91
    
    checks_flags = {}  
    # checks_flags['chin_shape']=True if ratios_dict['chin_shape']=='round' else False
    checks_flags['tragi_jaw']=True if ratios_dict['tragi_jaw']>=thresh_dict['tragi_jaw'] else False
    checks_flags['tragi_height']=True if (ratios_dict['tragi_height']>=thresh_dict['tragi_height'][0]
                                         and ratios_dict['tragi_height']<thresh_dict['tragi_height'][1]
                                         ) else False
    checks_flags['chin_angle']=True if ratios_dict['chin_angle']<=thresh_dict['chin_angle'] else False
    checks_flags['forehead_jaw']=True if ratios_dict['forehead_jaw']<=thresh_dict['forehead_jaw'] else False
    
    return check_shapes_and_distances(ratios_dict,thresh_dict,checks_flags)
    
    
def shape_check_oblong(ratios_dict):
    thresh_dict = {}
    thresh_dict['tragi_jaw'] = 1.3
    thresh_dict['tragi_height'] = (0.75,0.81)
    thresh_dict['chin_angle'] = 120
    thresh_dict['forehead_jaw'] = 0.89
    
    checks_flags = {} 
    # checks_flags['chin_shape']=True if ratios_dict['chin_shape']=='round' else False
    checks_flags['tragi_jaw']=True if ratios_dict['tragi_jaw']<=thresh_dict['tragi_jaw'] else False    
    checks_flags['tragi_height']=True if (ratios_dict['tragi_height']>=thresh_dict['tragi_height'][0]
                                         and ratios_dict['tragi_height']<thresh_dict['tragi_height'][1]
                                         ) else False
    checks_flags['chin_angle']=True if ratios_dict['chin_angle']<thresh_dict['chin_angle'] else False
    checks_flags['forehead_jaw']=True if ratios_dict['forehead_jaw']>=thresh_dict['forehead_jaw'] else False
    
    return check_shapes_and_distances(ratios_dict,thresh_dict,checks_flags)
    
def shape_check_triangle(ratios_dict):
    thresh_dict = {}
    thresh_dict['tragi_jaw'] = 1.3
    thresh_dict['tragi_height'] = (0.81,0.86)
    thresh_dict['chin_angle'] = 117
    thresh_dict['forehead_jaw'] = 0.89
    
    checks_flags = {}  
    # checks_flags['chin_shape']=True if ratios_dict['chin_shape']=='pointy' else False
    checks_flags['tragi_jaw']=True if ratios_dict['tragi_jaw']>=thresh_dict['tragi_jaw'] else False
    checks_flags['tragi_height']=True if (ratios_dict['tragi_height']>=thresh_dict['tragi_height'][0]
                                         and ratios_dict['tragi_height']<thresh_dict['tragi_height'][1]
                                         ) else False
    checks_flags['chin_angle']=True if ratios_dict['chin_angle']<=thresh_dict['chin_angle'] else False
    checks_flags['forehead_jaw']=True if ratios_dict['forehead_jaw']>=thresh_dict['forehead_jaw'] else False
    
    return check_shapes_and_distances(ratios_dict,thresh_dict,checks_flags)


def generate_ratios_dict(img,img_landmarks,meas_dict):
    '''
    calculates the ratios between different face parts to assist in detecting face shapes 
    '''
    image_rows,image_cols,_ = img.shape
    
    ratios_dict = {}
    ratios_dict['forehead_tragi'] = round(meas_dict['forehead_width']/meas_dict['inter_tragi'],2)
    ratios_dict['tragi_jaw'] = round(meas_dict['inter_tragi']/meas_dict['low_jaw_wid'],2)
    ratios_dict['tragi_height'] = round(meas_dict['inter_tragi']/meas_dict['face_height_mp'],2)
    
    # ratios_dict['forehead_cheeks'] = round(meas_dict['forehead_width']/meas_dict['cheek_bones_width'],2)
    # ratios_dict['cheeks_jaw'] = round(meas_dict['cheek_bones_width']/meas_dict['low_jaw_wid'],2)
    # ratios_dict['cheeks_height'] = round(meas_dict['cheek_bones_width']/meas_dict['face_height'],2)
    
    # ratios_dict['forehead_cheeks_jaws_equal'] = round(ratios_dict['forehead_cheeks']*ratios_dict['cheeks_jaw'],2)
    
    
    ratios_dict['forehead_jaw'] = round(meas_dict['forehead_width']/meas_dict['low_jaw_wid'],2)
    ratios_dict['chin_jaw'] = round(meas_dict['chin']/meas_dict['low_jaw_wid'],2)
    ratios_dict['chin_tragi'] = round(meas_dict['chin']/meas_dict['inter_tragi'],2)
    
    jaw_angle_and_shape_right = detect_chin_shape_angle(img_landmarks, landmark_pts=[234,58,152],
                                                           thresh_angle=118,
                                                            image_cols=image_cols, image_rows=image_rows)
    
    jaw_angle_and_shape_left = detect_chin_shape_angle(img_landmarks, landmark_pts=[454,288,152],
                                                           thresh_angle=118,
                                                            image_cols=image_cols, image_rows=image_rows)
        
    ratios_dict['jaw_angle'] = min(jaw_angle_and_shape_right[0],jaw_angle_and_shape_left[0]) 
    
    ratios_dict['chin_angle'], ratios_dict['chin_shape'] = detect_chin_shape_angle(img_landmarks, landmark_pts=[136,152,365],
                                                           thresh_angle=118,
                                                            image_cols=image_cols, image_rows=image_rows)
    
    return ratios_dict


def decide_face_shape(ratios_dict):
    '''
    checks is a face meets all the criterias for different face shapes
    if not then decides based on the distance that is calculated between the face ratios and the criteria ratios
    '''
    detected_shapes_dict = {}
    detected_shapes_dict['round'] = shape_check_round(ratios_dict)
    detected_shapes_dict['square'] = shape_check_square(ratios_dict)
    detected_shapes_dict['oblong'] = shape_check_oblong(ratios_dict)
    detected_shapes_dict['triangle'] = shape_check_triangle(ratios_dict)
    detected_shapes_dict['oval'] = shape_check_oval(ratios_dict)
    
    # print(detected_shapes_dict)
    # if True for more than one shape then decide on distance
    detected_shape = None
    min_dist = 1000
    for k in detected_shapes_dict.keys():
        if detected_shapes_dict[k][0]==True:
            if detected_shapes_dict[k][1]<=min_dist:
                detected_shape = k

    # if doesn't meet any face criteria then decide on distance
    if detected_shape==None:
        for k in detected_shapes_dict.keys():
            if detected_shapes_dict[k][1]<=min_dist:
                min_dist = detected_shapes_dict[k][1]
                detected_shape = k
                

    return detected_shape

def detect_face_shape(img,img_landmarks,img_parsed):
    meas_dict = calculate_landmark_measurements(img, img_landmarks, img_parsed=img_parsed)
    ratios_dict = generate_ratios_dict(img,img_landmarks,meas_dict)
    return decide_face_shape(ratios_dict)