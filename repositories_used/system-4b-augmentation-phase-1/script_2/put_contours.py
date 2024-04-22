import sys
sys.path.append('..')

import argparse
import os

from utils.feature_extraction_utils import get_cropped_part_overlay_delta_3d, apply_smooth_color,preserve_region,preserve_region_mask
from utils.makeup_regions import *
from utils.load_and_preprocess_utils import load_img
from utils.landmark_utils import find_landmarks
import cv2
import numpy as np

from utils.misc_params import generate_contour_json
from utils.load_and_preprocess_utils import save_json

from utils.face_parsing_utils import load_parsing_model,parse_img,part_color_dict
parsing_model = load_parsing_model('../pretrained_models/parsing_model.pth')

from utils.contour_templates import *
from utils.face_shape_utils import detect_face_shape

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_image", help="path of input image")
    parser.add_argument("--path_results", help="path to save resultant image", default='contouring_results/')

    parser.add_argument("--max_faces", help="maximum number of faces in an image to be detected", default=2, type=int)
    parser.add_argument("--min_confidence", help="minimum confidence level for face detection", default=0.5, type=float)


    ##--------------------------------------
    ## Cheeks
    ##--------------------------------------
    parser.add_argument("--color_dark", help="color of dark areas of contours",default=[150,75,0], action='store',nargs='*',dest='color_dark',type=int)
    parser.add_argument("--color_light", help="color of light areas of contours",default=[255,255,200], action='store',nargs='*',dest='color_light',type=int)

    parser.add_argument("--int_cheek_dark", help="overlay intensity cheeks contours",default=0.5, type=float)
    parser.add_argument("--int_cheek_light", help="overlay intensity cheeks contours",default=0.5, type=float)

    ##--------------------------------------
    ## Chin
    ##--------------------------------------
    parser.add_argument("--int_chin_dark", help="overlay intensity chin contours",default=0.4, type=float)
    parser.add_argument("--int_chin_light", help="overlay intensity chin contours",default=0.4, type=float)

    ##--------------------------------------
    ## Nose
    ##--------------------------------------
    parser.add_argument("--int_nose_dark", help="overlay intensity nose contours",default=0.4, type=float)
    parser.add_argument("--int_nose_light", help="overlay intensity nose contours",default=0.4, type=float)

    ##--------------------------------------
    ## Forehead
    ##--------------------------------------
    parser.add_argument("--int_forehead_dark", help="overlay intensity forehead contours",default=0.5, type=float)
    parser.add_argument("--int_forehead_light", help="overlay intensity forehead contours",default=0.5, type=float)

    # extension to save the image with
    parser.add_argument("--save_extension", help="save save_extention PNG jpg",default='.jpg', type=str)

    # template selection
    parser.add_argument('--template',help='the contour template you would like to use',default=6,type=int)

    # set attributes
    parser.add_argument("--ethnicity", help="set ethnicity of the subject",default='asian',type=str)
    parser.add_argument("--age", help="set the age of the subject", default=25, type=int)
    parser.add_argument("--gender", help="set the gender of the subject", default='female', type=str)

    args = parser.parse_args()
    assert args.template==6 or args.template==1 or args.template==2 or args.template==3 
    
    
    if not os.path.isdir(args.path_results):
        os.mkdir(args.path_results)

    img = load_img(args.path_image)
    img_landmarks = find_landmarks(img)

    img_parsed = parse_img(img,parsing_model,(512,512))[0]
    img_parsed = cv2.resize(img_parsed,img.shape[::-1][1:], None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

    face_shape = detect_face_shape(img,img_landmarks,img_parsed)
    args.face_shape = face_shape


    if args.template==6:
        makeup_img = put_contour_template_6(img,img_landmarks,img_parsed,args)
    elif args.template==1:
        makeup_img = put_contour_template_1(img,img_landmarks,img_parsed,args)
    elif args.template==2:
        makeup_img = put_contour_template_2(img,img_landmarks,img_parsed,args)
    elif args.template==3:
        makeup_img = put_contour_template_3(img,img_landmarks,img_parsed,args)
    else:
        print('ERROR: Please select valid template ID.')
        return
    
    makeup_img = preserve_region_mask(img,makeup_img,img_parsed,['hair','u_lip','l_lip','background','neck','l_brow', 'r_brow'])
    makeup_img = cv2.cvtColor(makeup_img,cv2.COLOR_RGB2BGR)
    

    img_name = args.path_image.split('/')[-1]
    sp_name,image_ext = os.path.splitext(img_name)

    new_name = sp_name +'_contours'+args.save_extension
    makeup_img_path = os.path.join(args.path_results, new_name)
    cv2.imwrite(makeup_img_path, makeup_img)
    
    makeup_json = generate_contour_json(args.path_image,makeup_img_path,args)

    img_name = sp_name+'_contour'
    save_json(img_name, makeup_json, args.path_results)
    
    
if __name__ == '__main__':
    main()
