import sys
sys.path.append('..')

import argparse
from glob import glob
import os
from tqdm import tqdm

from utils.feature_extraction_utils import get_cropped_part_overlay_delta_3d, apply_smooth_color,preserve_region,preserve_region_mask
from utils.makeup_regions import *
from utils.load_and_preprocess_utils import load_img
from utils.landmark_utils import find_landmarks
import cv2
import numpy as np

from utils.misc_params import generate_makeup_json
from utils.load_and_preprocess_utils import save_json

from utils.face_parsing_utils import load_parsing_model,parse_img,part_color_dict
parsing_model = load_parsing_model('../pretrained_models/parsing_model.pth')

dir_dict = {'up':'down',
            'left':'right',
            'down':'up',
            'right':'left'
    }

def invert_direction(direction):
    if direction in ['left','right']:
        try:
            return dir_dict[direction]
        except:
            return None
    else:
        return direction
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="path of input images directory")
    parser.add_argument("--path_results", help="path to save resultant image", default='makeup_results/')

    parser.add_argument("--max_faces", help="maximum number of faces in an image to be detected", default=2, type=int)
    parser.add_argument("--min_confidence", help="minimum confidence level for face detection", default=0.5, type=float)
    
    
    ##--------------------------------------
    ## Lips
    ##--------------------------------------
    parser.add_argument("--lip_makeup", help="put makeup on the lips",default='YES', type=str)

    parser.add_argument("--lip_line_color", help="color of lip line",default=[255,20,147], action='store',nargs='*',dest='lip_line_color',type=int)
    
    parser.add_argument("--upper_lip_color", help="color 1 of upper lip",default=[128,0,128], action='store',nargs='*',dest='upper_lip_color',type=int)
    ##parser.add_argument("--upper_lip_color_2", help="color 2 of upper lip",default=None, type=list)
    
    parser.add_argument("--lower_lip_color", help="color 1 of lower lip",default=[128,0,128], action='store',nargs='*',dest='lower_lip_color',type=int)
    ##parser.add_argument("--lower_lip_color_2", help="color 2 of lower lip",default=None, type=list)
    
    parser.add_argument("--int_lip_line", help="overlay intensity of lip line",default=0.5, type=float)
    parser.add_argument("--int_lip_upper", help="overlay intensity of upper lip",default=0.4, type=float)
    parser.add_argument("--int_lip_lower", help="overlay intensity of lower lip",default=0.4, type=float)
    ##--------------------------------------
    ## Eyes
    ##--------------------------------------
    parser.add_argument("--eye_makeup", help="put makeup on the eyes",default='YES', type=str)
    parser.add_argument("--eyeliner", help="put eyeliner on the eyes",default='YES', type=str)

    
    parser.add_argument("--eye_line_color_start", help="start color of eye line",default=[75, 0, 130], action='store',nargs='*',dest='eye_line_color_start',type=int)
    parser.add_argument("--eye_line_color_dest", help="destination color of eye line",default=[75, 0, 130], action='store',nargs='*',dest='eye_line_color_dest',type=int)
    parser.add_argument("--eye_shade_color_top", help="color of eye shade top",default=[75, 0, 130], action='store',nargs='*',dest='eye_shade_color_top',type=int)
    parser.add_argument("--eye_shade_color_bottom", help="color of eye shade bottom",default=[255,20,147], action='store',nargs='*',dest='eye_shade_color_bottom',type=int)
    
    parser.add_argument("--int_eye_line", help="overlay intensity of eye line",default=0.7, type=float)
    parser.add_argument("--int_eye_shade", help="overlay intensity of eye shade",default=1, type=float)
    parser.add_argument("--int_eye_highlight", help="overlay intensity of eye highlighter",default=1, type=float)
    parser.add_argument("--int_eyeliner", help="overlay intensity of eyeliner",default=1, type=float)
    
    parser.add_argument('--eye_line_direction',help="The direction (up,down,left,right,None) of the top eye line colors",default=None)
    parser.add_argument('--eye_shade_direction',help="The direction (up,down,left,right,None) of the shade region colors",default='up')

    ## thickness
    parser.add_argument("--thickness_lip_line", help="thickness of the lip liner", default=5, type=int)
    parser.add_argument("--thickness_eyeliner", help="thickness of the eyeliner", default=3, type=int)

    parser.add_argument("--save_extension", help="save save_extention PNG jpg",default='.jpg', type=str)
    
    # set attributes
    parser.add_argument("--ethnicity", help="set ethnicity of the subject",default='asian',type=str)
    parser.add_argument("--age", help="set the age of the subject", default=25, type=int)
    parser.add_argument("--gender", help="set the gender of the subject", default='female', type=str)
    
    args = parser.parse_args()
    
    assert args.eye_makeup=='YES' or args.eye_makeup=='NO'
    assert args.lip_makeup=='YES' or args.lip_makeup=='NO'
    
    

    if not os.path.isdir(args.path_results):
        os.mkdir(args.path_results)

    all_images = glob(os.path.join(args.image_dir,'*'))
    
    for i in tqdm(range(0,len(all_images))):
        path_image = all_images[i]    
        img = load_img(path_image)
        img_landmarks = find_landmarks(img)
        
        img_parsed = parse_img(img,parsing_model,(512,512))[0]
        img_parsed = cv2.resize(img_parsed,img.shape[::-1][1:], None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

        lip_makeup_img = img.copy()    
        if args.lip_makeup=='YES':
            
            ##--------------------------------------
            ## Lips
            ##--------------------------------------
            lip_makeup_img,f_mask,d_mask = apply_smooth_color(lip_makeup_img, img_landmarks, 
                                    [regions_dict['lip_upper']], 
                                    [deltas['lip_upper']], 
                                    overlay_color=args.upper_lip_color, 
                                    dest_color=args.upper_lip_color,
                                    overlay_intensity=args.int_lip_upper,
                                    close_flag=True,
                                    bezier_flag=False,
                                    shade='down',
                                    line_thickness=2,
                                    fill_mask_flag=True,
                                    shade_type='hsv_grad',
                                    dilate_iters=0,
                                    int_blur=1,
                                    blur_iters=0,
                                    blur_kernel=(51,51),
                                    seam_flag=True
                                    )
            
            lip_makeup_img,f_mask,d_mask = apply_smooth_color(lip_makeup_img, img_landmarks, 
                                    [regions_dict['lip_lower']], 
                                    [deltas['lip_lower']], 
                                    overlay_color=args.lower_lip_color, 
                                    dest_color=args.lower_lip_color,
                                    overlay_intensity=args.int_lip_lower,
                                    close_flag=True,
                                    bezier_flag=False,
                                    shade='down',
                                    line_thickness=2,
                                    fill_mask_flag=True,
                                    shade_type='hsv_grad',
                                    dilate_iters=0,
                                    int_blur=1,
                                    blur_iters=0,
                                    blur_kernel=(51,51),
                                    seam_flag=True
                                    )
            
            lip_makeup_img,f_mask,d_mask = apply_smooth_color(lip_makeup_img, img_landmarks, 
                                    [regions_dict['lip_boundary']], 
                                    [deltas['lip_boundary']], 
                                    overlay_color=args.lip_line_color, 
                                    dest_color=args.lip_line_color,
                                    overlay_intensity=args.int_lip_line,
                                    close_flag=True,
                                    bezier_flag=False,
                                    shade='down',
                                    line_thickness=args.thickness_lip_line,
                                    fill_mask_flag=False,
                                    shade_type='simple_grad',
                                    dilate_iters=0,
                                    int_blur=2,
                                    blur_iters=2,
                                    blur_kernel=(3,3),
                                    seam_flag=True,
                                    seam_mask_iters=5
                                    )



        if args.eye_makeup=='YES':
            ##--------------------------------------
            ## Eyes
            ##--------------------------------------

            eye_shade_img,f_mask,d_mask = apply_smooth_color(lip_makeup_img.copy(), img_landmarks, 
                                    eye_all_regions_r, 
                                    eye_all_deltas_r, 
                                    overlay_color=args.eye_shade_color_top, 
                                    dest_color=args.eye_shade_color_bottom,
                                    overlay_intensity=args.int_eye_shade,
                                    close_flag=True,
                                    bezier_flag='smooth_poly',
                                    shade=args.eye_shade_direction,
                                    line_thickness=5,
                                    fill_mask_flag=True,
                                    shade_type='simple_grad',
                                    dilate_iters=5,
                                    int_blur=1,
                                    blur_iters=2,
                                    blur_kernel=(21,21),
                                    seam_flag=True,
                                    seam_mask_iters=20)
                                    #mask_connections=full_eye_region, mask_deltas=full_eye_region_deltas)
                
            eye_shade_img,f_mask,d_mask = apply_smooth_color(eye_shade_img.copy(), img_landmarks, 
                                    eye_all_regions_l, 
                                    eye_all_deltas_l, 
                                    overlay_color=args.eye_shade_color_top, 
                                    dest_color=args.eye_shade_color_bottom,
                                    overlay_intensity=args.int_eye_shade,
                                    close_flag=True,
                                    bezier_flag='smooth_poly',
                                    shade=invert_direction(args.eye_shade_direction),
                                    line_thickness=5,
                                    fill_mask_flag=True,
                                    shade_type='simple_grad',
                                    dilate_iters=5,
                                    int_blur=1,
                                    blur_iters=2,
                                    blur_kernel=(21,21),
                                    seam_flag=True,
                                    seam_mask_iters=20)
                                    #mask_connections=full_eye_region, mask_deltas=full_eye_region_deltas)

            eye_shade_img = preserve_region(img,img_landmarks,eye_shade_img,
                                     [regions_dict['r_eyeliner'],regions_dict['l_eyeliner']])


            eye_line_img,f_mask,d_mask = apply_smooth_color(lip_makeup_img.copy(), img_landmarks, 
                                    eye_line_regions_r,#[regions_dict['r_eye_line']], 
                                    eye_line_deltas_r,#[deltas['r_eye_line']], 
                                    overlay_color=args.eye_line_color_start, 
                                    dest_color=args.eye_line_color_dest,
                                    overlay_intensity=args.int_eye_line,
                                    close_flag=True,
                                    bezier_flag='smooth_poly_slpev',
                                    shade=args.eye_line_direction,
                                    line_thickness=10,
                                    fill_mask_flag=False,
                                    shade_type='simple_grad',
                                    dilate_iters=5,
                                    int_blur=1,
                                    blur_iters=3,
                                    blur_kernel=(21,21),
                                    seam_flag=True,
                                    seam_mask_iters=20)#,
                                    #mask_connections=full_eye_region, mask_deltas=full_eye_region_deltas)    
            
             
            eye_line_img,f_mask,d_mask = apply_smooth_color(eye_line_img.copy(), img_landmarks, 
                                    eye_line_regions_l,#[regions_dict['r_eye_line']], 
                                    eye_line_deltas_l,#[deltas['r_eye_line']], 
                                    overlay_color=args.eye_line_color_start, 
                                    dest_color=args.eye_line_color_dest,
                                    overlay_intensity=args.int_eye_line,
                                    close_flag=True,
                                    bezier_flag='smooth_poly_slpev',
                                    shade=invert_direction(args.eye_line_direction),
                                    line_thickness=10,
                                    fill_mask_flag=False,
                                    shade_type='simple_grad',
                                    dilate_iters=5,
                                    int_blur=1,
                                    blur_iters=3,
                                    blur_kernel=(21,21),
                                    seam_flag=True,
                                    seam_mask_iters=20)#,
                                    #mask_connections=full_eye_region, mask_deltas=full_eye_region_deltas)    

            eye_line_img = preserve_region(img,img_landmarks,eye_line_img,
                                     [regions_dict['r_eyeliner'],regions_dict['l_eyeliner']])


            eye_highlight_img,f_mask,d_mask = apply_smooth_color(lip_makeup_img.copy(), img_landmarks,                                     
                                    [regions_dict['r_eye_high_1'],
                                    regions_dict['r_eye_high_2'],
                                    regions_dict['l_eye_high_1'],
                                    regions_dict['l_eye_high_2']],
                                    [deltas['r_eye_high_1'],
                                    deltas['r_eye_high_2'],
                                    deltas['l_eye_high_1'],
                                    deltas['l_eye_high_2']],
                                                               
                                    overlay_color=[255,255,255], 
                                    dest_color=[255,255,255],
                                    overlay_intensity=args.int_eye_highlight,
                                    close_flag=True,
                                    bezier_flag=False,
                                    shade=None,
                                    line_thickness=5,
                                    fill_mask_flag=True,
                                    shade_type='simple_grad',
                                    dilate_iters=5,
                                    int_blur=1,
                                    blur_iters=1,
                                    blur_kernel=(21,21),
                                    seam_flag=True,
                                    seam_mask_iters=20)#,
                                    #mask_connections=full_eye_region, mask_deltas=full_eye_region_deltas)

            
            merged_img = np.uint8(eye_shade_img*0.6+ eye_line_img*0.2+ eye_highlight_img*0.2) 
            makeup_img = merged_img.copy()

            if args.eyeliner == 'YES':
                makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks,
                                        eyeliner_regions,#[regions_dict['r_eyeliner']], 
                                        eyeliner_deltas,#[deltas['r_eyeliner']], 
                                        overlay_color=[1,1,1], 
                                        dest_color=[1,1,1],
                                        overlay_intensity=args.int_eyeliner,
                                        close_flag=True,
                                        bezier_flag=False,
                                        shade=None,
                                        line_thickness=args.thickness_eyeliner,
                                        fill_mask_flag=False,
                                        shade_type=None,
                                        dilate_iters=0,
                                        int_blur=1,
                                        blur_iters=1,
                                        blur_kernel=(7,7),
                                        seam_flag=True,
                                        seam_mask_iters=2)#,
                                        #mask_connections=full_eye_region, mask_deltas=full_eye_region_deltas) 
        else:
            makeup_img = lip_makeup_img    
            
        # makeup_img = preserve_region(img,img_landmarks,makeup_img,
        #                              [regions_dict['r_eyeliner'],regions_dict['l_eyeliner']])

        makeup_img = preserve_region_mask(img,makeup_img,img_parsed,['hair','background','neck'])#,'l_eye','r_eye'])
        # makeup_img = np.concatenate((img,makeup_img),1)
        
        makeup_img = cv2.cvtColor(makeup_img,cv2.COLOR_RGB2BGR)

        img_name = path_image.split('/')[-1]
        sp_name,image_ext = os.path.splitext(img_name)

        new_name = sp_name +'_makeup'+args.save_extension#+path_image.split('.')[-1]
        makeup_img_path = os.path.join(args.path_results, new_name)
        cv2.imwrite(makeup_img_path, makeup_img)
        
        if args.eye_makeup=='YES':

            # intermediate steps
            steps_name = os.path.join(args.path_results, sp_name)
            step_1_path = steps_name+'_eyeShadowStep1'+args.save_extension
            step_2_path = steps_name+'_eyeTopLineStep2'+args.save_extension
            step_3_path = steps_name+'_eyeHighlighterStep3'+args.save_extension
            step_4_path = steps_name+'_eyeMergeStep4'+args.save_extension

            cv2.imwrite(step_1_path, cv2.cvtColor(eye_shade_img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(step_2_path, cv2.cvtColor(eye_line_img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(step_3_path, cv2.cvtColor(eye_highlight_img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(step_4_path, cv2.cvtColor(merged_img,cv2.COLOR_RGB2BGR))

            args.eyeShadowStep1 = step_1_path
            args.eyeTopLineStep2 = step_2_path
            args.eyeHighlighterStep3 = step_3_path
            args.eyeMergeStep4 = step_4_path
        else:
            args.eyeShadowStep1 = 'NA'
            args.eyeTopLineStep2 = 'NA'
            args.eyeHighlighterStep3 = 'NA'
            args.eyeMergeStep4 = 'NA'


        # generate and save json
        makeup_json = generate_makeup_json(path_image,makeup_img_path,args)

        img_name = sp_name+'_makeup'
        save_json(img_name, makeup_json, args.path_results)

    
    
if __name__ == '__main__':
    main()
