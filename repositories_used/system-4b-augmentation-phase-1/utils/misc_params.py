import os
import numpy as np
class blend_args_class:
    def __init__(self,path_image,path_results='output_images/',path_json='output_jsons/',blend_modes=['addition','divide'],
        opacity=0.5):#,num_pieces=50,num_directions=2):
        self.path_image=path_image
        self.path_results=path_results
        self.path_json=path_json
        self.blend_modes=blend_modes
        self.opacity=opacity
        # self.num_pieces=num_pieces
        # self.num_directions=num_directions
  

class face_matching_args_class:
    def __init__(self,path_encodings='encodings_data/face_encodings.npy',path_encodings_paths='encodings_data/face_imgs_paths.npy',
                 threshold=0.5,max_imgs=1,find_nearest_imgs=True,return_distance=False):
        
        self.path_encodings = path_encodings
        self.path_encodings_paths = path_encodings_paths
        self.threshold = threshold
        self.max_imgs = max_imgs
        self.find_nearest_imgs = find_nearest_imgs
        self.return_distance = return_distance
        self.all_encodings = np.load(self.path_encodings)
        self.encoding_paths = np.load(self.path_encodings_paths)




def make_final_json(blending_json,measurements_json,args):
    final_out_json = {}
    final_out_json['image_name'] = blending_json['image_name']

    final_out_json['image_paths'] = {}
    final_out_json['image_paths']['image_path'] = blending_json['image_path']

    final_out_json['image_paths']['swatch_paths'] = blending_json['swatch_paths'] 

    final_out_json['image_paths']['annotation_paths'] = measurements_json['annotation_paths']
    
    final_out_json['lighting'] = {}
    final_out_json['lighting']['light_angle'] = blending_json['light_angle']
    final_out_json['lighting']['light_angle_flag'] = blending_json['light_angle_flag']
    final_out_json['lighting']['light_source'] = blending_json['light_source']
    final_out_json['lighting']['light_type'] = blending_json['light_type']
    final_out_json['lighting']['light_source'] = 'NA'
    final_out_json['lighting']['light_type'] = 'NA'
    
    
    final_out_json['measurements'] = {}
    final_out_json['measurements']['landmark_distances'] = measurements_json['landmark_distances']
    final_out_json['measurements']['landmark_coordinates'] = measurements_json['landmarks']
    final_out_json['measurements']['face_shape'] = measurements_json['face_shape']
    
    final_out_json['features'] = {}
    final_out_json['features']['age'] = args.age
    final_out_json['features']['gender'] = args.gender
    final_out_json['features']['ethnicity'] = args.ethnicity
    
      
    final_out_json['features']['skin_color'] = {}
    final_out_json['features']['ITA_values'] = {}
    final_out_json['features']['LAB_values'] = {}
    final_out_json['features']['ITA_labels'] = {}

    for bl in blending_json['ITA_values'].keys():
        # if bl=='face':
        #     final_out_json['features']['skin_color'][bl] = blending_json['RGB_values'][bl]
        #     final_out_json['features']['ITA_values'][bl] = blending_json['ITA_values'][bl]
        #     final_out_json['features']['LAB_values'][bl] = blending_json['LAB_values'][bl]
        #     final_out_json['features']['ITA_labels'][bl] = blending_json['ITA_labels'][bl]
        # else:
        #     final_out_json['features']['skin_color'][bl] = blending_json['RGB_values'][bl]
        #     final_out_json['features']['ITA_values'][bl] = blending_json['ITA_values'][bl]
        #     final_out_json['features']['LAB_values'][bl] = blending_json['LAB_values'][bl]
        #     final_out_json['features']['ITA_labels'][bl] = blending_json['ITA_labels'][bl]
        final_out_json['features']['skin_color'][bl] = blending_json['RGB_values'][bl]
        final_out_json['features']['ITA_values'][bl] = blending_json['ITA_values'][bl]
        final_out_json['features']['LAB_values'][bl] = blending_json['LAB_values'][bl]
        final_out_json['features']['ITA_labels'][bl] = blending_json['ITA_labels'][bl]

        
    return final_out_json
   


def generate_makeup_json(img_path,makeup_img_path, args):

    ## json data

    img_name = img_path.split('/')[-1]
    img_name,image_ext = os.path.splitext(img_name)

    makeup_json = {}
    makeup_json['image_path'] = img_path
    makeup_json['image_name'] = img_name
    makeup_json['output_path'] = makeup_img_path
    makeup_json['eye_makeup'] = args.eye_makeup
    makeup_json['lip_makeup'] = args.lip_makeup

    # intermediate steps
    makeup_json['eye_makeup_steps'] = {}
    makeup_json['eye_makeup_steps']['eyeShadowStep1'] = args.eyeShadowStep1
    makeup_json['eye_makeup_steps']['eyeTopLineStep2'] = args.eyeTopLineStep2
    makeup_json['eye_makeup_steps']['eyeHighlighterStep3'] = args.eyeHighlighterStep3
    makeup_json['eye_makeup_steps']['eyeMergeStep4'] = args.eyeMergeStep4
 

    # face features
    makeup_json['features'] = {}
    makeup_json['features']['age'] = args.age
    makeup_json['features']['gender'] = args.gender
    makeup_json['features']['ethnicity'] = args.ethnicity


    makeup_json['eye_features'] = {}
    makeup_json['eye_features']['colors'] = {}
    makeup_json['eye_features']['intensities'] = {}
    makeup_json['eye_features']['directions'] = {}
    makeup_json['eye_features']['thickness'] = {}

    if args.eye_makeup=='YES':
        makeup_json['eye_features']['eyeliner'] = str(args.eyeliner)

        # eye line color
        makeup_json['eye_features']['colors']['top_eye_line_color_start'] = args.eye_line_color_start
        makeup_json['eye_features']['colors']['top_eye_line_color_stop'] = args.eye_line_color_dest

        # eye shade color
        makeup_json['eye_features']['colors']['eye_shade_color_top'] = args.eye_shade_color_top
        makeup_json['eye_features']['colors']['eye_shade_color_bottom'] = args.eye_shade_color_bottom

        # intensities
        makeup_json['eye_features']['intensities']['eyeliner'] = args.int_eyeliner
        makeup_json['eye_features']['intensities']['top_eye_line'] = args.int_eye_line
        makeup_json['eye_features']['intensities']['eye_shade'] = args.int_eye_shade
        makeup_json['eye_features']['intensities']['highlighter'] = args.int_eye_highlight

        # directions
        makeup_json['eye_features']['directions']['eye_line'] = args.eye_line_direction
        makeup_json['eye_features']['directions']['eye_shade'] = args.eye_shade_direction

        # thickness
        makeup_json['eye_features']['thickness']['eyeliner'] = args.thickness_eyeliner

    else:
        makeup_json['eye_features']['eyeliner'] = str(args.eyeliner)

        # eye line color
        makeup_json['eye_features']['colors']['top_eye_line_color_start'] = 'NA'
        makeup_json['eye_features']['colors']['top_eye_line_color_stop'] = 'NA'

        # eye shade color
        makeup_json['eye_features']['colors']['eye_shade_color_top'] = 'NA'
        makeup_json['eye_features']['colors']['eye_shade_color_bottom'] = 'NA'

        # intensities
        makeup_json['eye_features']['intensities']['eyeliner'] = 'NA'
        makeup_json['eye_features']['intensities']['top_eye_line'] = 'NA'
        makeup_json['eye_features']['intensities']['eye_shade'] = 'NA'
        makeup_json['eye_features']['intensities']['highlighter'] = 'NA'

        # directions
        makeup_json['eye_features']['directions']['eye_line'] = 'NA'
        makeup_json['eye_features']['directions']['eye_shade'] = 'NA'

        # thickness
        makeup_json['eye_features']['thickness']['eyeliner'] = 'NA'
        

    makeup_json['lip_features'] = {}
    makeup_json['lip_features']['colors'] = {}
    makeup_json['lip_features']['intensities'] = {}
    makeup_json['lip_features']['thickness'] = {}

    if args.lip_makeup=='YES':

        # colors
        makeup_json['lip_features']['colors']['lip_line_color'] = args.lip_line_color
        makeup_json['lip_features']['colors']['upper_lip_color'] = args.upper_lip_color
        makeup_json['lip_features']['colors']['lower_lip_color'] = args.lower_lip_color

        # intensities
        makeup_json['lip_features']['intensities']['lip_line'] = args.int_lip_line
        makeup_json['lip_features']['intensities']['upper_lip'] = args.int_lip_upper
        makeup_json['lip_features']['intensities']['lower_lip'] = args.int_lip_lower

        # thickness 
        makeup_json['lip_features']['thickness']['lip_line'] = args.thickness_lip_line

    else:
        # colors
        makeup_json['lip_features']['colors']['lip_line_color'] = 'NA'
        makeup_json['lip_features']['colors']['upper_lip_color'] = 'NA'
        makeup_json['lip_features']['colors']['lower_lip_color'] = 'NA'

        # intensities
        makeup_json['lip_features']['intensities']['lip_line'] = 'NA'
        makeup_json['lip_features']['intensities']['upper_lip'] = 'NA'
        makeup_json['lip_features']['intensities']['lower_lip'] = 'NA'

        # thickness 
        makeup_json['lip_features']['thickness']['lip_line'] = 'NA'
    
    return makeup_json



def generate_contour_json(img_path,contour_img_path, args):

    ## json data
    img_name = img_path.split('/')[-1]
    img_name,image_ext = os.path.splitext(img_name)

    contour_json = {}
    contour_json['image_path'] = img_path
    contour_json['image_name'] = img_name
    contour_json['output_path'] = contour_img_path
    # face features
    contour_json['features'] = {}
    contour_json['features']['age'] = args.age
    contour_json['features']['gender'] = args.gender
    contour_json['features']['ethnicity'] = args.ethnicity
    contour_json['features']['face_shape'] = args.face_shape
    
    contour_json['contour_features'] = {}
    contour_json['contour_features']['colors'] = {}
    contour_json['contour_features']['intensities'] = {}
    
    # colors
    contour_json['contour_features']['colors']['dark'] = args.color_dark
    contour_json['contour_features']['colors']['light'] = args.color_light
    
    # intensities
    contour_json['contour_features']['intensities']['chin_dark'] = args.int_chin_dark
    contour_json['contour_features']['intensities']['chin_light'] = args.int_chin_light
    
    contour_json['contour_features']['intensities']['nose_dark'] = args.int_nose_dark
    contour_json['contour_features']['intensities']['nose_light'] = args.int_nose_light
    
    contour_json['contour_features']['intensities']['forehead_dark'] = args.int_forehead_dark
    contour_json['contour_features']['intensities']['forehead_light'] = args.int_forehead_light
    
    return contour_json
    