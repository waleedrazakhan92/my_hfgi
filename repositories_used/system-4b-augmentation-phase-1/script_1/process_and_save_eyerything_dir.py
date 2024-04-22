import sys
sys.path.append('..')

# import cv2
from glob import glob
import os
from tqdm import tqdm

# face recognition
import dlib

# lighting
from utils.change_lighting_utils import load_light_model

## loading models
# parsing 
from utils.face_parsing_utils import load_parsing_model

# bg utils
from utils.bg_removal_utils import load_modnet

# beauty score
from utils.beauty_score_utils import load_beauty_model

# process everything and get jsons
from utils.process_image_and_jsons_utils import process_everything


from utils.misc_params import blend_args_class,face_matching_args_class
import argparse
from utils.load_and_preprocess_utils import save_json

# load model
light_model_ckpt = '../pretrained_models/trained_model_1024_03.t7'
light_network = load_light_model(light_model_ckpt)

parsing_model = load_parsing_model('../pretrained_models/parsing_model.pth')

# bg model
modnet = load_modnet(ckpt_path='../pretrained_models/modnet_photographic_portrait_matting.ckpt')

# beauty score model
beauty_model = load_beauty_model('../pretrained_models/ComboNet_SCUTFBP5500.pth', backbone='SEResNeXt50')

# face recognition models
predictor = dlib.shape_predictor("../pretrained_models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
face_rec_model = dlib.face_recognition_model_v1('../pretrained_models/dlib_face_recognition_resnet_model_v1.dat')


# args
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", help="path of input image directory")
parser.add_argument("--path_results", help="path to save resultant images", default='annotation_images/')

parser.add_argument("--max_faces", help="maximum number of faces in an image to be detected", default=2, type=int)
parser.add_argument("--min_confidence", help="minimum confidence level for face detection", default=0.5, type=float)

# blending options
parser.add_argument("--blend_modes", help="select blending modes", default=['addition','divide'], 
                    action='store',nargs='*',dest='blend_modes',type=str)
parser.add_argument("--opacity", help="set opacity level for blending mode", default=0.5, type=float)
parser.add_argument("--num_pieces", help="number of pieces to divide the light globe in", default=50, type=int)
parser.add_argument("--num_directions", help="number of directions to find the light direction from", 
                    default=2, type=int)
parser.add_argument("--mean_thresh", help="threshold for difference between min and max light directions", default=190, type=int)

# degree of background change color 
# parser.add_argument("--bg_degree", help="background degree color range 0 to 1 0 means lighter 1 means darker", 
#                     default=0.6, type=float)
parser.add_argument("--bg_color",help="background color for image. A float between(0-1) or a specific rgb value [255 0 0]"
                    ,default=[255,255,255], action='store',nargs='*',dest='bg_color',type=float)

# set attributes
parser.add_argument("--ethnicity", help="set ethnicity of the subject",default='asian',type=str)
parser.add_argument("--age", help="set the age of the subject", default=25, type=int)
parser.add_argument("--gender", help="set the gender of the subject", default='female', type=str)

# face rec parameters
parser.add_argument("--encodings", help="Path to face encodings numpy array",default='encodings_data/face_encodings.npy')
parser.add_argument("--encoding_paths", help="Path to encodings paths numpy array",default='encodings_data/face_imgs_paths.npy')
parser.add_argument("--threshold", default=0.5, help="threshold value for distances between images for matching", type=float)
parser.add_argument("--max_imgs", help="maximum number of images to return", default=1, type=int)
#parser.add_argument("--return_distance", help="Return the distances with the colsest persons found", action='store_true')

parser.add_argument("--path_rejected", help="path to save rejected images and jsons", default='rejected_images/')


args = parser.parse_args()

blending_args = blend_args_class(None,
              args.path_results,
              args.path_results,
              args.blend_modes,
              args.opacity,
              )


face_matching_args = face_matching_args_class(args.encodings,
    args.encoding_paths,
    args.threshold,
    args.max_imgs,
    find_nearest_imgs=True,
    return_distance=False)


def main():
    if not os.path.isdir(args.path_results):
        os.mkdir(args.path_results)
        
    if not os.path.isdir(args.path_rejected):
        os.mkdir(args.path_rejected)

    args.path_rejected_light = os.path.join(args.path_rejected,'light_rejected/')
    args.path_rejected_other = os.path.join(args.path_rejected,'other_rejected/')

    if not os.path.isdir(args.path_rejected_light):
        os.mkdir(args.path_rejected_light)

    if not os.path.isdir(args.path_rejected_other):
        os.mkdir(args.path_rejected_other)

    
    all_images = glob(os.path.join(args.image_dir,'*'))
    skipped_images = {}
    for i in tqdm(range(0,len(all_images))):
        
        img_path = all_images[i]##args.path_image
        
        blending_args.path_image = img_path
        
        try:
            process_everything(img_path, args,blending_args,face_matching_args,
                           parsing_model,light_network,modnet,beauty_model,
                          predictor,detector,face_rec_model)
        except:
            skipped_images[i] = img_path
            # print('Encountered unknown error: Skipping ',img_path)
    
    save_json('skipped_images', skipped_images, args.path_rejected)
        
if __name__ == '__main__':
    main()
