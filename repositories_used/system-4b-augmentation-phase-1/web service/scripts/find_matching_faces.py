import sys
sys.path.append('..')

import dlib
import numpy as np
import sys
import argparse
import os
import PIL
import cv2
from utils.misc_params import face_matching_args_class
from utils.face_recognition_utils import find_matches
from utils.load_and_preprocess_utils import load_img

if __name__ == "__main__":
    
    # Load the models
    predictor = dlib.shape_predictor("../pretrained_models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    face_rec_model = dlib.face_recognition_model_v1('../pretrained_models/dlib_face_recognition_resnet_model_v1.dat')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", help="Path to new faces to add")
    parser.add_argument("--encodings", help="Path to face encodings numpy array",default='encodings_data/face_encodings.npy')
    parser.add_argument("--encoding_paths", help="Path to encodings paths numpy array",default='encodings_data/face_imgs_paths.npy')
    parser.add_argument("--threshold", default=0.5, help="threshold value for distances between images for matching", type=float)
    parser.add_argument("--max_imgs", help="maximum number of images to return", default=5, type=int)
    parser.add_argument("--path_results", help="Path to new person to add", default = 'matching_faces/')
    #parser.add_argument("--find_nearest_imgs", help="Find nearest images in case the person is not found", action='store_true')
    #parser.add_argument("--return_distance", help="Return the distances with the colsest persons found", action='store_true')
    args = parser.parse_args()
    
    # args = parser.parse_args(args=[])

    
    face_matching_args = face_matching_args_class(args.encodings,
    args.encoding_paths,
    args.threshold,
    args.max_imgs,
    find_nearest_imgs=True,
    return_distance=False)
    
    
    if not os.path.isdir(args.path_results):
        os.mkdir(args.path_results)
    

    all_encodings = np.load(args.encodings)
    encoding_paths = np.load(args.encoding_paths)

    img = load_img(args.img_path)

    matched_imgs_paths = find_matches(PIL.Image.fromarray(img), 
                                        face_matching_args.all_encodings, face_matching_args.encoding_paths, 
                                        predictor, detector, face_rec_model, 
                                        tolerance=face_matching_args.threshold, max_imgs=face_matching_args.max_imgs,
                                        find_nearest=face_matching_args.find_nearest_imgs)
    
      
    for i in range(0,len(matched_imgs_paths)):
        img_name = matched_imgs_paths[i].split('/')[-1]
        new_path = os.path.join(args.path_results, img_name)
        m_img = load_img(matched_imgs_paths[i],rgb=False)

        cv2.imwrite(new_path,m_img)

