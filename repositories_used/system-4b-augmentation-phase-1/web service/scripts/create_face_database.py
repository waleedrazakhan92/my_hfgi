import sys
sys.path.append('..')

from utils.face_recognition_utils import register_all_persons
import dlib
import numpy as np
import argparse
import os

if __name__ == "__main__":
    # Load the models
    predictor = dlib.shape_predictor("../pretrained_models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    face_rec_model = dlib.face_recognition_model_v1('../pretrained_models/dlib_face_recognition_resnet_model_v1.dat')

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", help="Path to dataset")
    parser.add_argument("--savename_encodings", help="name to save encoding vectors as npy array", default='face_encodings.npy')
    parser.add_argument("--savename_encodings_paths", help="name to save encoding paths as npy array", default='face_imgs_paths.npy')
    parser.add_argument("--save_dir", help="Directory to save encoding data", default='encodings_data/')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    all_encodings, encoding_paths = register_all_persons(args.path_dataset, predictor, detector, face_rec_model)
    
    np.save(os.path.join(args.save_dir,args.savename_encodings), all_encodings)
    np.save(os.path.join(args.save_dir,args.savename_encodings_paths), encoding_paths) 
