import sys
sys.path.append('..')

from utils.face_recognition_utils import register_new_person
import dlib
import numpy as np
import sys
import argparse
import os

if __name__ == "__main__":
    # Load the models
    predictor = dlib.shape_predictor("../pretrained_models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    face_rec_model = dlib.face_recognition_model_v1('../pretrained_models/dlib_face_recognition_resnet_model_v1.dat')

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_new_person", help="Path to new person to add")
    parser.add_argument("--path_encodings_old", help="Path to existing encodings database",default='encodings_data/face_encodings.npy')
    parser.add_argument("--path_encodings_paths_old", help="Path to existing encodings paths database",default='encodings_data/face_imgs_paths.npy')
    parser.add_argument("--savename_encodings_updated", help="name to save encoding vectors as npy array", default='face_encodings_updated.npy')
    parser.add_argument("--savename_encodings_paths_updated", help="name to save encoding paths as npy array", default='face_imgs_paths_updated.npy')
    parser.add_argument("--save_dir", help="Directory to save encoding data", default='encodings_data/')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)


    all_encodings = np.load(args.path_encodings_old)
    encoding_paths = np.load(args.path_encodings_paths_old)

    all_encodings_updated, encoding_paths_updated = register_new_person(args.path_new_person, all_encodings, encoding_paths, predictor, detector, face_rec_model)
    
    np.save(os.path.join(args.save_dir,args.savename_encodings_updated), all_encodings_updated)
    np.save(os.path.join(args.save_dir,args.savename_encodings_paths_updated), encoding_paths_updated)  
