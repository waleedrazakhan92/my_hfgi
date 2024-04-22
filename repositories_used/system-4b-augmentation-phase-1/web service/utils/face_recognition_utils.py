
import sys
# sys.path.insert(1,'face_recognition/')
from tqdm import tqdm
import os
import numpy as np
from glob import glob
from utils.load_and_preprocess_utils import load_img_pil
from utils.face_alignment_utils import align_face

def img_2_encoding(image, model_predictor, model_detector, model_face_rec):
    detected_faces = model_detector(image, 1)
    shapes_faces = [model_predictor(image, face) for face in detected_faces]
    return [np.array(model_face_rec.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def get_all_encodings(path_images, predictor, detector, face_rec_model, features_list=[], path_list=[]):
    
    print('Total Images =', len(path_images)) 
    print('Finding embeddings for', len(path_images), 'images.')
    
    for i in tqdm(range(0,len(path_images))):
        img_path = path_images[i]
        img = load_img_pil(img_path)
        try:
            aligned_img = align_face(img, predictor, detector)
            # lab = img_path.split('/')[-2]
            # lab = lab_dict[lab]
            img_enc = img_2_encoding(np.asarray(aligned_img), predictor, detector, face_rec_model)
            
            if np.shape(img_enc) !=(1,128):
                print('Encoding shape',np.shape(img_enc),'is not equal to','(1,128)')
                print('Please check!!!')
                print(img_path)    
            else:
                features_list.append(img_enc)          
                path_list.append(os.path.relpath(img_path))
        except:
            print('Alignment not working for image:',img_path)
        
    return features_list, path_list

def compare_face_encodings(face_encoding, all_encodings, tolerance=0.5):
    dist = np.linalg.norm(all_encodings - face_encoding, axis=2)
    match =  dist<= tolerance
    return match, dist

def find_matches(img, all_encodings, encoding_paths, predictor, detector, face_rec_model, tolerance=0.5, max_imgs=5, find_nearest=False, return_distances=False):
    aligned_img = align_face(img, predictor, detector)
    img_encoding = img_2_encoding(np.asarray(aligned_img), predictor, detector, face_rec_model)
    matches, enc_distances = compare_face_encodings(np.array(img_encoding), np.array(all_encodings), tolerance=tolerance)

    acceptable_distances = enc_distances[np.where(matches==True)[0]]
    sorted_indexes = np.argsort(acceptable_distances.squeeze())
    # print('Found {} matching faces in the tolerance level {}'.format(len(acceptable_distances),tolerance))
    
    if len(acceptable_distances)==0:
        #print('Could not find the face in the system !')
        if find_nearest==True:
            #print('Returning top',max_imgs,'nearest images')
            sorted_indexes = np.argsort(enc_distances.squeeze())[:max_imgs]
            sel_paths = list(map(encoding_paths.__getitem__, sorted_indexes))
            if return_distances==True:
                sel_dists = list(map(enc_distances.__getitem__, sorted_indexes)) 
                return sel_paths, sel_dists
            else:
                return sel_paths
    else:
        encoding_paths = list(map(encoding_paths.__getitem__, np.where(matches==True)[0]))
        #print('This person is:',encoding_paths[sorted_indexes[0]].split('/')[-2])

    sel_paths = list(map(encoding_paths.__getitem__, sorted_indexes))[:max_imgs]
    if return_distances==True:
        sel_dists = list(map(acceptable_distances.__getitem__, sorted_indexes))[:max_imgs]
        return sel_paths, sel_dists
    else:
        return sel_paths


def register_new_person(path_dataset, all_encodings, encoding_paths, predictor, detector, face_rec_model):
    path_images = glob(os.path.join(path_dataset,'*'))
    all_encodings, encoding_paths = get_all_encodings(path_images, predictor, detector, face_rec_model, features_list=list(all_encodings), path_list=list(encoding_paths))
    return all_encodings, encoding_paths

def register_multiple_persons(path_dataset, all_encodings, encoding_paths, predictor, detector, face_rec_model):
    all_folders = os.listdir(path_dataset)

    for i in range(0,len(all_folders)):
        path_images = glob(os.path.join(path_dataset,all_folders[i],'*'))
        all_encodings, encoding_paths = get_all_encodings(path_images, predictor, detector, face_rec_model, features_list=list(all_encodings), path_list=list(encoding_paths))
    
    return all_encodings, encoding_paths
    
def register_all_persons(path_dataset, predictor, detector, face_rec_model):
    path_images = glob(os.path.join(path_dataset,'*'))
    all_encodings, encoding_paths = get_all_encodings(path_images, predictor, detector, face_rec_model)
    return all_encodings, encoding_paths
