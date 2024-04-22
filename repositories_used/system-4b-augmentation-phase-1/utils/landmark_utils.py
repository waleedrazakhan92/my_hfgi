import json
from tqdm import tqdm
import cv2
from glob import glob
import os
from utils.load_and_preprocess_utils import load_img

from protobuf_to_dict import protobuf_to_dict
import numpy as np


from utils.drawing_spec_utils import *

def add_landmarks_id(img_landmarks):
    landmarks_with_ids = protobuf_to_dict(img_landmarks.multi_face_landmarks[0])
    for i in range(0,len(landmarks_with_ids['landmark'])):
        landmarks_with_ids['landmark'][i]['id']=str(i)

    return landmarks_with_ids


def find_landmarks(img, max_faces=2, min_confidence=0.5):
    # Run MediaPipe Face Mesh.
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence) as face_mesh:

        results = face_mesh.process(img)
    
    # if not results.multi_face_landmarks:
    #     print('No landmarks found')

    return results

def draw_landmarks(img, landmark_results, landmark_spec, tesselation=True, contours=True, iris=True):
    image = img.copy()
    for face_landmarks in landmark_results.multi_face_landmarks:
        if tesselation==True:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        
        if contours==True:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        
        if iris==True:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
    return image


