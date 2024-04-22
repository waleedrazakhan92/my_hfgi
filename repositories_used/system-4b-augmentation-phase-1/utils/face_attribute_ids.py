import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


eye_l = frozenset([(229,230),(230,231),(231,232),(232,233),(232,244),
                (118,119),(119,120),(120,121),(121,128),(128,245),
                (50,101),(101,100),(100,47),(47,114),(114,188),(188,122),
                (207,205),(205,36),(36,142),(142,126),(126,217),(217,174),(174,196)
                ])

eye_r = frozenset([(464,453),(453,452),(452,451),(451,450),(450,449),(449,448),
                (465,357),(357,350),(350,349),(349,348),(348,347),(347,346),
                (351,412),(412,343),(343,277),(277,329),(329,330),(330,280),
                (419,399),(399,437),(437,355),(355,371),(371,266),(266,425),(425,427)
                ])

eye_contours = frozenset().union(*[
    eye_l,eye_r
])


nose_bridge_landmarks =  frozenset([(221,441),(441,343),(343,114),(114,221)
])

nose_mask_landmarks = frozenset([(8,193),(193,245),(245,188),(188,174),(174,236),(236,198),(198,209),
                        (209,49),(49,102),(102,64),(64,240),(240,59),(59,166),(166,218),
                        (218,239),(239,238),(238,242),(242,141),(141,94),(94,370),
                        (370,462),(462,458),(458,459),(459,438),(438,392),(392,289),(289,460),
                        (460,294),(294,331),(331,279),(279,429),(429,420),(420,456),(456,399),
                        (399,412),(412,465),(465,417),(417,8)
                        
])

idx_dict = {'eye_nasion_right': frozenset([(8,468)]),
'eye_nasion_left': frozenset([(8,473)]),
'inter_eye_width_right': frozenset([(33,133)]),
'inter_eye_width_left': frozenset([(263,362)]),
'nose_height': frozenset([(8,195)]),
'inter_tragi': frozenset([(234,454)]), #(234,345)
'low_jaw_wid': frozenset([(172,397)]),
'alae_breadth': frozenset([(102,344)]),
'mental_fold': frozenset([(17,18)]),
'alea_mid_face_height': frozenset([(8,0)]),
'chin': frozenset([(18,152)]),
'cupis_len': frozenset([(2,0)]),
'face_height': frozenset([(152,480)]),
'face_height_mp': frozenset([(152,10)]),
'forehead_width': frozenset([(103,332)]),
'cheek_bones_width': frozenset([(111,340)])
}

idx_dict_color = {'eye_nasion_right': [255,0,0],
'eye_nasion_left': [255,0,0],
'inter_eye_width_right': [255,255,0],
'inter_eye_width_left': [255,255,0],
'nose_height': [155,38,182],
'inter_tragi': [255,0,0],
'low_jaw_wid': [255,0,0],
'alae_breadth': [155,38,182],
'mental_fold': [255,255,0],
'alea_mid_face_height': [255,0,0],
'chin': [155,38,182],
'cupis_len': [255,255,0],
'face_height':[1,1,1],
'face_height_mp':[128,128,128],
'forehead_width': [0,255,0],
'cheek_bones_width':[0,255,255]
}

seg_mask_dict = {'left_eye': mp_face_mesh.FACEMESH_LEFT_EYE,
                 'right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE,
                 #'left_iris':mp_face_mesh.FACEMESH_LEFT_IRIS,
                 #'right_iris':mp_face_mesh.FACEMESH_RIGHT_IRIS,
                 'lips':mp_face_mesh.FACEMESH_LIPS,
                 'nose':nose_mask_landmarks
                 }

seg_mask_color = {'left_eye': [255,255,0],
                 'right_eye': [255,255,0],
                 #'left_iris':[255,0,0],
                 #'right_iris':[255,0,0],
                 'lips':[75,0,130],
                 'nose':[255,255,0]
                 }

