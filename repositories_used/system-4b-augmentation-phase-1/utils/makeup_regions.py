
def append_areas_and_deltas(areas_dict, deltas_dict, keys):
    areas = []
    deltas = []
    for k in keys:
        areas.append(areas_dict[k])
        deltas.append(deltas_dict[k])
    
    return areas,deltas

def complete_deltas_dict(areas_dict, deltas_dict):
    # print('Should be empty:',set(deltas_dict)-set(areas_dict))
    for key in areas_dict:
        if not key in deltas_dict.keys():
            # print(key)
            deltas_dict[key] = None
    
    return deltas_dict

## -------------------------------------------------------
## extended forehead landmarks
## -------------------------------------------------------

# forehead_landmarks = {478:67, 479:109, 480:10, 481:338, 482:297}
# forehead_region = [(69,67), (67,478),(478,479),(479,480),(480,481),(481,338),(338,69)]

forehead_landmarks = {478:67, 479:109, 480:10, 481:338, 482:297, 483:103, 484:332}
forehead_region = [(69,67), (67,478),(478,479),(479,480),(480,481),(481,338),(338,69)]


## -------------------------------------------------------
## regions EYES and LIPS
## -------------------------------------------------------



regions_dict = {}


##Eye liner
regions_dict['r_eyeliner'] = [(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (133,173),(173,157),(157,158),(158,159),(159,160),
                                (160,161),(161,246),(246,33)
]

regions_dict['l_eyeliner'] = [(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (362,398),(398,384),(384,385),(385,386),(386,387),
                               (387,388),(388,466),(466,263)
]                                

regions_dict['r_eye_line'] = [(156,225), (225,224), (224,223),(223,222), (222,221), (221,189)
]

regions_dict['l_eye_line'] = [(383,445), (445,444), (444,443), (443,442), (442,441), (441,413)

]

regions_dict['r_eye_shade_1'] =  [(156,225), (225,224), (224,223), (223,222), (222,221),(221,189),
                              (189,190), (190,56), (56,28), (28,27), (27,29), (29,30), (30,113), (113,124), 
                              (124,156)
]

regions_dict['l_eye_shade_1'] = [(383,445), (445,444), (444,443), (443,442), (442,441), (441,413), 
									(413,414),(414,286),(286,258),(258,257),(257,259),(259,260),(260,342),(342,353), 
                                    (353,383)
]

regions_dict['r_eye_shade_2'] = [(190,28), (28,27), (27,29), (29,30), (30,113), (113,130),
                                    (130,246), (246,161), (161,160), (160,159), (159,158), 
                                    (158, 157), (157,173), (173,190) 
] 

regions_dict['l_eye_shade_2'] = [(414,258),(258,257),(257,259),(259,260),(260,342),(342,359),
									(359,263),(263,466),(466,388),(388,387),(387,386),(386,385),
									(385,384),(384,398),(398,414)
]

regions_dict['lip_boundary'] = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),(314, 405), 
                                (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),(270, 269), 
                                (269, 267), (267, 0), (0, 37), (37, 39), (39, 40), (40, 185), (185, 61)
]

regions_dict['lip_upper'] = [(291, 409), (409, 270), (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), 
                             (39, 40), (40, 185), (185, 61), (61, 76), (76, 62), (62, 78), (78, 191), (191, 80),
                             (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308), (308, 291)
]

regions_dict['lip_lower'] = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), 
                             (375, 291), (291, 308), (308, 324), (324, 318), (318, 402), (402, 317), (317, 14), (14, 87), (87, 178), 
                             (178, 88), (88, 95), (95, 78), (78, 61)
]

regions_dict['full_eye'] = [(55,65), (65,52), (52,53), (53,46), (46,70), (70,139), (139,143), (143,35), 
                            (35,226), (226,33), (33,246), (246,161), (161,160), (160,159), (159,158), 
                            (158,157), (157,173), (173,133), (133,243), (243,244), (244,193), (193,55) 
]

regions_dict['r_eye_high_1'] = [(222, 52), (52, 53), (53, 224), (224, 223), (223, 222)]
regions_dict['l_eye_high_1'] = [(442,282), (282,283), (283,444), (444,443), (443,442)]

regions_dict['r_eye_high_2'] = [(173, 190), (190, 244), (244, 112), (112, 155), (155, 133), (133, 173)]
regions_dict['l_eye_high_2'] = [(398, 414), (414, 464), (464, 341), (341, 382), (382, 362), (362, 398)]


## -------------------------------------------------------
## deltas EYES and LIPS
## -------------------------------------------------------

deltas = {}

deltas['r_eyeliner'] = None
deltas['l_eyeliner'] = None

deltas['r_eye_line'] = {156: (0,-0.01),
                        225: (-0.003,-0.003)} 

deltas['l_eye_line'] = {383:(0,-0.01),
						445:(0.003,-0.003)
}

deltas['r_eye_shade_1'] = {156: (0,-0.01),
                            224: (0,0.003),
                            223: (0,0.003),
                            222: (0,0.004),
                            221: (0,0.005),
                            189: (0,0.005),
}
deltas['l_eye_shade_1'] = {383: (0,-0.01),
                            444: (0,0.003),
                            443: (0,0.003),
                            442: (0,0.004),
                            441: (0,0.005),
                            413: (0,0.005),
}



deltas['r_eye_shade_2'] = {113: (-0.01, 0),
                            130: (0, -0.01),
                            157: (0, -0.008),
                            158: (0, -0.009),
                            159: (0, -0.01),
                            160: (0, -0.012),
                            161: (0, -0.013),
                            173: (0, -0.01),
                            190: (0, -0.01),
                            246: (-0.01, -0.01)
}

deltas['l_eye_shade_2'] = {342: (0.01, 0),
                            359: (0, -0.01),
                            384: (0, -0.008),
                            385: (0, -0.009),
                            386: (0, -0.01),
                            387: (0, -0.012),
                            388: (0, -0.013),
                            398: (0, -0.01),
                            414: (0, -0.01),
                            466: (0.01, -0.01)
}

deltas['lip_boundary'] = None

deltas['lip_upper'] = {39: [0.001, 0.001], 
                        40: [0.001, 0.001], 
                        269: [-0.001, 0.001], 
                        270: [-0.001, 0.001]
}

deltas['lip_lower'] = None

deltas['full_eye'] = {33: (0, -0.005),
                        46: (0, 0.01),
                        52: (0, 0.01),
                        53: (0, 0.01),
                        55: (-0.01, 0.01),
                        65: (0, 0.01),
                        157: (0, -0.005),
                        158: (0, -0.005),
                        159: (0, -0.005),
                        160: (0, -0.005),
                        161: (0, -0.005),
                        246: (0, -0.005)
}

deltas['r_eye_high_1'] = {52: (0, 0.005), 
                          53: (0, 0.005)
}
deltas['l_eye_high_1'] = {282: (0, 0.005), 
                          283: (0, 0.005)
}

deltas['r_eye_high_2'] = {112: (0, 0.01)
}
deltas['l_eye_high_2'] = {341: (0, 0.01)
}


eye_all_regions_r = []
eye_all_deltas_r = []
eye_all_regions_r.append(regions_dict['r_eye_shade_1'])
eye_all_deltas_r.append(deltas['r_eye_shade_1'])
eye_all_regions_r.append(regions_dict['r_eye_shade_2'])
eye_all_deltas_r.append(deltas['r_eye_shade_2'])


eye_all_regions_l = []
eye_all_deltas_l = []
eye_all_regions_l.append(regions_dict['l_eye_shade_1'])
eye_all_deltas_l.append(deltas['l_eye_shade_1'])
eye_all_regions_l.append(regions_dict['l_eye_shade_2'])
eye_all_deltas_l.append(deltas['l_eye_shade_2'])

eye_high_regions_1 = []
eye_high_deltas_1 = []
eye_high_regions_1.append(regions_dict['r_eye_high_1'])
eye_high_deltas_1.append(deltas['r_eye_high_1'])
eye_high_regions_1.append(regions_dict['l_eye_high_1'])
eye_high_deltas_1.append(deltas['l_eye_high_1'])

eye_high_regions_2 = []
eye_high_deltas_2 = []
eye_high_regions_2.append(regions_dict['r_eye_high_2'])
eye_high_deltas_2.append(deltas['r_eye_high_2'])
eye_high_regions_2.append(regions_dict['l_eye_high_2'])
eye_high_deltas_2.append(deltas['l_eye_high_2'])

eye_line_regions_r = []
eye_line_deltas_r = []
eye_line_regions_r.append(regions_dict['r_eye_line'])
eye_line_deltas_r.append(deltas['r_eye_line'])

eye_line_regions_l = []
eye_line_deltas_l = []
eye_line_regions_l.append(regions_dict['l_eye_line'])
eye_line_deltas_l.append(deltas['l_eye_line'])


eyeliner_regions = []
eyeliner_deltas = []
eyeliner_regions.append(regions_dict['r_eyeliner'])
eyeliner_deltas.append(deltas['r_eyeliner'])
eyeliner_regions.append(regions_dict['l_eyeliner'])
eyeliner_deltas.append(deltas['l_eyeliner'])



## -------------------------------------------------------
## regions CONTOURS
## -------------------------------------------------------
contour_dict = {}

contour_dict['w_nose_forehead'] = [(4,51), (51,3), (3,196), (196,122), (122,193), (193,107), (107,66),
                                   (66,108), (108,10), (10,337), (337,296), (296,336), (336,417), (417,351),
                                   (351,419), (419,248), (248,281), (281,4)
]
 
contour_dict['b_forehead_1'] = [(21,54), (54,478), (478,479), (479,480), (480,10), (10,151), (151,107), 
                                (107,66), (66,105), (105,63), (63,70), (70,71), (71,21) 
]

contour_dict['b_forehead_2'] = [(480,481), (481,482), (482,284),(284,251), (251,301), (301,293), (293,334), (334,296),
                                (296,336), (336,151), (151,10), (10,480)
]

contour_dict['b_forehead_3'] = [(21,54), (54,478), (478,104), (104,105), (105,63), (63,70), (70,21)
] 

contour_dict['b_forehead_4'] = [(251,284), (284,482), (482,299), (299,296), (296,334), (334,293), (293,300),(300,251)
                                
]

contour_dict['b_forehead_5'] = [(21,54), (54,478), (478,479), (479,480), (480,481), (481,482), (482,332), (332,284), (284,251),
                                (251,300), (300,293), (293,334), (334,297), (297,338), (338,10), (10,109), (109,67), (67,104), (104,63),
                                (63,70), (70,21)                               
] 

contour_dict['b_nose_1'] = [(189,245), (245,188), (188,174), (174,236), (236,198), (198,209), (209,49), (49,48), (48,115), (115,220), (220,44),
                          (44,45), (45,51), (51,3), (3,196), (196,122), (122,193), (193,189)
]

contour_dict['b_nose_2'] = [(413,465), (465,412), (412,399), (399,420), (420,360), (360,279), (279,278), (278,344),(344, 440), (440,274), 
                            (274,275), (275,281), (281,248), (248,419), (419,351), (351,417), (417,413)
]

contour_dict['b_cheek_1'] = [(227,123), (123,50), (50,205), (205,207), (207,187), (187,147), (147,177), (177,227)
]

contour_dict['b_cheek_2'] = [(447,352), (352,280), (280,425), (425,427), (427,411), (411,376), (376,447)
]

contour_dict['w_cheek_1'] = [(35,31), (31,228), (228,229), (229,230), (230,231), (231,232), (232,233),
                             (233,128), (128,47), (47,142), (142,203), (203,50), (50,143), (143,35)
                             
]
contour_dict['w_cheek_2'] = [(265,261), (261,448), (448,449), (449,450), (450,451), (451,452), (452,453),
                             (453,357), (357,277), (277,371), (371,423), (423,280), (280,372), (372, 265)
                             
]


contour_dict['b_chin_1'] = [(172,210), (210,211), (211,32), (32,171), (171,148), 
                            (148,176), (176,149), (149,150), (150,136), (136,172) 
]

contour_dict['b_chin_2'] = [(397,430), (430,431), (431,262), (262,396), (396,377),
                            (377,400), (400,378), (378,379), (379,365), (365,397)
                            
]

contour_dict['w_chin'] = [(201,200), (200,421), (421,199), (199,201)
                          
]

## -------------------------------------------------------
## deltas CONTOURS
## -------------------------------------------------------

deltas_contour = {}
deltas_contour['w_nose_forehead'] = {
                                    108: (-0.02,-0.02),
                                    # 10:  (0,0.003),
                                    337: (0.02,-0.02)
                                                                 
}

deltas_contour['b_forehead_1'] = None
deltas_contour['b_forehead_2'] = None
deltas_contour['b_forehead_3'] = None
deltas_contour['b_forehead_4'] = None
deltas_contour['b_forehead_5'] = {297: (-0.005,0)}


deltas_contour['b_nose_1'] = None
deltas_contour['b_nose_2'] = None
deltas_contour['b_cheek_1'] = {
                                50:  (0,0.03),
                                205: (-0.01,0.02),
                                207: (0,0.005),
                                187: (0,0.04),
                                147: (0,0.04)   
}
deltas_contour['b_cheek_2'] = {
                                280: (0,0.03),
                                425: (0.01,0.02), 
                                427: (0,0.005),
                                411: (0,0.04),
                                376: (0,0.04)

}
deltas_contour['w_cheek_1'] = {50: (0.005, -0.005)

}
deltas_contour['w_cheek_2'] = {280: (-0.005, -0.005)

}

deltas_contour['b_chin_1'] = None
deltas_contour['b_chin_2'] = None

deltas_contour['w_chin'] = None


## -------------------------------------------------------
## combined contour regions
## -------------------------------------------------------

################################################################
# cheeks_dark = []
# cheeks_dark.append(contour_dict['b_cheek_1'])
# cheeks_dark.append(contour_dict['b_cheek_2'])

# cheeks_dark_delta = []
# cheeks_dark_delta.append(deltas_contour['b_cheek_1'])
# cheeks_dark_delta.append(deltas_contour['b_cheek_2'])

# cheeks_light = []
# cheeks_light.append(contour_dict['w_cheek_1'])
# cheeks_light.append(contour_dict['w_cheek_2'])

# cheeks_light_delta = []
# cheeks_light_delta.append(deltas_contour['w_cheek_1'])
# cheeks_light_delta.append(deltas_contour['w_cheek_2'])
# ##################################################################
# chin_dark = []
# chin_dark.append(contour_dict['b_chin_1'])
# chin_dark.append(contour_dict['b_chin_2'])

# chin_dark_delta = []
# chin_dark_delta.append(deltas_contour['b_chin_1'])
# chin_dark_delta.append(deltas_contour['b_chin_2'])

# chin_light = []
# chin_light.append(contour_dict['w_chin'])

# chin_light_delta = []
# chin_light_delta.append(deltas_contour['w_chin'])

# ###################################################################

# nose_dark = []
# nose_dark.append(contour_dict['b_nose_1'])
# nose_dark.append(contour_dict['b_nose_2'])

# nose_dark_delta = []
# nose_dark_delta.append(deltas_contour['b_nose_1'])
# nose_dark_delta.append(deltas_contour['b_nose_2'])

# ####################################################################
# forehead_dark = []
# forehead_dark.append(contour_dict['b_forehead_3'])
# forehead_dark.append(contour_dict['b_forehead_4'])
# forehead_dark.append(contour_dict['b_forehead_5'])

# forehead_dark_delta = []
# forehead_dark_delta.append(deltas_contour['b_forehead_3'])
# forehead_dark_delta.append(deltas_contour['b_forehead_4'])
# forehead_dark_delta.append(deltas_contour['b_forehead_5'])

# forehead_light = []
# forehead_light.append(contour_dict['w_nose_forehead'])

# forehead_light_delta = []
# forehead_light_delta.append(deltas_contour['w_nose_forehead'])

####################################################################



## -------------------------------------------------------
## Template 1 regions
## -------------------------------------------------------

contour_dict_temp1 = {}

contour_dict_temp1['w_forehead'] = [(9,108),(108,10),(10,337),(337,9)

]

contour_dict_temp1['w_nose'] = [(1,45),(45,51),(51,3),(3,196),(196,122),(122,193),(193,8),
                          (8,417), (417,351),(351,419),(419,248),(248,281),(281,275),(275,1)
    
]
contour_dict_temp1['w_cheek_r'] = [(162,139),(139,35),(35,31),(31,228),(228,229),(229,119),(119,101),
                             (101,117),(117,111),(111,143),(143,162)
    
]
contour_dict_temp1['w_cheek_l'] = [(330,348),(348,449),(449,448),(448,261),(261,265),(265,368),(368,389),
                             (389,372),(372,340),(340,346),(346,330)
]
contour_dict_temp1['w_eye_r'] = [(70,53),(53,46),(46,70)
    
]
contour_dict_temp1['w_eye_l'] = [(283,300),(300,276),(276,283)
    
]
contour_dict_temp1['chin'] = [(17,201),(201,200),(200,421),(421,17)

]

contour_dict_temp1['b_forehead_r'] = [(21,54),(54,483),(483,478),(478,67),
                                      (67,103),(103,68),(68,21)
]
contour_dict_temp1['b_forehead_l'] = [(301,284),(284,484),(484,482),(482,297),
                                      (297,333),(333,298),(298,301)]

contour_dict_temp1['b_nose_r'] = [(193,189),(189,244),(244,245),(245,188),(188,174),(174,236),
                                  (236,134),(134,45),(45,51),(51,3),(3,196),(196,122),(122,193)
]
contour_dict_temp1['b_nose_l'] = [(417,413),(413,465),(465,412),(412,399),(399,456),(456,363),(363,275),
                                  (275,281),(281,248),(248,419),(419,351),(351,417)
]

contour_dict_temp1['b_cheek_r'] = [(123,207),(207,212),(212,147),(147,123)
]
contour_dict_temp1['b_cheek_l'] = [(432,427),(427,352),(352,376),(376,432)
] 

deltas_temp1 = {}

deltas_temp1['w_eye_r'] = {70: (0.001,0.001),
                           46: (0.001,0.001)
    
}
deltas_temp1['w_eye_l'] = {300: (-0.001,0.005),
                           276: (-0.001,0.003)
    
}

deltas_temp1['b_forehead_r'] = {54:  (-0.025,0),
                                483: (-0.05,0.01),
                                478: (-0.06,0.03),
                                67:  (-0.03,-0.03),
                                68:  (-0.03,0)
}


deltas_temp1['b_forehead_l'] = {301: (0,0.001),
                                333: (0.01,0),
                                482: (0.03,0.03),
                                284: (-0.003,0),
                                484: (0.02,0.001)
}


## -------------------------------------------------------
## Template 2 regions
## -------------------------------------------------------

contour_dict_temp2 = {}
contour_dict_temp2['w_forehead'] = [(109,108),(108,9),(9,337),(337,338),(338,151),(151,109)
]

contour_dict_temp2['w_nose'] = [(8,168),(168,6),(6,197),(197,195),(195,5),(5,4)   
]

contour_dict_temp2['w_cheek_r_1'] = [(118,143),(143,34),(34,116),(116,50),(50,118)
]

contour_dict_temp2['w_cheek_r_2'] = [(119,101),(101,100),(100,121),(121,232),(232,119)
]

contour_dict_temp2['w_cheek_l_1'] = [(264,372),(372,347),(347,280),(280,345),(345,264)    
]
contour_dict_temp2['w_cheek_l_2'] = [(452,350),(350,329),(329,330),(330,348),(348,452)
]
contour_dict_temp2['w_chin'] = [(18,201),(201,199),(199,421),(421,18)
]

contour_dict_temp2['b_nose_r'] = [(193,122),(122,196),(196,3),(3,51)
]
contour_dict_temp2['b_nose_l'] = [(417,351),(351,419),(419,248),(248,281)
]
contour_dict_temp2['b_cheek_r'] = [(234,187),(187,212),(212,192),(192,213),(213,234)
] 
contour_dict_temp2['b_jaw_r'] = [(176,140),(140,170),(170,169),(169,135),(135,138),
                                     (138,172),(172,136),(136,150),(150,149),(149,176)
]

contour_dict_temp2['b_cheek_l'] = [(454,411),(411,432),(432,416),(416,433),(433,454)
]
contour_dict_temp2['b_jaw_l'] = [(400,369),(369,395),(395,394),(394,364),(364,367),
                                     (367,397),(397,365),(365,379),(379,378),(378,400)
]

contour_dict_temp2['b_forehead_r'] = [(162,21),(21,54),(54,483),(483,478),(478,103),(103,68),
                                    (68,71),(71,162)
]
contour_dict_temp2['b_forehead_l'] = [(389,251),(251,284),(284,484),(484,482),(482,332),(332,298),
                                    (298,301),(301,389)

]

deltas_temp2 = {}
deltas_temp2['w_cheek_r_1'] = {50: (0.01,-0.03)
    
}
deltas_temp2['w_cheek_r_2'] = {119: (-0.025,0)
    
}
deltas_temp2['w_cheek_l_1'] = {280: (-0.01,-0.03)
    
}
deltas_temp2['w_cheek_l_2'] = {348: (0.025,0)
    
}
deltas_temp2['w_chin'] = {199: (0,-0.01)
    
}
deltas_temp2['b_jaw_r'] = {169: (0.003,-0.003),
                           135: (0.003,-0.003)
}
deltas_temp2['b_jaw_l'] = {394: (-0.003,-0.003),
                           395: (-0.003,-0.003)
}


## -------------------------------------------------------
## Template 2 regions
## -------------------------------------------------------

contour_dict_temp3 = {}

contour_dict_temp3['w_forehead'] = [(10,151),(151,9)
]
contour_dict_temp3['w_nose'] = [(8,168),(168,6),(6,197),(197,195),(195,5),(5,4)
]
contour_dict_temp3['w_brow_l'] = [(300,293),(293,334),(334,296)
]
contour_dict_temp3['w_brow_r'] = [(70,63),(63,105),(105,66)
]

contour_dict_temp3['w_cheek_l'] = [(347,330),(330,371)
]
contour_dict_temp3['w_cheek_r'] = [(118,101),(101,142)
]

contour_dict_temp3['w_jaw_l'] = [(433,367),(367,364)
]
contour_dict_temp3['w_jaw_r'] = [(213,138),(138,135)
]
contour_dict_temp3['w_chin'] = [(199,175)
]

contour_dict_temp3['b_brow_l'] = [(300,293),(293,299),(299,337)
]
contour_dict_temp3['b_brow_r'] = [(70,63),(63,69),(69,108)
]

contour_dict_temp3['b_forehead_l'] = [(300,333),(333,297),(297,338)
]
contour_dict_temp3['b_forehead_r'] = [(70,104),(104,67),(67,109)
]
contour_dict_temp3['b_cheek_l_1'] = [(352,425)
]
contour_dict_temp3['b_cheek_l_2'] = [(352,411),(411,427)
]
contour_dict_temp3['b_cheek_r_1'] = [(123,205)
]
contour_dict_temp3['b_cheek_r_2'] = [(123,187),(187,207)
]





deltas_temp3 = {}
deltas_temp3['w_brow_l'] = {300: (0,-0.01),
                          293: (0,-0.01),
                          334: (0,-0.01),
                          296: (0,-0.01)
}
deltas_temp3['w_brow_r'] = {70: (0,-0.01),
                          63: (0,-0.01),
                          105: (0,-0.01),
                          66: (0,-0.01)
}

deltas_temp3['b_brow_l'] = {300: (0,-0.01),
                          293: (0,-0.01),

}
deltas_temp3['b_brow_r'] = {70: (0,-0.01),
                          63: (0,-0.01),

}

deltas_temp3['b_forehead_l'] = {300: (0,-0.01)
}
deltas_temp3['b_forehead_r'] = {70: (0,-0.01)
}

## -------------------------------------------------------
## Complete and combine Template  regions
## -------------------------------------------------------

deltas_temp1 = complete_deltas_dict(contour_dict_temp1,deltas_temp1)
deltas_temp2 = complete_deltas_dict(contour_dict_temp2,deltas_temp2)
deltas_temp3 = complete_deltas_dict(contour_dict_temp3,deltas_temp3)
