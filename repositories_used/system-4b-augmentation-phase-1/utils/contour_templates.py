from utils.makeup_regions import *
from utils.feature_extraction_utils import apply_smooth_color

def put_contour_template_6(img, img_landmarks, img_parsed, args):

    sel_keys = ['b_cheek_1','b_cheek_2']
    cheeks_dark,cheeks_dark_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)
    
    sel_keys = ['w_cheek_1','w_cheek_2']
    cheeks_light,cheeks_light_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)
    
    
    sel_keys = ['b_chin_1','b_chin_2']
    chin_dark, chin_dark_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)
    
    sel_keys = ['w_chin']
    chin_light, chin_light_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)
    
    
    sel_keys = ['b_nose_1','b_nose_2']
    nose_dark,nose_dark_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)


    
    sel_keys = ['b_forehead_3','b_forehead_4','b_forehead_5']
    forehead_dark,forehead_dark_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)
    
    sel_keys = ['w_nose_forehead']
    forehead_light, forehead_light_delta = append_areas_and_deltas(contour_dict,deltas_contour,sel_keys)
    
    ##--------------------------------------
    ## Cheeks
    ##--------------------------------------

    makeup_img = img.copy()
    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_dark, 
                        cheeks_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_cheek_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=2,
                        blur_kernel=(101,101),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=10,
                        seam_mask_iters=20
                        )

    
    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_light, 
                        cheeks_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_cheek_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=20,
                        int_blur=1,
                        blur_iters=2,
                        blur_kernel=(71,71),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )

    

    ##--------------------------------------
    ## Chin
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        chin_dark, 
                        chin_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_chin_dark,
                        close_flag=True,
                        bezier_flag='smooth_poly',
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(101,101),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=30
                        )


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        chin_light, 
                        chin_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_chin_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=40,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(71,71),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )


    ##--------------------------------------
    ## Nose
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        nose_dark, 
                        nose_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_nose_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=7,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )


    ##--------------------------------------
    ## Forehead
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_dark[0:2], 
                        forehead_dark_delta[0:2], 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_forehead_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=10,
                        int_blur=1,
                        blur_iters=10,
                        blur_kernel=(31,31),
                        seam_flag=True,
                        img_parsed=img_parsed,
                        seam_mask_iters=10
                        )

    

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        [forehead_dark[2]], 
                        [forehead_dark_delta[2]], 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_forehead_dark,
                        close_flag=True,
                        bezier_flag='smooth_poly',
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(201,201),
                        seam_flag=True,
                        img_parsed=img_parsed,
                        seam_mask_iters=20,
                        dilate_shade=10,
                        )


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_light, 
                        forehead_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_forehead_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=7,
                        int_blur=1,
                        blur_iters=10,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=50
                        )
    return makeup_img


def put_contour_template_1(img, img_landmarks, img_parsed, args):
    ## cheek regions
    sel_keys = ['b_cheek_r', 'b_cheek_l']
    cheeks_dark,cheeks_dark_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    sel_keys = ['w_cheek_r', 'w_cheek_l']
    cheeks_light,cheeks_light_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    ## chin regions
    sel_keys = ['chin']
    chin_light,chin_light_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    ## nose
    sel_keys = ['b_nose_r', 'b_nose_l']
    nose_dark,nose_dark_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    sel_keys = ['w_nose']
    nose_light,nose_light_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    ## forehead
    sel_keys = ['b_forehead_r','b_forehead_l']
    forehead_dark,forehead_dark_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    sel_keys = ['w_forehead']
    forehead_light,forehead_light_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)

    ## eye highlighter
    sel_keys = [ 'w_eye_r', 'w_eye_l']
    eye_light,eye_light_delta = append_areas_and_deltas(contour_dict_temp1,deltas_temp1,sel_keys)
    
    
    ##--------------------------------------
    ## Cheeks
    ##--------------------------------------

    makeup_img = img.copy()
    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_dark, 
                        cheeks_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_cheek_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=3,
                        int_blur=1,
                        blur_iters=5,
                        blur_kernel=(61,61),#(101,101),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=1,
                        seam_mask_iters=20
                        )


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_light, 
                        cheeks_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_cheek_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=5,
                        int_blur=1,
                        blur_iters=5,
                        blur_kernel=(61,61),#(71,71),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=1,
                        seam_mask_iters=20
                        )



    ##--------------------------------------
    ## Chin
    ##--------------------------------------


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        chin_light, 
                        chin_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_chin_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=40,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )


    ##--------------------------------------
    ## Nose
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        nose_dark, 
                        nose_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_nose_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=4
                        )

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        nose_light, 
                        nose_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_nose_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(41,41),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=4
                        )


    ##--------------------------------------
    ## Forehead
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_dark, 
                        forehead_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_forehead_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=3,
                        int_blur=1,
                        blur_iters=4,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=img_parsed,
                        seam_mask_iters=20
                        )



    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_light, 
                        forehead_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_forehead_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=5,
                        int_blur=1,
                        blur_iters=10,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=5,
                        seam_mask_iters=20
                        )

    ##--------------------------------------
    ## eye highlighter
    ##--------------------------------------


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        eye_light, 
                        eye_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_cheek_light,#args.int_eye_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=20,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(21,21),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )

    return makeup_img

def put_contour_template_2(img, img_landmarks, img_parsed, args):

    ## cheek regions
    sel_keys = ['b_cheek_r', 'b_jaw_r', 'b_cheek_l', 'b_jaw_l']
    cheeks_dark,cheeks_dark_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)

    sel_keys = ['w_cheek_r_1', 'w_cheek_r_2','w_cheek_l_1', 'w_cheek_l_2']
    cheeks_light,cheeks_light_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)

    ## chin regions
    sel_keys = ['w_chin']
    chin_light,chin_light_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)

    ## nose
    sel_keys = ['b_nose_r', 'b_nose_l']
    nose_dark,nose_dark_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)

    sel_keys = ['w_nose']
    nose_light,nose_light_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)

    ## forehead
    sel_keys = ['b_forehead_r','b_forehead_l']
    forehead_dark,forehead_dark_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)

    sel_keys = ['w_forehead']
    forehead_light,forehead_light_delta = append_areas_and_deltas(contour_dict_temp2,deltas_temp2,sel_keys)


    ##--------------------------------------
    ## Cheeks
    ##--------------------------------------

    makeup_img = img.copy()
    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_dark, 
                        cheeks_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_cheek_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=3,
                        int_blur=1,
                        blur_iters=5,
                        blur_kernel=(61,61),#(101,101),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=1,
                        seam_mask_iters=20
                        )


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_light, 
                        cheeks_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_cheek_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=5,
                        int_blur=1,
                        blur_iters=5,
                        blur_kernel=(61,61),#(71,71),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=1,
                        seam_mask_iters=20
                        )



    # ##--------------------------------------
    # ## Chin
    # ##--------------------------------------


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        chin_light, 
                        chin_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_chin_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=40,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )


    # ##--------------------------------------
    # ## Nose
    # ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        nose_dark, 
                        nose_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_nose_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(51,51),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=4
                        )

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        nose_light, 
                        nose_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_nose_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(41,41),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=4
                        )


    ##--------------------------------------
    ## Forehead
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_dark, 
                        forehead_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_forehead_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=3,
                        int_blur=1,
                        blur_iters=4,
                        blur_kernel=(61,61),
                        seam_flag=True,
                        img_parsed=img_parsed,
                        seam_mask_iters=20
                        )



    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_light, 
                        forehead_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_forehead_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=5,
                        int_blur=1,
                        blur_iters=10,
                        blur_kernel=(41,41),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=5,
                        seam_mask_iters=20
                        )

    return makeup_img

def put_contour_template_3(img, img_landmarks, img_parsed, args):

    ## cheek regions
    sel_keys = ['b_cheek_l_1', 'b_cheek_l_2', 'b_cheek_r_1', 'b_cheek_r_2']
    cheeks_dark,cheeks_dark_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)

    sel_keys = ['w_cheek_l', 'w_cheek_r', 'w_jaw_l', 'w_jaw_r']
    cheeks_light,cheeks_light_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)

    ## chin regions
    sel_keys = ['w_chin']
    chin_light,chin_light_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)

    ## nose
    # sel_keys = ['b_nose_r', 'b_nose_l']
    # nose_dark,nose_dark_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)

    sel_keys = ['w_nose']
    nose_light,nose_light_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)

    ## forehead
    sel_keys = ['b_forehead_r','b_forehead_l','b_brow_l', 'b_brow_r']
    forehead_dark,forehead_dark_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)

    sel_keys = ['w_forehead','w_brow_l', 'w_brow_r']
    forehead_light,forehead_light_delta = append_areas_and_deltas(contour_dict_temp3,deltas_temp3,sel_keys)



    ##--------------------------------------
    ## Cheeks
    ##--------------------------------------

    makeup_img = img.copy()
    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_dark, 
                        cheeks_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_cheek_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=1,
                        int_blur=1,
                        blur_iters=2,
                        blur_kernel=(41,41),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=1,
                        seam_mask_iters=20
                        )


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        cheeks_light, 
                        cheeks_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_cheek_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=3,
                        int_blur=1,
                        blur_iters=2,
                        blur_kernel=(41,41),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=3,
                        seam_mask_iters=1
                        )



    ##--------------------------------------
    ## Chin
    ##--------------------------------------


    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        chin_light, 
                        chin_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_chin_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=10,
                        int_blur=1,
                        blur_iters=4,
                        blur_kernel=(41,41),
                        seam_flag=True,
                        img_parsed=None,
                        dilate_shade=3,
                        seam_mask_iters=20
                        )


    # ##--------------------------------------
    # ## Nose
    # ##--------------------------------------

    # makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
    #                     nose_dark, 
    #                     nose_dark_delta, 
    #                     overlay_color=color_dark, 
    #                     dest_color=color_dark,
    #                     overlay_intensity=int_nose_dark,
    #                     close_flag=True,
    #                     bezier_flag=False,
    #                     shade=None,
    #                     line_thickness=5,
    #                     fill_mask_flag=True,
    #                     shade_type='simple_grad',
    #                     dilate_iters=7,
    #                     int_blur=1,
    #                     blur_iters=1,
    #                     blur_kernel=(51,51),
    #                     seam_flag=True,
    #                     img_parsed=None,
    #                     seam_mask_iters=1
    #                     )

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        nose_light, 
                        nose_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_nose_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=7,
                        int_blur=1,
                        blur_iters=1,
                        blur_kernel=(31,31),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=1
                        )


    ##--------------------------------------
    ## Forehead
    ##--------------------------------------

    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_dark, 
                        forehead_dark_delta, 
                        overlay_color=args.color_dark, 
                        dest_color=args.color_dark,
                        overlay_intensity=args.int_forehead_dark,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=10,
                        int_blur=1,
                        blur_iters=4,
                        blur_kernel=(71,71),
                        seam_flag=True,
                        img_parsed=img_parsed,
                        seam_mask_iters=20
                        )



    makeup_img,f_mask,d_mask = apply_smooth_color(makeup_img, img_landmarks, 
                        forehead_light, 
                        forehead_light_delta, 
                        overlay_color=args.color_light, 
                        dest_color=args.color_light,
                        overlay_intensity=args.int_forehead_light,
                        close_flag=True,
                        bezier_flag=False,
                        shade=None,
                        line_thickness=5,
                        fill_mask_flag=True,
                        shade_type='simple_grad',
                        dilate_iters=10,
                        int_blur=1,
                        blur_iters=4,
                        blur_kernel=(61,61),
                        seam_flag=True,
                        img_parsed=None,
                        seam_mask_iters=20
                        )
    
    return makeup_img