import numpy as np
import cv2
from utils.face_attribute_ids import seg_mask_dict,seg_mask_color
from utils.drawing_spec_utils import mp_drawing, drawing_spec
from scipy import ndimage
# from google.colab.patches import cv2_imshow

from utils.face_parsing_utils import part_color_dict


def make_mask(img, img_landmarks, connections):
    mask = np.zeros(np.shape(img), np.uint8)
    # drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3, color=[255,255,255])
    mp_drawing.draw_landmarks(
                image=mask,
                landmark_list=img_landmarks.multi_face_landmarks[0],
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
    
    return mask

def fill_mask(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray,[c], 0, (255,255,255), -1)
    
    return gray

def crop_part(img, mask):
    masked = cv2.bitwise_and(img, img, mask=mask)
    
    return masked

def get_cropped_part(img, img_landmarks, connections):
    mask = make_mask(img, img_landmarks, connections)
    filled_mask = fill_mask(mask)
    cropped_part = crop_part(img, filled_mask)

    return cropped_part

def crop_part_overlay(img, mask, overlay_color, overlay_intensity):
    color_img = np.zeros(img.shape, img.dtype)
    color_img[:,:,:] = overlay_color
    color_mask = cv2.bitwise_and(color_img, color_img, mask=mask)
    # masked = cv2.bitwise_and(img, img, mask=mask)
    masked = cv2.addWeighted(color_mask, overlay_intensity, img, 1, 0, img)    
    return masked

def get_cropped_part_overlay(img, img_landmarks, connections, overlay_color=[255,255,255], overlay_intensity=1):
    mask = make_mask(img, img_landmarks, connections)
    filled_mask = fill_mask(mask)
    cropped_part = crop_part_overlay(img, filled_mask, overlay_color, overlay_intensity)
    return cropped_part

def make_segmentation_overlay(img, results, overlay_intensity=0.3):
    seg_img = img.copy()
    for key in seg_mask_dict.keys():
        seg_img = get_cropped_part_overlay(seg_img, results, seg_mask_dict[key], 
                                overlay_color=seg_mask_color[key], overlay_intensity=overlay_intensity)
    
    return seg_img

## ------------------------------------
## for delta landmarks
## ------------------------------------

def make_mask_delta(img, img_landmarks, connections, landmark_deltas, close_flag=True, bezier_flag=False, fill_mask_flag=True, line_thickness=2,s=3.0,k=2,
                    img_parsed=None):
    mask = np.zeros(np.shape(img), np.uint8)
    # drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3, color=[255,255,255])
    drawing_spec.thickness=line_thickness
    mp_drawing.draw_landmarks_delta(
                image=mask,
                landmark_list=img_landmarks.multi_face_landmarks[0],
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec,
                landmark_deltas=landmark_deltas,
                        close_flag=close_flag,
                        bezier_flag=bezier_flag,
                        s=s,
                        k=k,
                        img_parsed=img_parsed)
    
    
    if fill_mask_flag==True:
        mask = fill_mask(mask)
        ###mask = cv2.merge((mask, mask, mask))
        return mask
    else:
        mask = mask[:,:,0]
        return mask
        
def get_cropped_part_delta(img, img_landmarks, connections, landmark_deltas, close_flag=True, bezier_flag=False, line_thickness=2, img_parsed=None):
    filled_mask = make_mask_delta(img, img_landmarks, connections, landmark_deltas, close_flag=close_flag, bezier_flag=bezier_flag,
    line_thickness=line_thickness, fill_mask_flag=True, img_parsed=img_parsed)

    cropped_part = crop_part(img, filled_mask)

    return cropped_part
    
def get_cropped_part_overlay_delta(img, img_landmarks, connections, landmark_deltas, overlay_color=[255,255,255], overlay_intensity=1, 
            close_flag=True, bezier_flag=False, shade=None, shade_strength=0.6, line_thickness=2):
                
    filled_mask = make_mask_delta(img, img_landmarks, connections, landmark_deltas, close_flag=close_flag, bezier_flag=bezier_flag,
    line_thickness=line_thickness, fill_mask_flag=True)
    
    cropped_part = crop_part_overlay_shade(img, filled_mask, overlay_color, overlay_intensity, shade=shade, shade_strength=shade_strength)
    return cropped_part

def make_mask_delta_combined(img, img_landmarks, connections, landmark_deltas, close_flag=True, bezier_flag=False, line_thickness=2,s=3.0,k=2,
                            fill_mask_flag=True, img_parsed=None):
    combined_mask = np.zeros(np.shape(img)[0:2], np.uint8)
    
    if landmark_deltas!=None:
        for i in range(0,len(connections)):            
            mask = make_mask_delta(img, img_landmarks, connections[i], landmark_deltas[i], close_flag=close_flag, bezier_flag=bezier_flag,
            line_thickness=line_thickness,s=s,k=k,fill_mask_flag=fill_mask_flag,img_parsed=img_parsed)

            combined_mask = cv2.bitwise_or(combined_mask, mask)
    else:
        for i in range(0,len(connections)):            
            mask = make_mask_delta(img, img_landmarks, connections[i], None, close_flag=close_flag, bezier_flag=bezier_flag,
            line_thickness=line_thickness,s=s,k=k,img_parsed=img_parsed)

            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask

def get_cropped_part_overlay_delta_3d(img, img_landmarks, connections, landmark_deltas, overlay_color,dest_color, overlay_intensity=1, 
            close_flag=True, bezier_flag=False, shade=None, line_thickness=2,fill_mask_flag=True,s=3.0,k=2,shade_type='simple_grad', return_mask=False,
            img_parsed=None, dilate_shade=0):
    
    combined_mask = make_mask_delta_combined(img, img_landmarks, connections, landmark_deltas, close_flag=close_flag, bezier_flag=bezier_flag,
            line_thickness=line_thickness, fill_mask_flag=fill_mask_flag, s=s,k=k,img_parsed=img_parsed)
    
    if dilate_shade>0:
        kernel = np.ones((5,5), np.uint8)
        combined_mask_dilated = cv2.dilate(combined_mask, kernel, iterations=dilate_shade)
    elif dilate_shade<0:
        kernel = np.ones((5,5), np.uint8)
        combined_mask_dilated = cv2.erode(combined_mask, kernel, iterations=abs(dilate_shade))
    else:
        combined_mask_dilated = combined_mask
        
    # cropped_part = crop_part_overlay_shade(img, filled_mask, overlay_color, overlay_intensity, shade=shade, shade_strength=shade_strength)
    cropped_part = crop_part_overlay_shade_3d(img, combined_mask_dilated, overlay_color, dest_color, overlay_intensity, shade=shade,shade_type=shade_type)
    
    if return_mask==True:
        return cropped_part, combined_mask
        
    return cropped_part


def make_segmentation_overlay_delta(img, results, landmark_deltas, overlay_intensity=0.3, close_flag=True, bezier_flag=False, shade=None,line_thickness=2):
    seg_img = img.copy()
    for key in seg_mask_dict.keys():
        # seg_img = get_cropped_part_overlay_delta(seg_img, results, landmark_deltas, seg_mask_dict[key], 
        #                         overlay_color=seg_mask_color[key], overlay_intensity=overlay_intensity,
        #                         close_flag=close_flag, bezier_flag=bezier_flag, shade=shade, shade_strength=shade_strength,
        #                         line_thickness=line_thickness)
        seg_img = get_cropped_part_overlay_delta(seg_img, results, landmark_deltas, seg_mask_dict[key], 
                                overlay_color=seg_mask_color[key], overlay_intensity=overlay_intensity,
                                close_flag=close_flag, bezier_flag=bezier_flag, shade=shade, shade_strength=shade_strength,
                                line_thickness=line_thickness)
    
    return seg_img


## shaded part v1
def crop_part_overlay_shade(img, mask, overlay_color, overlay_intensity, shade=None,shade_strength=0.6):

    if shade!=None:
        assert shade=='left' or shade=='right' or shade=='up' or shade=='down'

        shaded_mask = create_shaded_image(img,mask,overlay_color,shade_strength=shade_strength,shade_dir=shade)
        color_mask = cv2.bitwise_and(shaded_mask, shaded_mask, mask=mask)
    else:
        color_img = np.zeros(img.shape, img.dtype)
        color_img[:,:,:] = overlay_color
        color_mask = cv2.bitwise_and(color_img, color_img, mask=mask)

    # masked = cv2.bitwise_and(img, img, mask=mask)
    masked = cv2.addWeighted(color_mask, overlay_intensity, img, 1, 0, img)  
    return masked
    
def create_shaded_image(img,mask,overlay_color,shade_strength=0.8,shade_dir='left'):
    indices = np.where(mask==255)
    x_min = indices[1].min()
    y_min = indices[0].min()
    x_max = indices[1].max()
    y_max = indices[0].max()

    color_img = np.zeros(img.shape, img.dtype)
    color_img[:,:,:] = overlay_color
    colored_patch = color_img[y_min:y_max,x_min:x_max,:]
    white_patch = np.ones((y_max-y_min,x_max-x_min,3), img.dtype)*255

    if shade_dir=='left':
        gradient = np.linspace(0, shade_strength, x_max-x_min)[None,:, None]
    elif shade_dir=='right':
        gradient = np.linspace(shade_strength, 0, x_max-x_min)[None,:, None]
    elif shade_dir=='up':
        gradient = np.linspace(shade_strength, 0, y_max-y_min)[:, None,None]
    else:
        gradient = np.linspace(0, shade_strength, y_max-y_min)[:, None,None]

    grad_img = colored_patch + (white_patch - colored_patch) * gradient
    color_img[y_min:y_max,x_min:x_max,:] = grad_img
    return color_img
 
##----------------------------
## gradient image functions 3d
##----------------------------

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
        
def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def crop_part_overlay_shade_3d(img, mask, overlay_color, dest_color, overlay_intensity, shade=None, shade_type='simple_grad'):
    
    if shade!=None:
        assert shade=='left' or shade=='right' or shade=='up' or shade=='down' or shade=='diag_1' or shade=='diag_2' or shade=='diag_3' or shade=='diag_4'
        
        assert shade_type=='simple_grad' or shade_type=='hsv_grad' 
        
        mask_grad = create_shaded_image_3d(img,mask,overlay_color,dest_color,shade_dir=shade)
        color_mask = cv2.bitwise_and(mask_grad, mask_grad, mask=mask)
        
        if shade_type=='simple_grad':
            img[color_mask[:,:,0]!=0]=img[color_mask[:,:,0]!=0]*(1-overlay_intensity)+color_mask[color_mask[:,:,0]!=0]*overlay_intensity
        else:# shade_type=='hsv_grad':
            img = change_color_hsv(img, color_mask, overlay_intensity=overlay_intensity, sharpen=False)
            
    else:
        color_img = np.zeros(img.shape, img.dtype)
        color_img[:,:,:] = overlay_color
        color_mask = cv2.bitwise_and(color_img, color_img, mask=mask)
    
        img[color_mask[:,:,0]!=0]=img[color_mask[:,:,0]!=0]*(1-overlay_intensity)+color_mask[color_mask[:,:,0]!=0]*overlay_intensity
    
    return img

def create_shaded_image_3d(img,mask,overlay_color,dest_color,shade_dir='left'):
    indices = np.where(mask==255)
    x_min = indices[1].min()
    y_min = indices[0].min()
    x_max = indices[1].max()
    y_max = indices[0].max()

    color_img = np.zeros(img.shape, img.dtype)
    # color_img[:,:,:] = overlay_color
    #colored_patch = color_img[y_min:y_max,x_min:x_max,:]
    #white_patch = np.ones((y_max-y_min,x_max-x_min,3), img.dtype)*255
    
    ##grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(False,False,False))

    if shade_dir=='left':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(True,True,True))
    elif shade_dir=='right':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, dest_color,overlay_color,(True,True,True))
    elif shade_dir=='up':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(False,False,False))
    elif shade_dir=='down':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, dest_color,overlay_color,(False,False,False))
    elif shade_dir=='diag_1':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(True,True,True))
        grad_img = cv2.resize(ndimage.rotate(grad_img, -45, mode='nearest'), (grad_img.shape[1],grad_img.shape[0]))
    elif shade_dir=='diag_2':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(True,True,True))
        grad_img = cv2.resize(ndimage.rotate(grad_img, -135, mode='nearest'), (grad_img.shape[1],grad_img.shape[0]))
    elif shade_dir=='diag_3':
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(True,True,True))
        grad_img = cv2.resize(ndimage.rotate(grad_img, 135, mode='nearest'), (grad_img.shape[1],grad_img.shape[0]))
    else:
        grad_img = get_gradient_3d(x_max-x_min,y_max-y_min, overlay_color,dest_color,(True,True,True))
        grad_img = cv2.resize(ndimage.rotate(grad_img, 45, mode='nearest'), (grad_img.shape[1],grad_img.shape[0]))
    
    
    
    ##grad_img = colored_patch + (white_patch - colored_patch) * gradient
    color_img[y_min:y_max,x_min:x_max,:] = grad_img
    return color_img
    
##----------------------------
## hsv color change
##----------------------------
from skimage.filters import gaussian
def sharpen_img(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def change_color_hsv(image, mask,overlay_intensity=0.5,sharpen=False):
    # print('hsv grad')
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    tar_hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)

    if np.sum(tar_hsv[:, :, 0:1])!=0:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]
    # elif np.sum(tar_hsv[:, :, 0:2])!=0:
    #     image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    #     image_hsv[:, :, 0:2] = image_hsv[:, :, 0:2]*(1-0.5) + tar_hsv[:, :, 0:2]*0.5
    # else:
    #     image_hsv[:, :, 0:3] = tar_hsv[:, :, 0:3]
    #     image_hsv[:, :, 0:3] = image_hsv[:, :, 0:3]*(1-0.7) + tar_hsv[:, :, 0:3]*0.3

    else:
        image_hsv[:, :, :] = tar_hsv[:, :, :]
        # image_hsv[:, :, :] = image_hsv[:, :, :]*(1-0.7) + tar_hsv[:, :, :]*0.3

        # sharpen=True


    # image_hsv[mask!=0] = tar_hsv[mask!=0]
    
    # image_hsv[:, :, :] = image_hsv[:, :, :]*(1-overlay_intensity) + tar_hsv[:, :, 0:1]*overlay_intensity

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    if sharpen==True:
        changed = sharpen_img(changed)
    
    changed[mask==0] = image[mask==0]
    changed[mask!=0] = changed[mask!=0]*overlay_intensity + image[mask!=0]*(1-overlay_intensity)
    #changed = changed*overlay_intensity + img*(1-overlay_intensity)
    return changed


def apply_smooth_color(img, img_landmarks, connections, landmark_deltas, overlay_color,dest_color, overlay_intensity=1, 
            close_flag=True, bezier_flag=False, shade=None, line_thickness=2,fill_mask_flag=True,
            s=3.0,k=2,shade_type='simple_grad', return_mask=True, dilate_iters=3, int_blur=0.5, 
            mask_connections=None, mask_deltas=None,blur_iters=1,blur_kernel=(51,51),seam_flag=False,
            img_parsed=None, seam_type='mixed', dilate_shade=0, seam_mask_iters=0):
    
    assert seam_type=='normal' or seam_type=='mixed'

    shaded_img, filled_mask = get_cropped_part_overlay_delta_3d(img.copy(), img_landmarks, connections, landmark_deltas, overlay_color,dest_color, 
            overlay_intensity=overlay_intensity, close_flag=close_flag, bezier_flag=bezier_flag, shade=shade, line_thickness=line_thickness,
            fill_mask_flag=fill_mask_flag,s=s,k=k,shade_type=shade_type, return_mask=return_mask, img_parsed=img_parsed, dilate_shade=dilate_shade)
    
    
    if mask_connections==None:
        if dilate_iters>0:
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv2.dilate(filled_mask, kernel, iterations=dilate_iters)
        elif dilate_iters<0:
            kernel = np.ones((5,5), np.uint8)
            dilated_mask = cv2.erode(filled_mask, kernel, iterations=abs(dilate_iters))
        else:
            dilated_mask = filled_mask
    else:
        if mask_deltas==None:
            dilated_mask = make_mask_delta(img, img_landmarks, mask_connections, None, close_flag=close_flag, bezier_flag=False,
            line_thickness=2, fill_mask_flag=fill_mask_flag, img_parsed=img_parsed)
        else:
            dilated_mask = make_mask_delta(img, img_landmarks, mask_connections, mask_deltas, close_flag=close_flag, bezier_flag=False,
            line_thickness=2, fill_mask_flag=fill_mask_flag, img_parsed=img_parsed)
    
    if blur_iters==0:
        smooth_img = shaded_img

    else:
        blur_img = cv2.GaussianBlur(shaded_img,blur_kernel,0)
        for n_b in range(1,blur_iters):
            blur_img = cv2.GaussianBlur(blur_img,blur_kernel,0)
        
        smooth_img = img.copy()
        smooth_img[dilated_mask!=0] = smooth_img[dilated_mask!=0]*(1-int_blur) + blur_img[dilated_mask!=0]*int_blur
    
    
    if seam_flag==True:
        if seam_mask_iters>0:
            kernel = np.ones((5,5), np.uint8)
            seam_mask = cv2.dilate(dilated_mask, kernel, iterations=seam_mask_iters)
        elif seam_mask_iters<0:
            kernel = np.ones((5,5), np.uint8)
            seam_mask = cv2.erode(dilated_mask, kernel, iterations=abs(seam_mask_iters))
        elif seam_mask_iters==0:
        	seam_mask = np.ones(img.shape, img.dtype)*255
        else:
        	seam_mask = dilated_mask


        if blur_iters==0:    
            smooth_img = seamless_masking(smooth_img, img, seam_mask, seam_type)
        else:
            smooth_img = seamless_masking(blur_img, img, seam_mask, seam_type)
    else:
    	seam_mask = None
    	    
    return smooth_img, filled_mask, (dilated_mask,seam_mask)
    
def seamless_masking(src,dst,mask,seam_type='mixed'):

    indices = np.where(mask==255)
    x_min = indices[1].min()
    y_min = indices[0].min()
    x_max = indices[1].max()
    y_max = indices[0].max()
    center = (round(abs(x_max+x_min)/2), round(abs(y_max+y_min)/2)) 
    if seam_type=='normal':
        seamless_img = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)#cv2.MIXED_CLONE #cv2.NORMAL_CLONE
    else:
        seamless_img = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)#cv2.MIXED_CLONE #cv2.NORMAL_CLONE
    
    return seamless_img

##----------------------------
## generate same color with different intensity
##----------------------------
def gen_shades(in_color, n_shades=10):
    half_1 = get_gradient_3d(round(n_shades/2)+1,2, [255,255,255],in_color,(True,True,True)).squeeze()[0]
    half_2 = get_gradient_3d(round(n_shades/2)+1,2, in_color,[1,1,1],(True,True,True)).squeeze()[0]
    return np.uint8(np.concatenate((half_1,half_2),axis=0))[1:n_shades]

def get_one_shade(in_color, n_shades=10, shade_strength=0.5):
    assert shade_strength>=0 and shade_strength<=1
    
    all_shades = gen_shades(in_color=in_color, n_shades=n_shades) 
    shade_idx = max(0,int(np.floor(n_shades*shade_strength)))
    shade_idx = min(len(all_shades)-1,int(np.floor(n_shades*shade_strength)))
    return all_shades[shade_idx]



##----------------------------
## preserve region you dont want to change
##----------------------------
def preserve_region(img,img_landmarks,makeup_img,region):#,dilate_iters=-2,bezier_flag='smooth_poly_slpev',k=3,s=3):
    _,_,d_mask = apply_smooth_color(np.zeros(img.shape, img.dtype), img_landmarks, 
                            region, 
                            None, 
                            overlay_color=[255,255,255], 
                            dest_color=[255,255,255],
                            overlay_intensity=1,
                            close_flag=True,
                            bezier_flag='smooth_poly_slpev',
                            # s=5,k=5,
                            shade=None,
                            line_thickness=3,
                            fill_mask_flag=True,
                            shade_type='simple_grad',
                            dilate_shade=0,
                            dilate_iters=-1,
                            int_blur=1,
                            blur_iters=3,
                            blur_kernel=(3,3),
                            seam_flag=True,
                            seam_mask_iters=10)
    
    d_mask = d_mask[0]
    # import PIL
    # d_mask = PIL.Image.fromarray(d_mask).filter(PIL.ImageFilter.SMOOTH_MORE)
    # d_mask = np.array(d_mask)

    # cv2.imshow('xx',d_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    makeup_img_new = makeup_img.copy()
    makeup_img_new[np.where(d_mask!=0)]=img[np.where(d_mask!=0)]
    return makeup_img_new

def preserve_region_mask(img,makeup_img,img_parsed,region_names=['hair','u_lip','l_lip','background']):

    for part_name in region_names:
        indexes = np.where(np.all(img_parsed==part_color_dict[part_name], axis=-1))
        makeup_img[indexes] = img[indexes] 


    return makeup_img