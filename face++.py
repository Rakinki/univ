# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:59:34 2022

@author: jinzhaohui
"""

# -*- coding: utf-8 -*-
"""
@author: jinzhaohui
"""
 
import cv2
import requests  
import json  
import numpy as np
from PIL import Image
from numpy import mat
from scipy.spatial import distance as dist


FEATHER_AMOUNT = 11
left_eye = list(range(65, 74))
right_eye = list(range(19, 28))
left_brow = list(range(75, 82))
right_brow = list(range(29, 36))
mouth = list(range(37, 54))
nose = list(range(55, 64))
OVERLAY_POINTS = left_eye + right_eye
align = (left_eye + right_eye + left_brow + right_brow + nose + mouth)

def eye_aspect_ratio(eye):
  A = dist.euclidean(eye[27],eye[22])
  B = dist.euclidean(eye[28],eye[23])
  C = dist.euclidean(eye[21],eye[25])
  ear = (A + B) / (2.0 * C)
  return ear


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def affine_matrix(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0) 
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])
    
def read_im_and_landmarks(openfile,j):
    im = Image.open(openfile)
    w,h = im.size
    im_ss = im.resize((w,h),Image.ANTIALIAS)

    filename = './dataset2/3_close2.jpg'
    im_ss.save(filename,"JPEG")
    img = cv2.imread(openfile, cv2.IMREAD_COLOR)
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'  
    files = {'image_file':open(filename, 'rb')}  
    payload = {'api_key': 'B4ntJTSCffm-pmMwgQ_AqkGXCm61ago-',  
              'api_secret':'jE5z3Df0yu3_3uwVdDjhDYcLJ6P0k5fZ',  
              'return_landmark': 1,
              'return_attributes':'eyestatus'
              }  
    r = requests.post(url,files=files,data=payload) 
    data=json.loads(r.text)
    a = np.empty((0, 2), dtype=int)
    for i in data['faces'][j]['landmark']:
        cor=data['faces'][j]['landmark'][i]
        x=cor["x"]
        y=cor["y"]
        b = np.array([x,y])
        a = np.append(a, np.array([b]), axis=0)
        mat_a = mat(a)
    return img,mat_a

def read_im_and_closelandmarks(closefile,j):
    im = Image.open(closefile)
    w,h = im.size
    im_ss = im.resize((w,h),Image.ANTIALIAS)

    filename = './dataset2/1_close1.jpg'
    im_ss.save(filename,"JPEG")
    img = cv2.imread(closefile, cv2.IMREAD_COLOR)
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'  
    files = {'image_file':open(filename, 'rb')}  
    payload = {'api_key': 'B4ntJTSCffm-pmMwgQ_AqkGXCm61ago-',  
              'api_secret':'jE5z3Df0yu3_3uwVdDjhDYcLJ6P0k5fZ',  
              'return_landmark': 1,
              'return_attributes':'eyestatus'
              }  
    r = requests.post(url,files=files,data=payload) 
    data=json.loads(r.text)
    a = np.empty((0, 2), dtype=int)
    for i in data['faces'][j]['landmark']:
        cor=data['faces'][j]['landmark'][i]
        x=cor["x"]
        y=cor["y"]
        b = np.array([x,y])
        a = np.append(a, np.array([b]), axis=0)
        mat_a = mat(a)
    return img,mat_a
    
def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    white = [OVERLAY_POINTS]
    #white = [mouth, left_eye, right_eye, left_brow, right_brow, nose]
    for group in white:
        points = landmarks[group]
        draw_convex_hull(im, points, 1)
    
    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def warp_im(img, M, shape):
    output_img = np.zeros(shape, dtype=img.dtype)
    cv2.warpAffine(img,
                   M[:2],
                   (shape[1], shape[0]),
                   dst=output_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_img

def correct_colours(im1, im2, landmarks1):
    blur_amount = 0.6 * np.linalg.norm(
                              np.mean(landmarks1[left_eye], axis=0) -
                              np.mean(landmarks1[right_eye], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))
    
def test(closefilename,openfilename,closenum, opennum):
    source, landmark1 = read_im_and_closelandmarks(closefilename,closenum)
    target, landmark2 = read_im_and_landmarks(openfilename,opennum)
    
    mask = get_face_mask(target, landmark2)
    M = affine_matrix(landmark1[align], landmark2[align])
    warped_mask = warp_im(mask, M, source.shape)
    combined_mask = np.max([get_face_mask(source, landmark1), warped_mask],
                         axis=0)
    warped_im2 = warp_im(target, M, source.shape)
    warped_corrected_im2 = correct_colours(source, warped_im2, landmark1)

    output_im = source * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('face++output_img.jpg', output_im)

closefilename = './dataset/1_close1.jpg'
openfilename = './dataset/3_close2.jpg'
test(closefilename,openfilename,8,6)