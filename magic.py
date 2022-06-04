# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:10:32 2021

@author: kin
"""

import cv2
import numpy as np
import dlib
from mask import seachclose
FEATHER_AMOUNT = 11
OVERLAY_POINTS = list(range(17, 26)) +  list(range(36, 47)) 
#OVERLAY_POINTS = list(range(36, 47)) 
#OVERLAY_POINTS = list(range(0, 61)) 

left_eye = list(range(42, 49))
right_eye = list(range(36, 41))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))


align = (left_eye + right_eye + left_brow + right_brow + nose + mouth)

dlib_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)


def get_landmark(img):
    # 処理高速化のためグレースケール化(任意)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gry, 1)
    shape = predictor(img_gry, faces[3]).parts()
    return np.matrix([[p.x, p.y] for p in shape])

def get_close_landmark(img):
    # 処理高速化のためグレースケール化(任意)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gry, 1)
    #seachclose(faces,img_gry)[0]
    shape = predictor(img_gry, faces[4]).parts()
    return np.matrix([[p.x, p.y] for p in shape])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 10, color=(0, 255, 255),thickness=-1)
    return im


def draw_convex_hull(im, points, color):
    img = im.copy()
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
    cv2.fillConvexPoly(img, points =points, color=(255, 25, 255))
    cv2.imwrite('./test/hull.jpg', img)
    
    
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


def read_im_and_landmarks(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #img = resize(img)
    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    s = get_landmark(img)
    return img, s

def read_im_and_closelandmarks(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #img = resize(img)
    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    s = get_close_landmark(img)
    return img, s


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


# the source get the target orgin
def swap_orgin(source_path, target_path):
    source, landmark1 = read_im_and_closelandmarks(source_path)
    target, landmark2 = read_im_and_landmarks(target_path)
    
    source1 = annotate_landmarks(source,landmark1)
    cv2.imwrite('./test/source1.jpg', source1)
    target1 = annotate_landmarks(target,landmark2)
    cv2.imwrite('./test/target1.jpg', target1)
    
    M = affine_matrix(landmark1[align], landmark2[align])
    
    mask = get_face_mask(target1, landmark2)
    #im_out = cv2.bitwise_and(target1, mask)
    #cv2.imwrite('./test/im_out.jpg', im_out)
    cv2.imwrite('./test/mask.jpg', mask)
    
    
    warp_mask = warp_im(mask, M, source.shape)
    cv2.imwrite('./test/warp_mask.jpg', warp_mask)
    
    combined_mask = np.max([get_face_mask(source, landmark1), warp_mask],
                              axis=0)
    cv2.imwrite('./test/combined_mask.jpg', combined_mask)
    
    
    
    warp_target = warp_im(target, M, source.shape)
    cv2.imwrite('./test/warp_target.jpg', warp_target)
    
    correct_target = correct_colours(source, warp_target, landmark1)
    cv2.imwrite('./test/correct_target.jpg', correct_target)
    output_img = source*(1.0-combined_mask) + correct_target*combined_mask
    cv2.imwrite('./test/output_img.jpg', output_img)
    #output_img = source*(1.0-combined_mask) + warp_target*combined_mask
    return output_img

def test(source_path, target_path):
    source, landmark1 = read_im_and_closelandmarks(source_path)
    target, landmark2 = read_im_and_landmarks(target_path)
    
    mask = get_face_mask(target, landmark2)
    M = affine_matrix(landmark1[align], landmark2[align])
    warped_mask = warp_im(mask, M, source.shape)
    combined_mask = np.max([get_face_mask(source, landmark1), warped_mask],
                         axis=0)
    warped_im2 = warp_im(target, M, source.shape)
    warped_corrected_im2 = correct_colours(source, warped_im2, landmark1)

    output_im = source * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('./test/1111output_img.jpg', output_im)

if __name__ == '__main__':
    open_path = './dataset/2_close2.jpg'
    close_path = './dataset/6_close3.jpg'
    
    mask = cv2.imread('./test/hull.jpg')
    target1 = cv2.imread('./dataset/6_close3.jpg')
    
    asas = cv2.bitwise_and(target1, mask)
    
    img = swap_orgin(close_path,open_path)
    cv2.imwrite('finalA.jpg', img)
    
    test(close_path,open_path)
    
    cv2.imwrite('./test/asasasasas.jpg', asas)
    
    
    '''   
    open_path = './dataset/4_close2.jpg'
    close_path = './dataset/11_close1.jpg'
    
    close_img = swap_orgin(close_path,open_path)
    cv2.imwrite('./final/face11.jpg', close_img)

    path3 = './test/eye.jpg'
    out_img1 = swap_orgin(path3, path2)
    cv2.imwrite('./test/eye1.jpg', out_img1)
    '''