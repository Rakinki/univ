# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:09:57 2021

@author: kin
"""

import cv2
from imutils import face_utils
import numpy as np
import dlib
from PIL import Image
from scipy.spatial import distance as dist

EYE_AR_THRESH = 0.20


dlib_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)


def eye_aspect_ratio(eye):
  A = dist.euclidean(eye[1],eye[5])
  B = dist.euclidean(eye[2],eye[4])
  C = dist.euclidean(eye[0],eye[3])
  ear = (A + B) / (2.0 * C)
  return ear


def seachclose(faces,gray):
    (leftstart,leftend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rightstart,rightend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    #print(len(faces))
    cnt = 0 
    closelist = []

    for face in faces:
        # 顔のランドマーク検出
        landmark = predictor(gray, face)
        print("landmark:",landmark)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)
        
        print("landmark:",landmark)
        
        lefteye = landmark[leftstart:leftend]
        righteye = landmark[rightstart:rightend]
        leftear = eye_aspect_ratio(lefteye)
        rightear = eye_aspect_ratio(righteye)
        
        print("faceindex:",cnt)


        print("leftear:",leftear)
        print("rightear:",rightear)

    
        ear = (leftear + rightear ) / 2.0 
    
        print("ear:",ear)
        
        if ear < EYE_AR_THRESH:
        
            print("close eye")
            closelist += [cnt]

        #load = [face.left(), face.top(), face.right(), face.bottom()]
        else:
            print("open eye")
            
        cnt = cnt+1
        print("close face index",closelist)
    return closelist

def seachclose1(faces,gray,img):
    (leftstart,leftend) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rightstart,rightend) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    #print(len(faces))
    cnt = 0 
    closelist = []
    earlist = []

    for face in faces:
        # 顔のランドマーク検出
        landmark = predictor(gray, face)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)

        pos = landmark[0]
        #cv2.putText(img, str(cnt), pos,
        #            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        # #           fontScale=2,
        #            color=(0, 255, 255))
        
        # ランドマーク描画
        for (i, (x, y)) in enumerate(landmark):
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
        
        lefteye = landmark[leftstart:leftend]
        righteye = landmark[rightstart:rightend]
        leftear = eye_aspect_ratio(lefteye)
        rightear = eye_aspect_ratio(righteye)
        
        print("faceindex:",cnt)

        print("leftear:",leftear)
        print("rightear:",rightear)

    
        ear = (leftear + rightear ) / 2.0 
    
        print("ear:",ear)
        
        if ear < EYE_AR_THRESH:
            pos = landmark[1]
            #cv2.putText(img, str(ear), pos,
             #       fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
             #       fontScale=3,
             #       color=(255, 255, 255))
            print("close eye")
            closelist += [cnt]
            earlist += [ear]

        #load = [face.left(), face.top(), face.right(), face.bottom()]
        else:
            print("open eye")
            
        cnt = cnt+1
        print("close face index",closelist)
    print("close face ear",earlist)
    return img

if __name__ == '__main__':
    img = cv2.imread('./dataset/ture.jpg')
    
    (x,y) = img.shape[:2]
    print("size ",x,y)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(img_gry, 1)
    
    img = seachclose1(faces,img_gry,img)

    cv2.imwrite('display.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
