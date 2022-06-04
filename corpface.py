# encoding:utf-8

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

path_save = "final/"
path_start = "final/used_open/"

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def face_alignment(faces):

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 用来预测关键点
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
        #order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        #for j in order:
            #x = shape.part(j).x
            #y = shape.part(j).y
            #cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned

def demo():
    count = 1
    
    for i in range(11):
        for j in range(3):
            im_raw = cv2.imread(path_start + str(i+1)+ "_close"+str(j+1)+".jpg").astype('uint8')
            print(str(i+1)+ "_close"+str(j+1)+".jpg")


            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
        
            src_faces = []
            for (k, rect) in enumerate(rects):
                (x, y, w, h) = rect_to_bb(rect)
                detect_face = im_raw[y:y+h,x:x+w]
                src_faces.append(detect_face)
       
            faces_aligned = face_alignment(src_faces)
        
            #cv2.imshow("src", im_raw)
            index = 0
            for face in faces_aligned:
                #cv2.imshow("det_{}".format(i), face)
                
                print("Save to:", "img" + str(count)+ "_face_" + str(index)+".jpg")
                face = cv2.resize(face, (250,250))
                cv2.imwrite(path_save+"img" + str(count)+ "_face_" + str(index) +".jpg", face)
                #print(face.shape[:2])
                index = index + 1
                cv2.waitKey(0)
            count =count +1
    return count      

def demo1():
    im_raw = cv2.imread('./dataset/ture3.jpg')
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
        
    src_faces = []
    for (k, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        detect_face = im_raw[y:y+h,x:x+w]
        src_faces.append(detect_face)
        
        faces_aligned = face_alignment(src_faces)
        
        #cv2.imshow("src", im_raw)
        index = 0
        for face in faces_aligned:
            #cv2.imshow("det_{}".format(i), face)
                
            face = cv2.resize(face, (250,250))
            cv2.imwrite(path_save+"ture_face_" + str(index) +".jpg", face)
            #print(face.shape[:2])
            index = index + 1
            cv2.waitKey(0)
            
            
def demo2():
    count = 1
    
    for i in range(11):
            im_raw = cv2.imread(path_start + "face" + str(i+1)+".jpg").astype('uint8')
            print("face" + str(i+1)+".jpg")


            detector = dlib.get_frontal_face_detector()
            gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
        
            src_faces = []
            for (k, rect) in enumerate(rects):
                (x, y, w, h) = rect_to_bb(rect)
                detect_face = im_raw[y:y+h,x:x+w]
                src_faces.append(detect_face)
                        
            faces_aligned = face_alignment(src_faces)
        
            #cv2.imshow("src", im_raw)
            index = 0
            for face in faces_aligned:
                #cv2.imshow("det_{}".format(i), face)
                
                print("Save to:", "img" + str(count)+ "_face_" + str(index)+".jpg")
                face = cv2.resize(face, (250,250))
                cv2.imwrite(path_save+"img" + str(count)+ "_face_" + str(index) +".jpg", face)
                #print(face.shape[:2])
                index = index + 1
                cv2.waitKey(0)
            count =count +1
    return count      

if __name__ == "__main__":

    print(demo2())