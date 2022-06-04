# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:31:06 2021

@author: jinzhaohui
"""

import dlib
import cv2
#import glob
import numpy as np
 
detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

path = "alignment_face/"
#path = "faces/"
RE_THRESH = 0.09
remin = 0.09
cnt=0
rec = []
img_index = []

bg_diff_path = "./sanbun/"
close_path = "./closeeyealpha/"
 
 
def get_feature(path):
	img = cv2.imread(path)
	dets = detector(img)
	#print('检测到了 %d 个人脸' % len(dets))
	# 这里假设每张图只有一个人脸
	shape = predictor(img, dets[0])
	face_vector = facerec.compute_face_descriptor(img, shape)
	return(face_vector)
 
def distance(a,b):
	a,b = np.array(a), np.array(b)
	sub = np.sum((a-b)**2)
	add = (np.sum(a**2)+np.sum(b**2))/2.
	return sub/add
 
imgindex = 19
faceindex = 10

tureface = 2

face = 11
    
#img_src01 = imread(path+"img" + str(imgindex) + "_face_" + str(faceindex) +".jpg")
#img_src01 = imread(path+"ture_face_" + str(tureface) +".jpg")
#ture face
feature_ture = get_feature(path+"ture_face_" + str(tureface) +".jpg")
#close face
feature_close = get_feature(path+"img" + str(imgindex) + "_face_" + str(faceindex) +".jpg")
feature_close = get_feature("./closeeyealpha/alpha_"+str(face)+".jpg")


for x in range(33):
    if (x+1) != (face-1)*3+1 and (x+1) != (face-1)*3+2 and (x+1) != (face-1)*3+3 and x+1 != 7:
        for y in range(11):
            img_src02 = cv2.imread(path+"img" + str(x+1)+ "_face_" + str(y) +".jpg")
            feature_test = get_feature(path+"img" + str(x+1)+ "_face_" + str(y) +".jpg")
            print("img" + str(x+1)+ "_face_" + str(y) +".jpg")
            
            out = distance(feature_close,feature_test)
            print("face recognition ",out)
            
            if out < RE_THRESH and out > 0.0:
                #cv2.imwrite(bg_diff_path+"img" + str(x+1)+ "_face_" + str(y) +"_diff_" + str(cnt)+"san_"+ str(san)+".jpg", fgmask)
                #cv2.imwrite(bg_diff_path+"img" + str(x+1)+ "_face_" + str(y) +"_diff_" + str(cnt)+"rec_"+ str(round(out, 4))+".jpg", img_src02)
                if out < remin :
                    remin = out
                    #cv2.imwrite(bg_diff_path+"img" + str(x+1)+ "_face_" + str(y) +"_diff_" + str(cnt)+"rec_"+ str(round(out, 4))+".jpg", img_src02)
                    rec += [out]
                    if int((x+1)%3) == 0:
                        accessface = (int((x+1)/3),3,y)
                    else:
                        accessface = (int((x+1)/3)+1,(x+1)%3,y)
                    best_face = img_src02
                    cv2.imwrite(close_path+ "close_best_face_" +str(face) + "_rec_"+str(remin)+"_access_"+str(accessface)+".jpg", best_face)
                    img_index += [((x+1),y)]
    cnt = cnt + 1
print("recognition ",rec,img_index)
#cv2.imwrite(bg_diff_path+ "best_face_" +str(tureface) + "_rec_"+str(remin)+"_access_"+str(accessface)+".jpg", best_face)
cv2.imwrite(close_path+ "close_best_face_" +str(face) + "_rec_"+str(remin)+"_access_"+str(accessface)+".jpg", best_face)
print("accessface and %",accessface,remin)
        
'''        
path_lists1 = ["./alignment_face/ture_face_7.jpg","./alignment_face/img1_face_8.jpg"]
path_lists2 = ["./alignment_face/ture_face_7.jpg","./alignment_face/img19_face_10.jpg"]
 
feature_lists1 = [get_feature(path) for path in path_lists1]
feature_lists2 = [get_feature(path) for path in path_lists2]
 
print("feature 1 shape",feature_lists1[0].shape)
 
out1 = distance(feature_lists1[0],feature_lists1[1])
out2 = distance(feature_lists2[0],feature_lists2[1])
 
print("open distance is",out1)
print("close distance is",out2)
 
def classifier(a,b,t = 0.09):
	if(distance(a,b)<=t):
		ret = True
	else :
		ret = False
	return(ret)
 
print("img1 is close",classifier(feature_lists1[0],feature_lists1[1]))
print("img2 is close",classifier(feature_lists1[0],feature_lists2[1]))
print("img2 is ture1.jpg",classifier(feature_lists2[0],feature_lists2[1]))

'''