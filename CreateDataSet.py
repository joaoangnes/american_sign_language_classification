import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import HandTrackingModule as hm

detector = hm.handDetector(static_image_mode=True, max_num_hands=1)
DATA_DIR = 'images'

data = []
classes = []
for dir_ in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    
    print('Creating dataset for class:', dir_)
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if not img_path.endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        lm_list, img = detector.findHands(img, draw=False, getPosition=True) 
        if len(lm_list) != 0:
            lm_lenght= detector.appendDistance(lm_list, img, draw=False)
            data.append(lm_lenght)
            classes.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'class': classes}, f)
f.close()