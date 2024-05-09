import cv2
import mediapipe as mp
import time
import HandTrackingModule as hm
import numpy as np
import pandas as pd
import pickle

pTime = 0
cTime = 0
largura_janela = 640
altura_janela = 480
cap = cv2.VideoCapture(1)
cap.set(3, largura_janela)
cap.set(4, altura_janela)

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

detector = hm.handDetector(static_image_mode=False, max_num_hands=1,  min_detection_confidence=0.7, min_tracking_confidence=0.7)
# labels_dict = {0: 'A', 
#                1: 'B', 
#                2: 'C',
#                3: 'D',
#                4: 'E',
#                5: 'F',
#                6: 'G',
#                7: 'I',
#                8: 'L',
#                9: 'M',
#                10: 'N',
#                11: 'O',
#                12: 'P',
#                13: 'Q',
#                14: 'R',
#                15: 'S',
#                16: 'T',
#                17: 'U',
#                18: 'V',
#                19: 'W',
#                }
# count = 1
while True:
    success, img = cap.read()
    cap.set(3, largura_janela)
    cap.set(4, altura_janela)
    lm_list, img = detector.findHands(img, draw=True, getPosition=True)
    
    data = []
    if lm_list:
        lm_lenght= detector.appendDistance(lm_list, img, draw=False)
        data.append(lm_lenght)
        df_dt_inference = detector.transformData(data)        
        df_dt_inference = detector.normalizeData(df_dt_inference)        
        prediction = model.predict(df_dt_inference)
        predicted_character = prediction[0]
        cv2.putText(img, f'Letra: {str(predicted_character)}', (450,70), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 3)
 
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime 
    
    cv2.putText(img, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 3)
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()