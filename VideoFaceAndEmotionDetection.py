from PIL import Image, ImageDraw
import cv2
import numpy as np
import face_recognition
import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle

cap = cv2.VideoCapture(0)

# 1st Model
# model = load_model("emotion_detector_model/model_v6_23.hdf5")
# emotion_dict = { 0 : 'Angry', 5 :'Sad', 4 : 'Neutral', 1 : 'Disgust', 6 : 'Surprise', 2 : 'Fear', 3 : 'Happy'}

# 2nd Model
#model = load_model("emotion_detector_model/emotion_model.hdf5")
#emotion_dict = {0: 'angry', 1:'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# 3rd Model - Best Fit
model = load_model("emotion_detector_model/model_filter.h5")
emotion_dict = {0: 'angry', 1:'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


while True:
    ret, frame = cap.read()
    if ret == True:
        fac_cor = face_recognition.face_locations(frame)
        print(fac_cor)
        for i in range(len(fac_cor)):
            top, right, bottom, left = fac_cor[i]
            faceImage = frame[top:bottom, left:right]
            cv2.rectangle(frame, (fac_cor[i][3], fac_cor[i][0]), (fac_cor[i][1], fac_cor[i][2]),(0,0,255), 1)
            faceImage = cv2.resize(faceImage, (48, 48))
            faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
            faceImage = np.reshape(faceImage, [1, faceImage.shape[0], faceImage.shape[1], 1])
            predicted_class = np.argmax(model.predict(faceImage))
            predicted_label = emotion_dict[predicted_class]
            print(predicted_label)
            frame = cv2.putText(frame, predicted_label, (fac_cor[i][3],fac_cor[i][0]), cv2.FONT_ITALIC, 1, (255, 255, 255), 1, cv2.LINE_4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

cv2.destroyAllWindows()