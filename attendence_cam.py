from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from flask import render_template, request, Response
from flask import Flask, redirect, url_for
from keras.preprocessing import image

from app import *

model = load_model('attendence_check\\train2.h5')
labels = ["Christiano Ronaldo", "David Luiz", "Lieonel Messi", "Neymar", "Sergio Ramos", "Shouma", "Aditta Das Nishad"]

# class Attend(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)

#     def __del__(self):
#         self.video.release()

#     def face_extractor(self):
#         my_pred = []
#         num = 0
#         while num < 2:
#             _, image = self.video.read()
#             colors = cv2.cvtColor(image, cv2.IMREAD_COLOR)
#             face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#             faces = face_cascade.detectMultiScale(image, 1.3, 5)
        
#             if faces is ():
#                 pass
        
#             # Crop all faces found
#             for (x,y,w,h) in faces:
#                 faces_rec = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
#                 cropped_face = faces_rec[y:y+h, x:x+w]
#                 cv2.imwrite(f"static//capture_image//open_cv_{num}.jpg", cropped_face)
#                 if type(cropped_face) is np.ndarray:
#                     face = cv2.resize(cropped_face, (224, 224))
#                     im = Image.fromarray(face, 'RGB')
#                     img_array = np.array(im)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     prediction = model.predict(img_array)
#                     pred_max = np.argmax(prediction)
#                     pred_max = int(pred_max)
#                     pred = labels[pred_max]
#                     my_pred.append(pred)
#                     num+=1
                
#             _, jpeg = cv2.imencode('.jpg', image)
#             frame = jpeg.tobytes()
#             yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


class Attend(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def face_extractor(self):
        my_pred = []
        num = 0
        while num < 2:
            _, image = self.video.read()
            colors = cv2.cvtColor(image, cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(image, 1.3, 5)

            if faces is ():
                pass

            # Crop all faces found
            for (x,y,w,h) in faces:
                faces_rec = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
                cropped_face = faces_rec[y:y+h, x:x+w]
                
                cv2.imwrite(f"static//capture_image//open_cv_{num}.jpg", cropped_face)
                if type(cropped_face) is np.ndarray:
                    face = cv2.resize(cropped_face, (224, 224))
                    im = Image.fromarray(face, 'RGB')
                    img_array = np.array(im)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prediction = model.predict(img_array)
                    pred_max = np.argmax(prediction)
                    pred_max = int(pred_max)
                    pred = labels[pred_max]
                    my_pred.append(pred) 
                    num += 1
        _, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes(), my_pred

    def make_pred(self):
        _, p = self.face_extractor()
        return p