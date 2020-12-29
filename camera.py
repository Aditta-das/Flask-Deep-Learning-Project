import cv2, glob
from flask import render_template, request, Response
import pytesseract

class VideoCamera(object):

    def __init__(self):
        path = glob.glob("static/licence_video/*")
        video_source = "".join(path[0])
        self.video = cv2.VideoCapture(video_source)
            

    def __del__(self):
        self.video.release()

    def get_frame(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\Biplob\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
        while True:
            new = 0
            _, image = self.video.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plates_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') #gives me error
            plates = plates_cascade.detectMultiScale(image, 1.2, 5)

            for (x,y,w,h) in plates:
                plates_rec = cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
                color_image = plates_rec[y:y+h, x:x+w]
                extract_image = cv2.imwrite(f'static/licence_video/{new}.jpg', color_image)
                # thresh = cv2.threshold(color_image, 0, 255, cv2.THRESH_BINARY)[1]
                data = pytesseract.image_to_string(color_image, config='-l eng --oem 1 --psm 3')
                # print(data)
                cv2.putText(plates_rec, 'Licence', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                new+=1
            _, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')