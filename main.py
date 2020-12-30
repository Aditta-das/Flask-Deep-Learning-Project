from flask import Flask, redirect, url_for
from flask import render_template, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import Flask, redirect, url_for
from werkzeug import secure_filename
from PIL import Image
import cv2, os, imutils, time, webbrowser
import numpy as np
import matplotlib.pyplot as plt
import pytesseract, glob, random
from datetime import datetime
# from camera import Camera
from camera import VideoCamera
from attendence_cam import Attend
from app import *
from sqlalchemy import desc


from flask import Flask
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
######################Admin################################
# set optional bootswatch theme
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'
# 
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')

admin.add_view(ModelView(Blogpost, db.session))
admin.add_view(ModelView(ImageList, db.session))
admin.add_view(ModelView(AdminPanel, db.session))
admin.add_view(ModelView(User, db.session))
# Add administrative views here
######################Admin################################




######################Index################################

@app.route("/")
def index():
    
    return render_template("index.html")
######################Index################################

######################Extra################################

@app.route("/contact", methods=['GET', 'POST'])
def blog():
    contents = 0
    if request.method == "POST":
        contents = request.form.get("content")
        post = Blogpost(content=contents)
        db.session.add(post)
        db.session.commit()
    return render_template("contact.html", contents=contents, post=post)
######################Extra################################

######################Person################################

@app.route("/person", methods=["GET", "POST"])
def person():
    sfname = 0
    b = 0
    if request.method == "POST":
        f = request.files["file"]
        sfname = 'static/upload/'+str(secure_filename(f.filename))
        f.save(sfname)
        # Load Yolo
        net = cv2.dnn.readNet("yolov3_custom_best.weights", "yolov3_custom.cfg")
        classes = ["Person"]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # Loading image
        img = cv2.imread(f"{sfname}")
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        predict = len(boxes)
        a = (confidences)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        b = len(indexes)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h),(0, 255, 255), 1)
                cv2.putText(img, label, (x, y - 6), font, 0.5, (0, 128, 156), 2)
        PIL_image = Image.fromarray(np.uint8(img)).convert("RGB")
        PIL_image.save(f"{sfname}")

    return render_template("person.html", sfname=sfname, b=b)
######################person#############################################################################################

############################################licence_plate#################################################################    
@app.route("/licence", methods=["GET", "POST"])

def licence_plate():
    image = 0
    licence_name = 0
    status = 0
    boxes = []
    l = []
    if request.method == "POST":
        ff = request.files["file"]
        licence_name = 'static/licence/'+str(secure_filename(ff.filename))
        ff.save(licence_name)
        l.append(licence_name)
        img_ = cv2.imread(f"{licence_name}")
        images = cv2.resize(img_, (720,720) )
        plates_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus_16stages.xml') #gives me error
        plates = plates_cascade.detectMultiScale(images, 1.2, 4)
        for (x,y,w,h) in plates:
            plates_rec = cv2.rectangle(images, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(plates_rec, 'Licence', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            boxes.append([x,y,w,h])
            color_plates = images[y:y+h, x:x+w]
        PIL_image = cv2.imwrite(f'{licence_name}', images)
        a = os.path.split(f"{licence_name}")[1]
        for i in range(len(boxes)):
            i+=1
            status = cv2.imwrite('static/licence_detect/det_' + a, color_plates)
        image = glob.glob("static/licence_detect/*.jpg")[-1]
        images_list = glob.glob("static/licence_detect/*.jpg")
    return render_template("licence.html", licence_name=licence_name, image=image)
#################################################licence_plate##################################################
#################################################LICENCE VIDEO##################################################
@app.route("/video_upload", methods=["GET", "POST"])
def video_upload():
    if request.method == "POST":
        ff = request.files["file"]
        licence_video = 'static/licence_video/'+str(secure_filename(ff.filename))
        ff.save(licence_video)
        if request.files['file'].filename == '':
            return 'No selected file'
        else:
            return render_template("licence_video.html")
    return render_template("video_upload.html")


@app.route('/video_feed')
def video_feed():
    a = VideoCamera().get_frame()
    return Response((a), mimetype='multipart/x-mixed-replace; boundary=frame')
#################################################LICENCE VIDEO##################################################

#############################################ATTENDENCE VIDEO##################################################
@app.route("/confirm")
def recognize_face():
    return render_template("recognize.html")

# @app.route('/attendence')
# def attendence():
#     as_ = Attend().face_extractor()
#     return Response((as_), mimetype='multipart/x-mixed-replace; boundary=frame')

# labels = ["Christiano Ronaldo", "David Luiz", "Lieonel Messi", "Neymar", "Sergio Ramos"]
video_stream = Attend()
def gen(camera):
    while True:
        frame, pred = camera.face_extractor()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n' b'Content-Type: image/jpeg\r\n\r\n')

@app.route('/attendence')
def attendence():
    return Response(gen(video_stream), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/person_confirm')
def person_confirm():
    mail_from_server = AdminPanel.query.order_by(desc(AdminPanel.id)).all()
    date_ = datetime.now().strftime("%c")
    t_int = int(date_.split()[3].split(":")[0])
    if t_int > 0 and t_int < 12:
        name = "Good Morning"
    elif t_int > 12 and t_int < 15:
        name = "Good Noon"
    elif t_int > 15 and t_int < 19:
        name = "Good Afternoon"
    else:
        name = "Good Night" 

    _, predi = Attend().face_extractor()
    if len(predi) == 0:
        return redirect(url_for("attendence"))
    elif len(predi) != 0:
        g = os.path.join("static/capture_image/", "".join(os.listdir("static/capture_image/")[-1:]))
        h = "".join(predi[0])
        image_content = ImageList(entry=h, time=date_)
        db.session.add(image_content)
        db.session.commit()
        return render_template("attendence_confirm.html", h=h, date_=date_, name=name, mail_from_server=mail_from_server)

############################################Attendence#################################################################

############################################Dashboard#################################################################    

@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    dash = User.query.all()
    if request.method == "POST":
        name_id = request.form["names"]
        mail_to = request.form["mails"]
        d = AdminPanel(name_id=name_id, mail_to=mail_to)
        db.session.add(d)
        db.session.commit()
        return redirect(url_for("dashboard"))
    return render_template("dashboard.html", dash=dash)

############################################Dashboard#############################################################

############################################LOGIN#################################################################    
@app.route('/login')
def login():

    return render_template("login.html")

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        name = request.form["name"]
        img = request.files["image"]
        filename = 'static/accountimage/'+str(secure_filename(img.filename))
        img.save(filename)
        mimetype = img.mimetype
        user = User(email=email, password=password, name=name, mimetype=mimetype, filename=filename)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("index"))

    return render_template("signup.html")
############################################LOGIN#################################################################    
if __name__ == "__main__":
    app.secret_key = 'asdahsdj21231asdah46374627'
    # construct the argument parser and parse command line arguments
    # webbrowser.open("127.0.0.1:5000")
    app.run(debug=True)
    





