from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import Flask
from config import *


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:1234@localhost:5432/newdata"
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Blogpost(db.Model):
    __tablename__ = 'Blog'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)

    def __init__(self, content):
        self.content = content

class ImageList(db.Model):
	__tablename__ = 'Image_box'
	id = db.Column(db.Integer, primary_key=True)
	entry=db.Column(db.Text)
	time = db.Column(db.DateTime)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, unique=True)
    password = db.Column(db.String)
    name = db.Column(db.Text, unique=True)
    filename = db.Column(db.Text)
    img = db.Column(db.Text)
    mimetype = db.Column(db.Text)
    pets = db.relationship('AdminPanel', backref='owner')
    def __repr__(self):
        return self.name

class AdminPanel(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	mail_to = db.Column(db.Text)
	name_id = db.Column(db.Text, db.ForeignKey('user.name'))
    
