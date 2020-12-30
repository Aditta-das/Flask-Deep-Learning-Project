import os

BASE_DIR = os.path.dirname(os.path.abspath(__name__))

class Config:
	DEBUG = True
	SQLALCHEMY_DATABASE_URI = "postgresql://postgres:1234@localhost:5432/newdata"
	SQLALCHEMY_TRACK_MODEIFICATION = False
	SECRET_KEY = 'asdahsdj21231asdah46374627'