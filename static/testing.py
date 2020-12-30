import pytesseract # this is tesseract module
# pytesseract.pytesseract.tesseract_cmd = r'D:\Programing Languages\Python\Tesseract-OCR\tesseract.exe')
import matplotlib.pyplot as plt 
import cv2 # this is opencv module 
import glob 
import os

# path_for_license_plates = os.getcwd() + "/licence_detect/*"
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# list_license_plates = [] 
# predicted_license_plates = [] 

# for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
# 	# print(path_to_license_plate) 
	
# 	license_plate_file = path_to_license_plate.split("/")[-1]
# 	print(license_plate_file) 
# 	license_plate, _ = os.path.splitext(license_plate_file)
# 	print(license_plate) 
# # 	''' 
# # 	Here we append the actual license plate to a list 
# # 	'''
# 	list_license_plates.append(license_plate) 
	
# # 	''' 
# # 	Read each license plate image file using openCV 
# # 	'''
# 	img = cv2.imread(path_to_license_plate) 
# 	img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# 	retval, threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# 	plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	thresh, plate = cv2.threshold(plate, 127, 255, cv2.THRESH_BINARY)
	
# 	predicted_result = pytesseract.image_to_string(threshold, lang ='eng', config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') 
# 	print(predicted_result)
# 	# print(predicted_result)
# 	filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "") 
# 	predicted_license_plates.append(filter_predicted_result) 
# # print(predicted_license_plates)


# print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy") 
# print("--------------------", "\t", "-----------------------", "\t", "--------") 




# def calculate_predicted_accuracy(predicted_list): 
	# for predict_plate in predicted_list:
		# print(predict_plate)
# 		accuracy = "0 %"
# 		num_matches = 0

	# for actual_plate, predict_plate in zip(actual_list, predicted_list): 
	# 	accuracy = "0 %"
	# 	num_matches = 0
	# 	if actual_plate == predict_plate: 
# 			accuracy = "100 %"
# 		else: 
# 			if len(actual_plate) == len(predict_plate): 
# 				for a, p in zip(actual_plate, predict_plate): 
# 					if a == p: 
# 						num_matches += 1
# 				accuracy = str(round((num_matches / len(actual_plate)), 2) * 100) 
# 				accuracy += "%"
# 		print("	 ", actual_plate, "\t\t\t", predict_plate, "\t\t ", accuracy) 

		
# calculate_predicted_accuracy(predicted_license_plates)


import cv2
cap = cv2.VideoCapture("E:\\ben10\\static\\russia.mp4")
plates_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml') #gives me error

while(True):
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  plates = plates_cascade.detectMultiScale(frame, 1.2, 4)

  for (x,y,w,h) in plates:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
  #     # cv2.putText(plates_rec, 'Licence', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
  #     boxes.append([x,y,w,h])
  #     color_plates = gray[y:y+h, x:x+w]
  cv2.imshow("imges", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()