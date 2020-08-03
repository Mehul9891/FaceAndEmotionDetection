import cv2
import face_recognition

mehul_img = face_recognition.load_image_file("G:\\Jupyter\\pic.JPG")
unknown_img = face_recognition.load_image_file("G:\\Jupyter\\pic2.JPG")

mehul_encoding = face_recognition.face_encodings(mehul_img)[0]
unknown_encoding = face_recognition.face_encodings(unknown_img)[0]

result = face_recognition.compare_faces([mehul_encoding],unknown_encoding,0.6)

if result[0]:
    print("Match")
else:
    print("No Match")