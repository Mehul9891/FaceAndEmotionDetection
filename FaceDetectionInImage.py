import face_recognition
import cv2

img = face_recognition.load_image_file("G:\\Jupyter\\pic.JPG")
img = cv2.resize(img,(1024,900))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fac_loc = face_recognition.face_locations(img)[0]

print(fac_loc)
img[fac_loc[0]]
cv2.rectangle(img,(fac_loc[3],fac_loc[0]),(fac_loc[1],fac_loc[2]),(255,0,255),2)

cv2.imshow('img',img)


cv2.waitKey(0)
cv2.destroyAllWindows()