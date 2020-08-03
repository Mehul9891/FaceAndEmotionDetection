import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from tensorflow.keras.models import load_model
import pickle



TEST_DIR = "G:\\Jupyter\\face-expression-recognition-dataset\\images\\test"
TRAIN_DIR = "G:\\Jupyter\\face-expression-recognition-dataset\\images\\train"
categories = ["angry","disgust","fear","happy","neutral","sad","surprise"]


train_data = []
train_label = []
IMG_SIZE = 48


def label_image(folder):
   return folder.split('\\')[5]


def create_train_data():
    for category in categories:
        folder = os.path.join(TRAIN_DIR, category)
        for img in tqdm(os.listdir(folder)):
            label = label_image(folder)
            path = os.path.join(folder,img)
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            train_data.append(np.array(img).ravel())
            train_label.append(label)


test_data = []
test_label = []

def process_test_data():
    test_data_set = []
    for category in categories:
        folder = os.path.join(TEST_DIR, category)
        for img in tqdm(os.listdir(folder)):
            label = label_image(folder)
            path = os.path.join(folder, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            test_data.append(np.array(img).ravel())
            test_label.append(label)


create_train_data()
process_test_data()


#model = KNeighborsClassifier(13)
# model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features='auto')
# model =  LinearDiscriminantAnalysis()
#model = DecisionTreeClassifier()


model.fit(train_data,train_label)

filename = 'emotion_detector_model/myModel.sav'
pickle.dump(model, open(filename, 'wb'))

p = model.predict(test_data)

count = 0
print(len(p))
for i in range(0,len(p)) :
    print(str(i)+") "+p[i] + " : "+ test_label[i])
    if p[i] == test_label[i]:
        count += 1

print("Accuracy is : "+ str(count/len(p) * 100))


# Available Model Test for Accuracy
'''
model = load_model("emotion_detector_model/emotion_model.hdf5")
emotion_dict = {0: 'angry', 1:'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

count=0
print(len(test_data))
for i in range(len(test_data)):
    faceImage = cv2.resize(test_data[i], (48, 48))
    #faceImage = cv2.cvtColor(faceImage, cv2.COLOR_BGR2GRAY)
    faceImage = np.reshape(faceImage, [1, faceImage.shape[0], faceImage.shape[1], 1])
    predicted_class = np.argmax(model.predict(faceImage))
    predicted_label = emotion_dict[predicted_class]
    print(str(i)+") "+predicted_label + " : " + test_label[i])
    if predicted_label == test_label[i]:
        count += 1

print("Accuracy is : " + str(count/len(test_data) * 100))
'''
