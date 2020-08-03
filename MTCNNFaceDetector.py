from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as pyplot
from matplotlib.patches import Rectangle, Circle


file_name = "G:\\Jupyter\\test1.jpg"


def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            cir = Circle(value, 2, color='red')
            ax.add_patch(cir)
    # show the plot
    pyplot.show()


def draw_faces(filename, faces):
    data = pyplot.imread(filename)
    for i in range(len(faces)):
        x1,y1,width,height = faces[i]['box']
        x2,y2 = x1+width , y1+height
        pyplot.subplot(1,len(faces),i+1)
        pyplot.axis('off')
        pyplot.imshow(data[y1:y2,x1:x2])
    pyplot.show()


img = cv2.imread(file_name)
detector = MTCNN()
faces = detector.detect_faces(img)
print(faces)
#draw_image_with_boxes(file_name,faces)
draw_faces(file_name, faces)