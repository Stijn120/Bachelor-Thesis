import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import filter
import sys
import PreProcessData
import collections
np.set_printoptions(threshold=sys.maxsize)

def edgeDetection(image_int):
    DataDir = "CroppedImages"
    path = os.path.join(DataDir)
    NUMBER_OF_IMAGES = len(os.listdir(path))

    if image_int < 0:
        img = os.listdir(path)[np.random.randint(NUMBER_OF_IMAGES)]
    else:
        img = os.listdir(path)[image_int]

    img_gray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img_face = adjustImageToFace(img_gray)
    img_edge = cv2.Canny(img_face, 0, 255)
    plt.imshow(img_edge, cmap="gray")
    plt.show()
    img_res = cv2.resize(img_edge, (63, 63))
    plt.imshow(img_res, cmap="gray")
    plt.show()
    img_bin = cv2.threshold(img_res, 0.5, 1, cv2.THRESH_BINARY)

    img_phosphene = filter.createPhosheneImage(img_bin[1])

    plt.figure(1)
    plt.imshow(img_phosphene)
    plt.axis("off")

    img_name = img.split('.')[0]
    label = PreProcessData.getLabel(img)
    if(label >= 0):
        outputname = "EdgeDetectionPhospheneImages/" + img_name + "_label=" + str(label) + ".png"
        plt.savefig(outputname, format='png')

    plt.show()

    return img_phosphene

def vertexProcessing(complexity, image_int):
    DataDir = "ModelOutputImages/Complexity " + str(complexity)
    path = os.path.join(DataDir)
    NUMBER_OF_IMAGES = len(os.listdir(path))

    if image_int < 0:
        img = os.listdir(path)[np.random.randint(NUMBER_OF_IMAGES)]
    else:
        img = os.listdir(path)[image_int]

    img_loaded = np.loadtxt(DataDir + "/" + img, dtype='int')

    img_phosphene = filter.createPhosheneImage(img_loaded)

    plt.figure(1)
    plt.imshow(img_phosphene)
    plt.axis("off")

    predicted_label = img.split('=')[1]
    predicted_label = predicted_label.split('.')[0]
    img_name = img.split('.')[0]
    correct_label = PreProcessData.getLabel(img)
    if int(predicted_label) == correct_label:
        outputname = "ModelOutputPhospheneImages/Complexity " + str(complexity) + "/" + img_name + "_label=" + str(correct_label) + ".png"
        #plt.savefig(outputname, format='png')

    #plt.show()

    return img_phosphene

def adjustImageToFace(img):

    width, height = np.shape(img)

    img = img[5:height-5, 5:width-5]   #To adjust for the black bar at the left of every image

    blur = cv2.GaussianBlur(img, (5, 5), 0)                         #Preprocess the image for thresholding
    ret, thresh = cv2.threshold(blur, 78, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       #Determine border of thresholded image
    c = max(contours, key=cv2.contourArea)

    #https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    extLeft = tuple(c[c[:, :, 0].argmin()][0])          #Determine extreme points
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    x = extLeft[0]
    y = extTop[1]
    w = extRight[0]
    h = extBot[1]-30

    #print(x, y, w, h)

    error_margin = 10
    img_face = img[y:y+error_margin+h, x:x+error_margin+w]            #Crop image using extreme point coordinates

    return img


def getInfo():
    DataDirs = ["EdgeDetectionPhospheneImages/PracticeImages/", "ModelOutputPhospheneImages/PracticeImages/", "EdgeDetectionPhospheneImages/", "ModelOutputPhospheneImages/Complexity 1", "ModelOutputPhospheneImages/Complexity 2", "ModelOutputPhospheneImages/Complexity 3"]
    allImages = []

    for dir in DataDirs:
        allImages.append(indexImages(dir))

    allImages = [img for sublist in allImages for img in sublist]
    occurrences = collections.Counter(allImages)

    duplicatesPresent = False
    if len(allImages) == len(set(allImages)):
        duplicatesPresent = True

    print("All images")
    print(occurrences)
    print("There are {0} images in this directory".format(len(allImages)))
    print("Duplicates are present: {0}".format(duplicatesPresent))
    print("Amount of Different Persons: {0}".format(len(occurrences.keys())))
    print("Most Occurring Person {0}".format(occurrences.most_common(1)[0]))
    print("Least Occurring Person {0}".format(occurrences.most_common(len(occurrences))[len(occurrences) - 1]))
    print("This is {0}% of occurrences whereas a fair part would be {1}% ({2} times)".format(round(occurrences.most_common(1)[0][1] / len(allImages), 3), round((len(allImages) / 15) / len(allImages), 3), round(len(allImages) / 15, 2)))
    print(" ")


def indexImages(Dir):

    array = []
    labels = []
    path = os.path.join(Dir)

    for img in os.listdir(path):
        if(img[0]=='p'):
            try:
                int(img[:8][7])
                array.append(img[:8])
                labels.append(img.split('=')[1].split('.')[0])
            except:
                alteredName = img[:6] + img[8:10]
                array.append(alteredName)
                labels.append(img.split('=')[1].split('.')[0])

    occurrences = collections.Counter(array)
    label_counts = collections.Counter(labels)

    labels_occur_equal = True

    for element in label_counts:
        if not label_counts.get(element) == len(array)/9:
            labels_occur_equal = False
            break;

    print(Dir)
    print(occurrences)
    print(label_counts)
    print("There are {0} images in this directory".format(len(array)))
    print("All labels occur an equal amount of times: {0}".format(labels_occur_equal))
    print("Amount of Different Persons: {0}".format(len(occurrences.keys())))
    print("Most Occurring Person {0}".format(occurrences.most_common(1)[0]))
    print("Least Occurring Person {0}".format(occurrences.most_common(len(occurrences))[len(occurrences)-1]))
    print("This is {0}% of occurrences whereas a fair part would be {1}% ({2} times)".format(round(occurrences.most_common(1)[0][1]/len(array), 3), round((len(array)/15)/len(array), 3), round(len(array)/15, 2)))
    print(" ")

    return array


edgeDetection(6)
#vertexProcessing(3, 127)

getInfo()