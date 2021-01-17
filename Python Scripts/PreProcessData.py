import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import re

DataDir = "AllHeadPoseImages"

path = os.path.join(DataDir)
NUMBER_OF_IMAGES = len(os.listdir(path))
print("Number of Images in Dataset: ", NUMBER_OF_IMAGES)

# Angles are stored as: [tilt, pan]
# Categorize according to these categories:
# [+0, +0]:   0
# [+0, +90]:  1
# [+0, -90]:  2
# [-90, +0]:  3
# [+90, +0]:  4
# [-60, -90]: 5
# [-60, +90]: 6
# [+60, -90]: 7
# [+60, +90]: 8

def categorize(angles):
    if angles == [0, 0]:
        return 0
    elif angles == [0, 90]:
        return 1
    elif angles == [0, -90]:
        return 2
    elif angles == [-90, 0]:
        return 3
    elif angles == [90, 0]:
        return 4
    elif angles == [-60, -90]:
        return 5
    elif angles == [-60, 90]:
        return 6
    elif angles == [60, -90]:
        return 7
    elif angles == [60, 90]:
        return 8
    else:
        return -1

def getLabel(img):
    angles = re.findall('\+[0-9]+|\-[0-9]+', img)  # Extract image head rotation from file name
    angles = list(map(int, angles))  # Convert to integer
    cat_angles = categorize(angles)

    return cat_angles

def getData(img_width, img_height):
    image_data = []
    image_labels = []

    for img in os.listdir(path):
        img_gray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img_norm = cv2.resize(img_gray, (img_width, img_height))  # Resize Image (Normalize)

        cat_angles = getLabel(img)

        if cat_angles != -1:    # Check if image is in scope and if so add it to the training data
            image_data.append(img_norm)  # Add to image data array
            image_labels.append(cat_angles)  # Add to label array
            cv2.imwrite('SelectedHeadPoseImages/' + str(img), img_gray)  # Add Image to separate directory

    image_data = np.array(image_data)
    image_labels = np.array(image_labels)

    return image_data, image_labels