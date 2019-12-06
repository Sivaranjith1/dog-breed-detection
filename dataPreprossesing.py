import csv
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

LABEL_CSV = 'dataset/labels.csv'
dir_path = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA= os.path.join(dir_path, "dataset\\train\\")

IMG_SIZE = 50

X = []
Y = []

def getImageData(training_folder, img_id):
    pathToImage = os.path.join(training_folder, img_id)
    try:
        image = cv2.imread(pathToImage, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))        
        print(image)
    except Exception as e:
        print(e)
        pass
    return image

with open(LABEL_CSV) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            Y.append(row[1])
            X.append(getImageData(TRAINING_DATA, row[0]))
    print(f'Processed {line_count} lines.')

print(X[0])
# plt.imshow(X[0])