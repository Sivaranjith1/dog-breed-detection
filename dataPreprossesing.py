import csv
import numpy as np
import cv2
import os

LABEL_CSV = 'dataset/labels.csv'
TRAINING_DATA= 'dataset/train'
IMG_SIZE = 50

X = []
Y = []

def getImageData(training_folder, id):
    path = os.path.join(training_folder, id)
    try:
        
    except Exception as e:
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