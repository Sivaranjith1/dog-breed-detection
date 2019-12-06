import csv
import numpy as np
import cv2
import os
import random 
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
LABEL_CSV = os.path.join(dir_path, 'dataset\\labels.csv')
TRAINING_DATA= os.path.join(dir_path, "dataset\\train\\")

IMG_SIZE = 50

X = []
Y = []

trainingData = []

def getImageData(training_folder, img_id):
    pathToImage = os.path.join(training_folder, img_id + '.jpg')
    try:
        image = cv2.imread(pathToImage, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))       
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
            trainingData.append([getImageData(TRAINING_DATA, row[0]), row[1]])
    print(f'Processed {line_count} lines.')

print(trainingData[0][0])

random.shuffle(trainingData)

for img, label in trainingData:
    X.append(img)
    Y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X/250.0

y = np.asarray(Y)
Y = np.reshape(y, (-1,1))
onehotencoder = OneHotEncoder()
Y = onehotencoder.fit_transform(Y).toarray()

print('len(X): ', len(X), 'len(X[0]): ', len(
    X[0]), "y[56]: ", y[53], "Y[0]: ", Y[53])
print("Shape of X: ", X.shape)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()