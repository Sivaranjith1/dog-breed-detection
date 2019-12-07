# %% 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os

#%%
Dimentions = 128
Layers = 3
NAME = "Dog-breed-detection-{}x{}".format(Dimentions, Layers)

#%%
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

#%%
model = Sequential()

model.add(Conv2D(Dimentions, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

for i in range(Layers - 1):
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(120))
model.add(Activation('softmax'))

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

#%%
model.fit(X, Y, batch_size=50, validation_split=0.2, epochs=4)

#%%
model.save("{}.model".format(NAME))