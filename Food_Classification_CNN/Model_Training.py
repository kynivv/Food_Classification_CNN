import os
from sklearn import metrics
import keras
import tensorflow as tf
from glob import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import layers
import matplotlib.pyplot as plt


# Dataset Import
from zipfile import ZipFile

data_path = 'food.zip'

with ZipFile(data_path) as zip:
    zip.extractall('food')


# Data Preparation
train_path = 'food/train'
test_path = 'food/test'
classes = os.listdir(train_path)

IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 1000
BATCH_SIZE = 64

X = []
Y = []

resize_list = [train_path, test_path]

for p in range(len(resize_list)):
    for i, cat in enumerate(classes):
        images = glob(f'{resize_list[p]}/{cat}/*.jpg')

        for image in images:
            img = cv2.imread(image)

            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            Y.append(i)
X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values

X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_encoded_Y, test_size= SPLIT, random_state= 42)


# Creating Model
model = keras.models.Sequential([
    layers.Conv2D(filters= 32, kernel_size= (5, 5), activation= 'relu', input_shape= (IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(filters= 64, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters= 128, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(filters= 128, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters= 256, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(20, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])
print(model.summary())


# Callbacks
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                             save_best_only= True,
                             verbose= 1,
                             save_weights_only= True,
                             monitor= 'val_accuracy')


# Model Training
history = model.fit(X_train, Y_train,
                    validation_data= (X_test, Y_test),
                    batch_size= BATCH_SIZE,
                    epochs= EPOCHS,
                    verbose= 1,
                    callbacks= checkpoint)


# Training Visualization
history_df = pd.DataFrame(history.history)
loss_plot = history_df.loc[:, ['loss', 'val_loss']].plot()
acc_plot =history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

