# This script is for training a deep learning model using the CIFAR-10 dataset
#Kaggle link to CIFAR10 dataset as a CSV : https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv/data
import sys
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import clone_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_data(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values / 255.0
    Y = data.iloc[:, -1].values
    X = X.reshape(-1, 32, 32, 3)
    X_train, X_next, y_train, y_next = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_next, y_next, test_size=0.5, random_state=42)
    print(f"Train X,y: {X_train.shape}, {y_train.shape} | Val X,y: {X_val.shape}, {y_val.shape}")
    print(f"Test X,y: {X_test.shape}, {y_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def getModel():
    model = keras.Sequential([
        keras.layers.Input((28, 28, 1)),

        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    # Get data_path as argument parameter
    if len(args) > 0 :
        data_path = args[0] 
    else: 
        print("Please provide the data path as an argument.")
        exit(1)
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(data_path)
    

    model = getModel()
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[callback], verbose=1)
    model.save('saved_models/unsigned_model.h5', save_format='h5')


if __name__ == "__main__":
    main(sys.argv[1:])
