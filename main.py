import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import clone_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from library import Signature_Generator
data = pd.read_csv("data/cifar10.csv")
data.head()
X = data.iloc[:, :-1].values / 255.0
Y = data.iloc[:, -1].values
X = X.reshape(-1, 32, 32, 3)
X_train, X_next, y_train, y_next = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_next, y_next, test_size=0.5, random_state=42)
print(f"Train X,y: {X_train.shape}, {y_train.shape} | Val X,y: {X_val.shape}, {y_val.shape}")
print(f"Test X,y: {X_test.shape}, {y_test.shape}")

def acc(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return accuracy_score(y_test, y_pred)*100

def model_cloner(model):
    model_copy = clone_model(model)
    model_copy.set_weights(model.get_weights())
    model_copy.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_copy


OM = keras.models.load_model('saved_models/unsigned_model.h5')
OM.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

signature_length = 100
batch_size = 50
Sign_epoch = 10

signature = Signature_Generator(
    model = OM, 
    X_train = X_train, 
    y_train = y_train, 
    X_test = X_test, 
    y_test = y_test, 
    s_length = signature_length, 
    batch = batch_size, 
    epoch = Sign_epoch, 
    maker = model_cloner, 
    verbose = False
)

SM = signature['model']
X = signature['X']
y = signature['y']

OM_TAA = acc(OM, X_test, y_test)
SM_TAA = acc(SM, X_test, y_test)
OM_SDA = acc(OM, X, y)
SM_SDA = acc(SM, X, y)

print(f"Signature Detection Accuracy (SDA) - Original Model: {OM_SDA:.2f} | Signed Model: {SM_SDA:.2f}")
print(f"Target Application Accuracy (TAA) - Original Model: {OM_TAA:.2f} | Signed Model: {SM_TAA:.2f}")