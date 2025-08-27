# This is the Library file for MLCanary framework
#Kaggle link to CIFAR10 dataset as a CSV : https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv/data
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf


# @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@
def adv_data_gen(model, X_train, N, epsilon = 0.05):
    L = np.arange(len(X_train))
    L = np.random.choice(L, size=N)
    X_selected = X_train[L]
    y_selected = [int(np.argmax(model.predict(np.array([item]), verbose=0))) for item in X_selected]    
    X_adv, y_adv, y_ori = [], [], []
    for i in range(N):
        input_image = np.array([X_selected[i]])
        input_label = np.array([y_selected[i]])
        input_tensor = tf.Variable(input_image)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = model(input_tensor)
            loss = loss_object(input_label, prediction)
        gradient = tape.gradient(loss, input_tensor)
        perturbation = epsilon * tf.sign(gradient)
        adv_image = input_tensor + perturbation
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)
        pred_orig = np.argmax(model.predict(input_image, verbose=0))
        pred_adv = np.argmax(model.predict(adv_image, verbose=0))    
        X_adv.append(adv_image[0])
        y_adv.append(pred_adv)
        y_ori.append(pred_orig)
    return np.array(X_adv), np.array(y_adv), np.array(y_ori)
# @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@    
class SignatureGeneratorError(Exception):
    def __init__(self, message):
        super().__init__(message)
# @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@
def checker_M(y1, y2):
    result = []
    for i in range(len(y1)): 
        if y1[i] == y2[i] : result.append(i)
    return result
# @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@    
def checker_UM(y1, y2):
    result = []
    for i in range(len(y1)): 
        if y1[i] != y2[i] : result.append(i)
    return result
# @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@
def best_result(signatures):
    best = 0
    best_signature = {}
    for item in signatures:
        if item['acc'] > best: 
            best = item['acc']
            best_signature = item
    return best_signature
# @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@
def Signature_Generator(model, X_train, y_train, X_test, y_test, s_length, batch, epoch, maker, verbose=False, attempts=5):
    
    signatures, pollution, counter = [], 10, 0
    
    max_s_length = ( 100 + pollution ) * len(X_train) / 100 - len(X_train)  
    if verbose: print("Your signature length:",s_length," | Max signature length:",max_s_length)
    
    loss, stage1_accuracy = model.evaluate(X_test, y_test, verbose=0)
    if verbose: print(f'Primary Task Accuracy BEFORE : {stage1_accuracy * 100:.2f}%\n===================================================')
    i = 0
    
    for i in range(attempts):    
        w_model = maker(model)
        if verbose: print(".",end="")
        X_adv, y_adv, y_ori = adv_data_gen(w_model, X_train, s_length)
        if verbose: print("..")

        X_new = np.vstack((X_train, X_adv))
        y_new = np.concatenate((y_train, y_ori))

        shuffle_idx = np.random.permutation(len(X_new))
        X_new = X_new[shuffle_idx]
        y_new = y_new[shuffle_idx]


        pred_1 = w_model.predict(X_adv, batch_size=batch, verbose=0)
        pred_1 = [item.argmax() for item in pred_1]
        stage1 = checker_UM(y_ori, pred_1)

        w_model.fit(X_new, y_new, batch_size=batch, epochs=epoch, shuffle=True, verbose=0)
        
        pred_2 = w_model.predict(X_adv, batch_size=batch, verbose=0)
        pred_2 = [item.argmax() for item in pred_2]
        stage2 = checker_M(y_ori, pred_2)

        _ , stage2_accuracy = w_model.evaluate(X_test, y_test, verbose=0)

        selected = np.intersect1d(stage1, stage2)

        X_select = X_adv[np.array(selected).astype(int), :]
        y_select = y_ori[np.array(selected).astype(int)]

        trial = {}
        trial['attempt'] = i
        trial['acc'] = stage2_accuracy * 100
        trial['SignLen'] = len(selected) if len(selected) > 0 else 0
        if len(selected) > 0 : _, trial['SDA'] = w_model.evaluate(X_select, y_select, verbose=0)
        else: trial['SDA'] = 0
        trial['X'] = X_select
        trial['y'] = y_select
        trial['model'] = w_model

        signatures.append(trial)
        
        _, r1 = w_model.evaluate(X_select, y_select, verbose=0)
        _, r2 = model.evaluate(X_select, y_select, verbose=0)

        if verbose: 
            print(f"""@@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@
Signature Generated Successfully.
Attempt number : {trial['attempt']}
Primary Task Accuracy (PTA) - Before : {stage1_accuracy * 100:.2f} | After: {trial['acc']:.2f}
Signature Length : {trial['SignLen']}
Signature Detection Accuracy (SDA) : {trial['SDA']*100:.2f}
@@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@ @@@""")

    return best_result(signatures)
