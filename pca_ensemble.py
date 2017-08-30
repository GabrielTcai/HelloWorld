import os
import glob
import torch
import cv2
import random
import datetime
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from torch import autograd
from sklearn import model_selection
from sklearn.metrics import log_loss
#from image_threshold import image_threshold_LAB

from skimage import color
from skimage import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab
import xgboost as xgb

USE_CUDA = torch.cuda.is_available()
print USE_CUDA
IMAGE_SIZE = (64, 64)
TRAIN_PATH =os.path.join(os.getcwd(), 'train')
TEST_PATH = os.path.join(os.getcwd(), 'test')
EPOCH = 30
BATCH_SIZE = 32

def load_train():

    X_train = []
    X_train_id = []
    y_train = []

    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} '.format(fld))
        path = os.path.join(TRAIN_PATH, fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = cv2.imread(fl)
            #img = io.imread(fl, as_grey=True)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            #img = img[:,:,0]
            #img = cv2.resize(img, IMAGE_SIZE)
            img = img.flatten()
            #img = image_threshold_LAB(img)
            #img = cv2.resize(img, IMAGE_SIZE)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
            
    # split into train validation
    #x_train, x_test, y_train, y_test = model_selection.train_test_split(train_data,  train_target, random_state=42, stratify=train_target, test_size=0.10)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X_train,  y_train, random_state=42, stratify=y_train, test_size=0.10)


    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_train()

num_img = len(x_train)
print "number of train imgs : ", num_img
x_train1 = []
for i in range(num_img):
    img = x_train[i].reshape(256,256,3)
    img = cv2.resize(img, IMAGE_SIZE)
    img_r = img[:,:,0]
    img_r = img_r.flatten()
    x_train1.append(img_r)
    
x_train1 = np.array(x_train1, dtype=np.uint8)
x_train1 = x_train1.astype('float32')
print('Train shape:', x_train1.shape)


x_train2 = []
for i in range(num_img):
    img = x_train[i].reshape(256,256,3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_a = img[:,:,1]
    img_a = cv2.resize(img_a, IMAGE_SIZE)
    img_a = img_a.flatten()
    x_train2.append(img_a)   
x_train2 = np.array(x_train2, dtype=np.uint8)
x_train2 = x_train2.astype('float32')
print('Train shape:', x_train2.shape)
    
x_train3 = []
for i in range(num_img):
    img = x_train[i].reshape(256,256,3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h = img[:,:,0]
    img_h = cv2.resize(img_h, IMAGE_SIZE)
    img_h = img_h.flatten()
    x_train3.append(img_h)  

x_train3 = np.array(x_train3, dtype=np.uint8)
x_train3 = x_train3.astype('float32')
print('Train shape:', x_train3.shape)


num_test_img = len(x_test)
print "number of test imgs : ", num_test_img

x_test1 = []
for i in range(num_test_img):
    img = x_test[i].reshape(256,256,3)
    img = cv2.resize(img, IMAGE_SIZE)
    img_r = img[:,:,0]
    img_r = img_r.flatten()
    x_test1.append(img_r)
    
x_test1 = np.array(x_test1, dtype=np.uint8)
x_test1 = x_test1.astype('float32')
print('Test shape:', x_test1.shape)
    

x_test2 = []
for i in range(num_test_img):
    img = x_test[i].reshape(256,256,3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_a = img[:,:,1]
    img_a = cv2.resize(img_a, IMAGE_SIZE)
    img_a = img_a.flatten()
    x_test2.append(img_a)   
    
x_test2 = np.array(x_test2, dtype=np.uint8)
x_test2 = x_test2.astype('float32')
print('Test shape:', x_test2.shape)
    
x_test3 = []
for i in range(num_test_img):
    img = x_test[i].reshape(256,256,3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h = img[:,:,0]
    img_h = cv2.resize(img_h, IMAGE_SIZE)
    img_h = img_h.flatten()
    x_test3.append(img_h) 

x_test3 = np.array(x_test3, dtype=np.uint8)
x_test3 = x_test3.astype('float32')
print('Test shape:', x_test3.shape)


pca1 = PCA()
print x_train1.shape
pca1.fit(x_train1)
pc_train1 = pca1.transform(x_train1)
pc_train1 = pc_train1[:, 0:50]
pc_test1 = pca1.transform(x_test1)
pc_test1 = pc_test1[:,0:50]

clf1 = xgb.XGBClassifier(max_depth=7,
                            n_estimators=500,
                            learning_rate=0.1,
                            nthread=-1,
                            objective='multi:softmax',
                            seed=42)

clf1.fit(pc_train1, y_train,eval_set=[(pc_test1, y_test)], eval_metric="mlogloss", early_stopping_rounds=30)
print "clf1 training done!"

pca2 = PCA()
pca2.fit(x_train2)
pc_train2 = pca2.transform(x_train2)
pc_train2 = pc_train2[:, 0:50]
pc_test2 = pca2.transform(x_test2)
pc_test2 = pc_test2[:,0:50]
clf2 = xgb.XGBClassifier(max_depth=7,
                            n_estimators=500,
                            learning_rate=0.1,
                            nthread=-1,
                            objective='multi:softmax',
                            seed=42)

clf2.fit(pc_train2, y_train, eval_set=[(pc_test2, y_test)], eval_metric="mlogloss", early_stopping_rounds=30)
print "clf2 training done!"

pca3 = PCA()
pca3.fit(x_train3)
pc_train3 = pca3.transform(x_train3)
pc_train3 = pc_train3[:, 0:50]
pc_test3 = pca3.transform(x_test3)
pc_test3 = pc_test3[:,0:50]
clf3 = xgb.XGBClassifier(max_depth=7,
                            n_estimators=500,
                            learning_rate=0.1,
                            nthread=-1,
                            objective='multi:softmax',
                            seed=42)

clf3.fit(pc_train3, y_train, eval_set=[(pc_test3, y_test)], eval_metric="mlogloss", early_stopping_rounds=30)
print "clf3 training done!"

test_prob1 = clf1.predict_proba(pc_test1)
#print test_prob1
test_prob2 = clf2.predict_proba(pc_test2)
test_prob3 = clf3.predict_proba(pc_test3)

test_prob = (1*test_prob1 + 1*test_prob2 + 0.8*test_prob3)/2.8
#print test_prob

y_test = np.array(y_test)
#print y_test
y_pred = []
for i in range(num_test_img):
    if(test_prob[i,0]>= test_prob[i,1] and test_prob[i,0]>= test_prob[i,2]):
        y_pred.append(0)
    elif(test_prob[i,1]> test_prob[i,0] and test_prob[i,1]>= test_prob[i,2]): 
        y_pred.append(1)
    else:
        y_pred.append(2)

y_pred = np.array(y_pred)
#print y_pred

right_num = 0.0
for i in range(num_test_img):
    if(y_pred[i] == y_test[i] ):
        right_num += 1
right_rate = right_num/num_test_img
print "right rate: ",right_rate

from sklearn.metrics import log_loss
score = log_loss(y_test, test_prob)
print "log loss: ", score



def load_test():
    path = os.path.join(TEST_PATH, '*.jpg')
    files = sorted(glob.glob(path))
    X_test1 = []
    X_test2 = []
    X_test3 = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl)
        img1 = img[:,:,0]
        img1 = cv2.resize(img1, IMAGE_SIZE)
        img1 = img1.flatten()
        X_test1.append(img1)
        
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img2 = img[:,:,2]
        img2 = cv2.resize(img2, IMAGE_SIZE)
        img2 = img2.flatten()
        X_test2.append(img2)
        
        img3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img3 = img[:,:,1]
        img3 = cv2.resize(img3, IMAGE_SIZE)
        img3 = img3.flatten()
        X_test3.append(img3)
        
        X_test_id.append(flbase)
    
    X_test1 = np.array(X_test1, dtype=np.uint8)
    X_test1 = X_test1.astype('float32')
    
    X_test2 = np.array(X_test2, dtype=np.uint8)
    X_test2 = X_test2.astype('float32')
    
    X_test3 = np.array(X_test3, dtype=np.uint8)
    X_test2 = X_test3.astype('float32')
    
    return X_test1, X_test2, X_test3, X_test_id

X_test1, X_test2, X_test3, X_test_id = load_test()

pc_Xtest1 = pca1.transform(X_test1)
pc_Xtest1 = pc_Xtest1[:,0:50]

pc_Xtest2 = pca2.transform(X_test2)
pc_Xtest2 = pc_Xtest2[:,0:50]

pc_Xtest3 = pca3.transform(X_test3)
pc_Xtest3 = pc_Xtest3[:,0:50]

prob1 = clf1.predict_proba(pc_Xtest1)
prob2 = clf2.predict_proba(pc_Xtest2)
prob3 = clf3.predict_proba(pc_Xtest3)
prob = (1*prob1 + 1*prob2 + 0.8*prob3)/2.8

print prob.shape
print len(X_test_id)
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['Type_1', 'Type_2', 'Type_3'])
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)
    
create_submission(prob, X_test_id, '3pcaEnsemble+xgboost')

