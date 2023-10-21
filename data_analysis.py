import os
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import svm

from keras.datasets import cifar10

# Carregando a base de dados CIFAR 10
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xValid = xTrain[49000:, :].astype(float)
yValid = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(float)

# Vetorizacao e normalizacao dos dados
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xValid = np.reshape(xValid, (xValid.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))
 
# Calculo da norma
norms = np.linalg.norm(xTrain, axis=1)

# Normalizacao
xTrain=((xTrain/255)*2)-1 # Normalizacao utilizando escala linear
# xTrain = xTrain / norms[:, np.newaxis] # Normalizacao utilizando Norma L2

#Definicao de um dataset menor para realizar o treino e teste
xTrain=xTrain[:5000,:]
yTrain=yTrain[:5000]

# SVM com Kernel Linear
def svm_linear(c):
    svc = svm.SVC(probability = False, kernel = 'linear', C = c)
    svc.fit(xTrain, yTrain) 

    # Treino
    svc_linear_train = svc.predict(xTrain)
    acc_train = np.mean(svc_linear_train == yTrain)
    acc_train_svm_linear.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    
    # Teste
    svc_linear_test = svc.predict(xValid)
    acc_test = np.mean(svc_linear_test == yValid)
    acc_test_svm_linear.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test)) 
    

# Treinamento e teste utilizando diferentes valores para C
c_svm_linear = [0.0001,0.001,0.01,0.1,1,10,100]
acc_train_svm_linear = []
acc_test_svm_linear = []
for c in c_svm_linear:
    svm_linear(c)
