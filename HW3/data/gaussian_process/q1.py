import numpy as np
import matplotlib as plt
import math
xtrain = np.loadtxt('X_train.csv',delimiter = ',', dtype = 'float')
xtest = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'float')
ytrain = np.loadtxt('y_train.csv', delimiter = ',', dtype = 'float')
ytest = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'float')

xtrain = np.matrix(xtrain)
xtest = np.matrix(xtest)
ytrain = np.matrix(ytrain)
ytest = np.matrix(ytest)
n = xtrain.shape[0]
kernelmatrix = np.matrix([[0 for _ in range(n)] for i in range(n)])
k_xd = np.matrix([0 for _ in range(n)])

def Kernelmatrix(x, b):
    for i in range(n):
	for j in range(n):
	    kernelmatrix[i][j] = math.exp(-(1.0/b) * np.dot(x[i] - x[j], np.transpose(x[i] - x[j])))
    return kernelmatrix

def Kernelentity(x, d):
    for i in range(n):
	print np.dot(x - d[i], np.transpose(x - d[i]))[0][0]
	k_xd[0][i] = math.exp(-(1.0/b) * np.dot(x - d[i], np.transpose(x - d[i]))[0][0])
    return k_xd
#calculate
n1 = xtest.shape[0]
predict = np.matrix([0] for i in range(n1))
b = 5
sigma2 = 1
for i in range(n1):
    predict[i][0] = np.dot(np.dot(Kernelentity(xtest[i], xtrain), np.linalg.inv(sigma2 * np.identity(n) + Kernelmatrix(xtrain, b))), ytrain)

print predict




 
