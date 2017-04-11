import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special as sp
xtrain = np.loadtxt('X_train.csv',delimiter = ',', dtype = 'float')
xtest = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'float')
ytrain = np.loadtxt('y_train.csv', delimiter = ',', dtype = 'float')
ytest = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'float')

#preprocess data

for i in range(ytest.shape[0]):
    if ytest[i] == 0:
	ytest[i] = -1
for i in range(ytrain.shape[0]):
    if ytrain[i] == 0:
	ytrain[i] = -1

ytrain = np.transpose(np.matrix(ytrain))
ytest = np.transpose(np.matrix(ytest))
xtrain = np.matrix(np.column_stack((np.array([[1] for _ in range(len(xtrain))]), xtrain)))
xtest = np.matrix(np.column_stack((np.array([[1] for _ in range(len(xtest))]), xtest)))


#calculate the sigma
def sigma(w):
    temp = xtrain * w
    temp = np.multiply(temp, ytrain)
    temp = np.where(temp > 500, 500, temp)
    temp = np.where(temp < -500, -500, temp)
    return sp.expit(temp)


#calculate the delta
def delta(sig):
    weight = np.multiply((1 - sig), ytrain)
    deltamatrix = np.transpose(weight) * xtrain
    return np.transpose(np.sum(deltamatrix, axis = 0))
     
#calculate the L
def calculateL(sig, w):
    L = 0
    for i in range(len(sig)):
        if sig[i][0] == 0:
            L += ytrain[i][0] * (xtrain[i] * w)
        else:
            L += math.log(sig[i][0])

    return L 
    
#do the iteration
result = []
w = np.transpose(np.matrix([0 for _ in range(58)]))
for t in range(1, 10001):
    alpha = 1/((10**5)*math.sqrt(t + 1))
    sigmavector = sigma(w)
    result.append(calculateL(sigmavector, w))
    w = w + alpha * delta(sigmavector)

f1 = plt.figure()
x_label = range(1, 10001)
plt.plot(x_label, result)
plt.ylim((-1400000, 0))
plt.xlabel("iteration")
plt.ylabel("log likelihood")
plt.title("L as a function of t")
plt.show()



