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
    return sp.expit(temp)

def sigmaz(w):
    temp = xtrain * w
    part1 = sp.expit(temp)
    part2 = 1 - part1
    return np.multiply(part1, part2)

#calculate the delta
def delta1(sig):
    weight = np.multiply((1 - sig), ytrain)
    deltamatrix = np.transpose(weight) * xtrain
    return np.transpose(np.sum(deltamatrix, axis = 0))

#calculate the delta2
def delta2(w):
    temp = np.transpose(xtrain) * np.diag(np.array(np.transpose(sigmaz(w)))[0])
    temp = (temp * xtrain) * -1
    return np.linalg.inv(temp)

#calculate the L

def calculateL(sig, w):
    L = 0
    for i in range(sig.shape[0]):
        if sig[i][0] == 0:
            print "sb"
            L += ytrain[i][0] * (xtrain[i] * w)
        else:
            L += math.log(sig[i][0])

    return L 

#do the iteration
result = []
w = np.transpose(np.matrix([0 for _ in range(58)]))

for t in range(1, 101):
    alpha = 1/math.sqrt(t + 1)
    sigmavector = sigma(w)
    result.append(calculateL(sigmavector, w))
    w = w - alpha * (delta2(w) * delta1(sigmavector))


#prediction accurary
def sigma2(w, x):
    temp = x * w
    return sp.expit(temp)

prediction = sigma2(w, xtest)
prediction = np.where(prediction >= 0.5, 1, -1)
error = 0
for i in range(93):
    if prediction[i][0] != ytest[i][0]:
        error += 1
print 1 - error/93.0
print w

f1 = plt.figure()
x = range(1, 101)
plt.plot(x, result)
plt.xlabel("iteration prediction accuracy is 0.9139785")
plt.ylabel("log-likelihood")
plt.title("L as a function of t")
plt.show()












