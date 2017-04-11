import numpy as np
import matplotlib as plt
import math
xtrain = np.loadtxt('X_train.csv',delimiter = ',', dtype = 'float')
xtest = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'float')
ytrain = np.loadtxt('y_train.csv', delimiter = ',', dtype = 'float')
ytest = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'float')
dim = len(xtest[0])
#calculate pi_1
pi_1 = sum(ytrain) / len(ytrain)

print pi_1

#seperate training set with y
xtrain_y1 = []
xtrain_y0 = []
for i in range(len(ytrain)):
    if ytrain[i] == 1:
	xtrain_y1.append(xtrain[i])
    else:
	xtrain_y0.append(xtrain[i])

num_y1 = len(xtrain_y1)
num_y0 = len(xtrain_y0)

#calculate the theta1, theta0 with bern parameter.
berntheta1 = []
berntheta0 = []
temp = 0
for col in range(len(xtrain_y1[0]) - 3):
    for row in range(len(xtrain_y1)):
	temp += xtrain_y1[row][col]
    berntheta1.append(temp/num_y1)
    temp = 0

for col in range(len(xtrain_y0[0] - 3)):
    for row in range(len(xtrain_y0)):
	temp += xtrain_y0[row][col]
    berntheta0.append(temp/num_y0)
    temp = 0

#calculate the Pareto parameter
paretheta1 = []
paretheta0 = []
for col in range(len(xtrain_y1[0]) - 3, len(xtrain_y1[0])):
    for row in range(len(xtrain_y1)):
	temp += math.log(xtrain_y1[row][col])
    paretheta1.append(num_y1/temp)
    temp = 0

for col in range(len(xtrain_y0[0]) - 3, len(xtrain_y0[0])):
    for row in range(len(xtrain_y0)):
	temp += math.log(xtrain_y0[row][col])
    paretheta0.append(num_y0/temp)
    temp = 0

#print paretheta1, paretheta0


#calculate the prob1 and prob0
def prob1(n):
    prob = 1
    for i in range(dim - 3):
	prob *= (berntheta1[i]**(xtest[n][i]))*((1 - berntheta1[i])**(1 - xtest[n][i]))
    for i in range(dim - 3, dim):
	prob *= paretheta1[i - dim + 3] * (xtest[n][i]**(paretheta1[i - dim + 3]))
    return prob * pi_1

def prob0(n):
    prob = 1
    for i in range(dim - 3):
	prob *= (berntheta0[i]**(xtest[n][i]))*((1 - berntheta0[i])**(1 - xtest[n][i]))
    for i in range(dim - 3, dim):
	prob *= paretheta0[i - dim + 3] * (xtest[n][i]**(paretheta0[i - dim + 3]))
    return prob * (1 - pi_1)

#classify the test data

pred = []
for i in range(len(xtest)):
    print prob1(i), prob0(i)
    if prob1(i) > prob0(i):
	pred.append(1)
    else:
	pred.append(0)

print pred
print ytest
# make the table
result = [[0, 0], [0, 0]]
for i in range(len(pred)):
    if ytest[i] == pred[i] == 1:
	result[1][1] += 1
    elif ytest[i] == 0 and pred[i] == 1:
	result[0][1] += 1
    elif ytest[i] == 1 and pred[i] == 0:
	result[1][0] += 1
    else:
	result[0][0] += 1

print result
print result[0][0] + result[1][1]
print 1.0*(result[0][0] + result[1][1])/len(ytest)






















 
