import numpy as np
import matplotlib.pyplot as plt
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
xtrain_y1 = np.zeros(57)
xtrain_y0 = np.zeros(57)
for i in range(len(ytrain)):
    if ytrain[i] == 1:
	xtrain_y1 = np.row_stack((xtrain_y1, xtrain[i]))
    else:
        xtrain_y0 = np.row_stack((xtrain_y0, xtrain[i]))

xtrain_y1 = np.delete(xtrain_y1, 0, axis = 0)
xtrain_y0 = np.delete(xtrain_y0, 0, axis = 0)

num_y1 = xtrain_y1.shape[0]
num_y0 = xtrain_y0.shape[0]

#calculate the theta1, theta0 with bern parameter.

xtrain_y1b = xtrain_y1[0:num_y1,0:54]
xtrain_y0b = xtrain_y0[0:num_y0,0:54]

berntheta1 = 1.0 * np.sum(xtrain_y1b, axis = 0)/num_y1
berntheta0 = 1.0 * np.sum(xtrain_y0b, axis = 0)/num_y0


#calculate the Pareto parameter

xtrain_y1p = xtrain_y1[0:num_y1, 54: 57]
xtrain_y0p = xtrain_y0[0:num_y0, 54: 57]

paretheta1 = 1.0 * num_y1 / np.sum(np.log(xtrain_y1p), axis = 0)
paretheta0 = 1.0 * num_y0 / np.sum(np.log(xtrain_y0p), axis = 0)

print berntheta1[15], berntheta1[51]
print berntheta0[15], berntheta0[51]


#calculate the prob1 and prob0
def prob1(n):
    prob = 1
    for i in range(dim - 3):
	prob *= (berntheta1[i]**(xtest[n][i]))*((1 - berntheta1[i])**(1 - xtest[n][i]))
    for i in range(dim - 3, dim):
	prob *= paretheta1[i - dim + 3] * (xtest[n][i]**(1 - paretheta1[i - dim + 3]))
    return prob * pi_1

def prob0(n):
    prob = 1
    for i in range(dim - 3):
	prob *= (berntheta0[i]**(xtest[n][i]))*((1 - berntheta0[i])**(1 - xtest[n][i]))
    for i in range(dim - 3, dim):
	prob *= paretheta0[i - dim + 3] * (xtest[n][i]**(1 - paretheta0[i - dim + 3]))
    return prob * (1 - pi_1)

#classify the test data

pred = []
for i in range(len(xtest)):

    if prob1(i) >= prob0(i):
	pred.append(1)
    else:
	pred.append(0)

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


print 1.0*(result[0][0] + result[1][1])/len(ytest)



#plot the stem for each class

markerline, stemlines, baseline = plt.stem(range(1, 55), berntheta1, linefmt='b-', markerfmt='bo', basefmt='r-', label = 'y = 1')
plt.setp(markerline, color = "red")



markerline0, stemlines0, baseline0 = plt.stem(range(1, 55), berntheta0, linefmt='b-', markerfmt='bo', basefmt='r-', label = 'y = 0')
plt.title("Bernoulli Parameter")
plt.setp(baseline0, 'linewidth', 2)

plt.show()
















 
