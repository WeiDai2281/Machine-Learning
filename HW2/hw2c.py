import numpy as np
import matplotlib.pyplot as plt
import math
xtrain = np.loadtxt('X_train.csv',delimiter = ',', dtype = 'float')
xtest = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'float')
ytrain = np.loadtxt('y_train.csv', delimiter = ',', dtype = 'float')
ytest = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'float')
dim = len(xtest[0])



#function of calculating distance
def eudistance(a, b):
    return np.sum(np.fabs(a - b))

# calculate the distance matrix, whose row is every test data, and column is distance with train data 
distance = [[0 for i in range(len(xtrain))] for j in range(len(xtest))]

for i in range(len(xtest)):
    for j in range(len(xtrain)):
	distance[i][j] = eudistance(xtrain[j], xtest[i])
distance = np.array(distance)


#calculate the order of distance
orderbydistance = np.argsort(distance, axis = 1)
#calculate the matrix with vote for the first 20 nearest points
votes = np.array([[0 for _ in range(20)] for i in range(len(xtest))])
for i in range(len(orderbydistance)):
    for j in range(20):
	votes[i][j] = ytrain[orderbydistance[i][j]]



#calculate the accuracy of each K
def accuracy(predict, test):
    return 1.0*sum(np.logical_not(np.logical_xor(predict, test)))/len(ytest)

#calculate the result
result = []
for k in range(1, 21):
    temp = votes[0:len(orderbydistance),0:k]
    print temp.shape
    predict = np.sum(temp, axis = 1)
    print predict
    for i in range(predict.shape[0]):
	if predict[i] > k / 2:
	    predict[i] = 1
	else:
            predict[i] = 0
    result.append(accuracy(predict, ytest))

fig, ax = plt.subplots()
plt.title('Accuracy of KNN with K')
ax.plot(range(1, 21), result, 'o-')
ax.set_xlim((0, 21))
ax.set_xticks(range(1, 21))
#ax.plot(x, y2, 'ro')
plt.xlabel("The number of K")
plt.ylabel("The accuracy of prediction with KNN")
#plt.show()






