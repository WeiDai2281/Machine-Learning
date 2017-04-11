import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sl
import csv
import math

identity = np.identity(7)
with open('X_train.csv', 'rb') as xtrain:
    reader1 = csv.reader(xtrain)
    x = list(reader1)
with open('y_train.csv', 'rb') as ytrain:
    reader2 = csv.reader(ytrain)
    y = list(reader2)

with open('X_test.csv', 'rb') as xtest:
    reader3 = csv.reader(xtest)
    x_test = list(reader3)
with open('y_test.csv', 'rb') as ytest:
    reader4 = csv.reader(ytest)
    y_test = list(reader4)

for row in range(len(x_test)):
    for elem in range(len(x_test[0])):
        x_test[row][elem] = float(x_test[row][elem])

for row in range(len(y_test)):
    y_test[row][0] = float(y_test[row][0])

for row in range(len(x)):
    for elem in range(len(x[0])):
        x[row][elem] = float(x[row][elem])

for row in range(len(y)):
    y[row][0] = float(y[row][0])


rmse = []

x_t = np.transpose(x)

for lamda in range(51):
    wrr_p1 = np.linalg.inv(lamda*identity + np.dot(x_t, x))
    wrr = np.dot(np.dot(wrr_p1, x_t), y)
    error = y_test - np.dot(x_test, wrr)
    rmse.append(math.sqrt(np.dot(np.transpose(error), error)))

plt.plot(range(51), rmse)
plt.xlabel('lamda')
plt.ylabel('RMSE')
plt.show()

