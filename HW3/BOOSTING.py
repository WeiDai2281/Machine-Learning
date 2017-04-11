import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
xtrain = np.loadtxt('X_train.csv',delimiter = ',', dtype = 'float')
xtrain = np.column_stack((xtrain, np.ones(shape = (xtrain.shape[0], 1))))
xtest = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'float')
xtest = np.column_stack((xtest, np.ones(shape = (xtest.shape[0], 1))))
ytrain = np.loadtxt('y_train.csv', delimiter = ',', dtype = 'float')
ytest = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'float')

def predict(f, x):
    prediction = np.sign(np.dot(x, f))
    return prediction

def Boostrap(w, x, y):
    index = np.array(range(x.shape[0]))
    index = np.random.choice(index, x.shape[0], p = w)
    return x[index], y[index], index

def train_regressor(x, y):
    temp = np.linalg.inv(np.dot(np.transpose(x), x))
    result = np.dot(np.dot(temp, np.transpose(x)), y)
    return result
    

def update_epsilon(y, w, prediction):
    flag = 1
    #error = (y != prediction)
    error = np.where(y + prediction == 0, 1, 0)
    result = np.dot(error, w)
    if result > 0.5:
        flag = -1
    return result, flag

def update_alpha(epsilon):
    #print epsilon
    result = 0.5 * math.log(1.0 * (1 - epsilon)/epsilon)
    return result

def update_w(w, alpha, y, prediction):
    temp = -1 * alpha * y * prediction
    #print temp
    result = w * np.exp(temp)
    result = result / np.sum(result)
    return result

# The main function
def main(x, y, step):
    count = 0
    xt = []
    n = x.shape[0]
    f = []                                     #storing the parameters vector of every regressor
    epsilon = []
    alpha = []
    error = []
    w = np.array([1.0 / n for _ in range(n)]) # The probability vector of every point, initialized by all equal to 1/n
    for t in range(step):

        x_t, y_t, index_t = Boostrap(w, x, y)                      #Boostrap the sample
        index.extend(index_t)
        f.append(train_regressor(x_t, y_t))
        prediction = predict(f[t], x)
        epsilon_t, flag = update_epsilon(y, w, prediction)
        while flag == -1:
            f[t] = f[t] * -1
            prediction = prediction * -1
            epsilon_t, flag = update_epsilon(y, w, prediction)
        epsilon.append(epsilon_t)
        alpha.append(update_alpha(epsilon_t))
        w = update_w(w, alpha[t], y, prediction)
    return np.transpose(np.array(f)), np.array(alpha).reshape(step, 1), np.array(epsilon)

index = []
f_boost, alpha, epsilon = main(xtrain, ytrain, 1500)


    
    

def error_calculate(x, alpha, y, step):
    n = x.shape[0]
    error = []
    prediction_matrix = np.dot(x, f_boost)
    for t in range(step):
        prediction = np.sign(prediction_matrix[:,:t+1])
        prediction = np.sign(np.dot(prediction, alpha[:t + 1,:]))
        #temp = (prediction != y.reshape(n, 1))
        temp = np.where(prediction + y.reshape(n, 1) == 0, 1, 0)
        result = (1.0 / n) * np.sum(temp)
        error.append(result)
    return error

train_error = error_calculate(xtrain, alpha, ytrain, 1500)
test_error = error_calculate(xtest, alpha, ytest, 1500)

plt.plot(range(1, 1501), train_error, 'k--', range(1, 1501), test_error, 'r-')
plt.show()



#question b. calculate upper bound
result = []
for t in range(1500):
    temp = (0.5 - epsilon[:t + 1]) ** 2
    result.append(math.exp(-2 * np.sum(temp)))

                  
plt.plot(range(1, 1501), result, 'k--')
plt.show()

#question c
plt.hist(index)
plt.show()



#question d
plt.figure(4)
print alpha.reshape(1500,)
plt.plot(range(1, 1501), alpha.reshape(1500,), 'k')
plt.show()
    

plt.figure(5)
plt.plot(range(1, 1501), epsilon, 'r--')
plt.show()

