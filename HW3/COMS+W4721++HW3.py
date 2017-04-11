import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
xtrain = np.loadtxt('X_train.csv',delimiter = ',', dtype = 'float')
xtest = np.loadtxt('X_test.csv', delimiter = ',', dtype = 'float')
ytrain = np.loadtxt('y_train.csv', delimiter = ',', dtype = 'float')
ytest = np.loadtxt('y_test.csv', delimiter = ',', dtype = 'float')

n = xtrain.shape[0]
def Kernelmatrix(x, b):
    kernelmatrix = []
    for i in range(n):
        temp = []
        for j in range(n):
            temp.append(math.exp(-(1.0/b) * np.dot(x[i] - x[j], x[i] - x[j])))
        kernelmatrix.append(temp)
    return np.array(kernelmatrix)

def Kernelentity(x, d, b):
    k_xd = []
    for i in range(n):
        k_xd.append(math.exp(-(1.0/b) * np.dot(x - d[i], x - d[i])))
    return np.array(k_xd)

#calculate
n1 = xtest.shape[0]   
def predict(sigma, kernelmatrix, b):
    prediction = [0 for i in range(n1)]
    for i in range(n1):
        temp = np.linalg.inv(sigma * np.identity(n) + kernelmatrix)
        prediction[i] = np.dot(Kernelentity(xtest[i], xtrain, b), np.dot(temp, ytrain))
    return np.array(prediction)

#rmse table
b = [5,7,9,11,13,15]
sigma2 = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
rmse = 0
result = []
for i in range(len(b)):
    tempresult = []
    for j in range(len(sigma2)):
        kernelmatrix = Kernelmatrix(xtrain, b[i])
        #print predict(sigma2[j], kernelmatrix, b[i])
        error = predict(sigma2[j], kernelmatrix, b[i]) - ytest
        #print error
        rmse = math.sqrt(np.dot(error, error)/(1.0 * n1))
        tempresult.append(rmse)
    result.append(tempresult)


# b). Below is the table of RMSE in different parameters


table = {'5' : pd.Series(result[0], index=sigma2),'7' : pd.Series(result[1], index=sigma2),'9' : pd.Series(result[2], index=sigma2),'11': pd.Series(result[3], index=sigma2),'13': pd.Series(result[4], index=sigma2),'15': pd.Series(result[5], index=sigma2)}
show = pd.DataFrame(table, columns=['5', '7', '9', '11', '13', '15'])
show


# c). The best value is when $b = 11$ and $\sigma^2 = 0.1$. The $RMSE = 1.890507$, so the result is better than the result in HW1 (when implement ridge regression with $\lambda = 23$ and $p = 2$, we get $RMSE = 2.19$).  
# However, the GP have some drawbacks. Normally, the GP method takes more time. Especially, when the dimension of variable is high, the complexity of the calculation on kernel matrix will be high too. When we try to find the best hyperparameters, we have to implement the GP method several times, the whole process will be much more time consuming.

# d). The following is the visualization on GP. 

# In[6]:

# question d
xd = np.transpose(xtrain)[3].reshape(350, 1)
kernelmatrix_d = Kernelmatrix(xd, 5)
def predict_d(sigma, kernelmatrix, b):
    prediction = [0 for i in range(n)]
    for i in range(n):
        temp = np.linalg.inv(sigma * np.identity(n) + kernelmatrix)
        prediction[i] = np.dot(Kernelentity(xd[i], xd, b), np.dot(temp, ytrain))
    return prediction


# In[7]:

prediction_d = predict_d(2, kernelmatrix_d, 5)


# In[8]:

#question d
xy_plot = np.column_stack((xd, prediction_d))
xy_plot = xy_plot[xy_plot[:,0].argsort()]


# In[9]:

fig, ax = plt.subplots()
ax.plot(xd, ytrain, 'ro')
ax.plot(xy_plot[:,0], xy_plot[:,1])
#ax.plot(x, y2, 'ro')
plt.show()

