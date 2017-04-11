"question e"
x_train1=np.asmatrix(xtrain1)
y_train1=np.asmatrix(ytrain1)

w1=np.asmatrix(np.zeros((58,1)))
L1=[]

b=0

for t in range(1,101):

   for i in range(0,len(xtrain1)):
        
        b=b+lnsigmoid(np.dot(xtrain1[i],w1)*ytrain1[i])
   delta1=np.dot(np.multiply((1-sigmoid(np.multiply(np.dot(xtrain1,w1),ytrain1))),ytrain1).T,xtrain1)
   delta2=-np.dot(np.transpose(xtrain1),np.multiply(np.multiply(sigmoid(np.dot(xtrain1,w1)),(1-sigmoid(np.dot(xtrain1,w1)))),xtrain1))
   w1=w1-(1./(np.sqrt(t+1)))*(np.linalg.inv(delta2))*delta1.T
   L1=np.append(L1,b)
   b=0
 
"L graph" 

plt.plot(range(1,101),L1)   
plt.xlabel("iteration")
plt.ylabel("L")
plt.show()

"prediction"

prediction = sigma(w1, xtest1)
prediction = np.where(prediction >= 0.5, 1, -1)
error = 0
for i in range(93):
    if prediction[i][0] != ytest[i][0]:
        error += 1
print 1 - error/93.0
