# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:02:30 2022

@author: hosam
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#read data
data = pd.read_csv('C:\\Users\\hosam\\OneDrive\\Desktop\\regression_data.csv', header=None, names=['X', 'Y'])

#show data
data.plot(kind='scatter', x='X', y='Y', figsize=(7,7))


# adding a new column called ones
data.insert(0, 'Ones', 1) #for base
columns = data.shape[1]
X = data.iloc[:,0:columns-1]
y = data.iloc[:,columns-1:columns]





# convert matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0])) #initialize the theta vector with zeros

# cost function
def Cost(X, y, theta):
    return np.sum(np.power(((X * theta.T) - y), 2))/2


# Batch Gredian Decent function
def BatchGD(X, y, theta, alpha, epochs):
    UpdateMatrixTheta = np.matrix(np.zeros(theta.shape))  
    parameters = theta.shape[0]*theta.shape[1]  #number of thetas 
    cost = np.zeros(epochs) #for old cost all of iteration
    theta_temp = np.zeros(epochs)
    theta_temp2 = np.zeros(epochs)#store old thetas
    
    for i in range(epochs):
        for j in range(parameters):
            UpdateMatrixTheta[0,j] = theta[0,j] - ((alpha/len(X))* np.sum(np.multiply((X * theta.T) - y, X[:,j]))) 
        theta_temp[i] =UpdateMatrixTheta[0,0]
        theta_temp2[i] =UpdateMatrixTheta[0,1]
        theta = UpdateMatrixTheta
        cost[i] = Cost(X, y, theta)
   
    return theta, cost , theta_temp , theta_temp2



alpha = 0.01  #learing rate
epochs = 1500 #number of iterations



FinalTheta, cost,thetas1,thetas2 = BatchGD(X, y, theta, alpha, epochs)



print("the cost when theta is zeros =",cost[0]) 
print("the last theta =",FinalTheta)

# get predict line

x = np.linspace(data.X.min(), data.X.max(), 100)
fun = FinalTheta[0, 0] + (FinalTheta[0, 1] * x)




# draw the line

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(x, fun, 'r', label='Prediction')
ax.scatter(data.X, data.Y, label='Data Point')
ax.legend(loc=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')



# draw error graph
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(thetas1,thetas2, cost, 'r')
ax.set_xlabel('first thetas')
ax.set_ylabel('second thetas')
ax.set_zlabel('Error')
ax.set_title('Error vs. thetas')


print("the predicted output =",np.array([1,3.5])*FinalTheta.T)

