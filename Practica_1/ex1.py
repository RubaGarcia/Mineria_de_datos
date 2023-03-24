# Exercise 1: Multivariate Linear Regression
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on this
#  exercise. You will need to complete the following functions in practica1.py:
#
#     normalEqn
#     featureNormalize
#     computeCost
#     gradientDescent
#     predict
#
#  For this exercise, you will not need to change any code in this file,
#  but feel free to do so.
#
# The dataset contains information about prices of houses 
# x1 refers to the size in square meters
# x2 refers to the number of rooms


# y refers to the price in dollars


from practica1 import normalEqn,computeCost,gradientDescent,featureNormalize,predict
import numpy as np

def loadData(filename):
    print('Loading data ...')
    data=np.loadtxt(filename,delimiter=",")
    m,p=data.shape
    X=np.c_[np.ones((m,1)),data[:,:-1]]
    y=data[:,-1:]
    return m,p,X,y



# ====================== Linear regression with multiple variables ======================
print('Linear regression with multiple variables')
m,p,X,y = loadData('ex1data.txt')
input("\nProgram paused. Press ENTER to continue")

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)



# Display normal equation's result
print('Solving with normal equations...')
print('Theta computed from the normal equations: (%.5f, %.5f, %.5f)'%tuple(theta[:,0]))
print('Theta should be:                          (89597.90954, 139.21067, -8738.01911)')
input("\nProgram paused. Press ENTER to continue")


# Estimate the price of a 1650 sq-ft, 3 br house

house=np.array([[1650,3]],dtype=float)
price = np.dot(np.c_[np.array([[1]]),house],theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): %.6f $'%price[00])
print('Correct answer is:                                                    293081.464335 $\n')

# Scale features and set them to zero mean
print('Normalizing Features ...')

X[:,1:],mu,sigma = featureNormalize(X[:,1:])

print('If done properly, the mean should be 0, and the standard deviation 1')
print('The mean values are: (%.1f,%.1f)'%tuple(np.mean(X[:,1:],axis=0)))
print('The standard deviation values are: (%.1f,%.1f)'%tuple(np.std(X[:,1:],axis=0)))

input("\nProgram paused. Press ENTER to continue")

print('Running Gradient Descent ...')

# Some gradient descent settings
iterations = 500
alpha = 0.01

# Initialize Theta and Run Gradient Descent 
theta = np.zeros((p,1))

# Compute and display initial cost

print('Initial cost is:        %.4f'%computeCost(X, y, theta))
print('Initial cost should be: 65591548106.4574')

input("\nProgram paused. Press ENTER to continue")

theta = gradientDescent(X, y, theta, alpha, iterations)

# Display gradient descent's result

print('Theta found by gradient descent: (%.3f, %.3f, %.3f)'%tuple(theta[:,0]))
print('Theta should be:                 (338175.984, 103032.124, -202.325)')

input("\nProgram paused. Press ENTER to continue")

price=predict(house,mu,sigma,theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): %.7f $' % price[0])
print('Correct answer is:                                                    292264.8818868 $')
