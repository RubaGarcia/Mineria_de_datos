import numpy as np
import math


# sigmoid(z) computes the sigmoid of z.
def sigmoid(z):
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar). You may find useful numpy.exp and numpy.power.
    g = 1/(1 + np.exp(-z))
# =============================================================
    assert (z.shape == g.shape)
    return g


#   computeCost(X, y, theta) computes the cost of using theta as the
#   parameter for logistic regression 
def computeCost(X, y, theta):
    # Initialize some useful values
    m,p = X.shape
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. You may find useful numpy.log
#               and the sigmoid function you wrote.
#
#
    A = X @ theta
    J1 = np.log(sigmoid(A))
    J1 = y.T @ J1
    J2 = (np.ones(y.shape) - y).T @  np.log(np.ones(A.shape) - sigmoid(A))
    J = float((J1 + J2)/-m)
 # =============================================================
    assert (isinstance(J,float))
    return J

#   gradientDescent(X, y, theta, alpha, iterations) updates theta by
#   taking "iterations" gradient steps with learning rate alpha

def gradientDescent(X, y, theta, alpha, iterations):
    # Initialize some useful values
    m,p = X.shape
    # ====================== YOUR CODE HERE ======================
    for i in range(iterations):
        theta = theta - alpha * ( np.transpose(X) @ (sigmoid(X @ theta) - y)/m)                               
    # ============================================================
    assert (theta.shape==(p,1))
    return theta

#   predict(theta, X) computes the predictions for X using a threshold at 0.5
#   (i.e., if sigmoid(theta'*x) >= 0.5, predict 1, otherwise predict 0)
def predict(X,theta):
    # Initialize some useful values
    m,p = X.shape 
# You need to return the following variable correctly

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#     s          your learned logistic regression parameters. 
#               You should set p to a vector of 0's and 1's
    pred=sigmoid(X @ theta)>=0.5
# =========================================================================
    assert (pred.shape==(m,1))
    return pred




