import math
import numpy as np
from scipy import linalg

#   normalEqn(X,y) computes the closed-form solution to linear 
#   regression using the normal equations.
def normalEqn(X, y):
    # Initialize some useful values
    m,p=X.shape
   
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#
    # Haciendo uso del método analítico: 
    # Traspuesta de X 
    xt=np.transpose(X)
    # Multiplicación de Transpuestade X por X
    x2 = xt @ X
    # Multiplicaion de transpuesta de X por y
    x3 = xt @ y
    # Inversa  de x2
    xi = np.linalg.inv(x2)
    # Calculo de la matriz Theta
    theta = xi @ x3
# ============================================================
    assert (theta.shape==(p,1))
    return theta

#   featureNormalize(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.
#   Set mu and sigma to a column vector containing the mean and
#   standard variation of each feature in X, respectively
#   (you may want to check the documentation of numpy.std and numpy.mean)

def featureNormalize(X):
    # Initialize some useful values
    m,p=X.shape
    # You need to set these values correctly
# ====================== YOUR CODE HERE ======================
    mu = np.mean(X,axis = 0) #promedio
    sigma = np.std(X,axis = 0) #desviacion estandar
    X_norm = (X - mu)/sigma #x estandarizada
# ============================================================
    assert (X_norm.shape==X.shape)
    assert (mu.shape==(p,))
    assert (sigma.shape==(p,))
    return X_norm, mu, sigma


#   computeCost(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
def computeCost(X,y,theta):
    # Initialize some useful values
    m,p=X.shape
    # You need to return the following variable correctly

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    J1=(X @ theta)-y
    J2 = np.transpose(J1) @ J1 #ya queda elevado al cuadrado y es el sumatorio
    J = float(J2)/float(2*m)
# =========================================================================    
    assert (isinstance(J,float))
    return J


#   gradientDescent(x, y, theta, alpha, num_iters) updates theta by
#   taking iterations gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, iterations):  
    # Initialize some useful values
    m,p=X.shape
   
    # ====================== YOUR CODE HERE ======================
    for i in range(iterations):
        theta = theta - alpha * ( np.transpose(X) @ ((X @ theta) - y)/m)
    # ============================================================  
    assert (theta.shape==(p,1))
    return theta

#   predict(newdata, mu, sigma) predicts the value of the variable y for
#   previously unseen data (in general, an m by p numpy.array)
def predict(newdata,mu,sigma,theta):
    # Initialize some useful values
    m,p=newdata.shape
    value=np.zeros((m,1))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Don't forget to add the column of ones to the
    # normlized version of newdata
    value = ((newdata - mu)/sigma) 
    value = theta @ np.c_[np.ones((m,1)), value] 
    # ============================================================  
    assert (value.shape==(m,1))
    return value
