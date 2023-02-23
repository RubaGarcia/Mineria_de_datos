import numpy as np
import copy


def sigmoid(z):
    return 1/(1+np.exp(-z))

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1):
    # nnCostFunction implements the neural network cost function for a two layer
    # neural network which performs classification.
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    #
    # Note: The vector y passed into the function is a vector of labels containing
    # values from 0 to num_labels-1. You need to map this vector into a 
    # binary vector of 1's and 0's.
    m=y.shape[0]
    yh = np.zeros([m,len(np.unique(y))])
    yh[np.arange(m),y]=1 
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network

    nn=hidden_layer_size*(input_layer_size+1)
    Theta1 = nn_params[:nn].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[nn:].reshape(num_labels, (hidden_layer_size + 1))

    # You need to return the following variable correctly 
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Feedforward the neural network and return the cost in the variable J. 
   
    #prop hacia delante
    unos = np.ones((X.rows(),1))
    X = np.concatenate(unos,X,axis=1)
    a1 = X
    z2 = a1 @ Theta1.T 
    gz2= sigmoid(z2)
    unos = np.ones((gz2.rows(),1))
    a2 = np.concatenate(unos,gz2,axis=1)
    z3 = a2 @ Theta2.T
    gz3 = sigmoid(z3)
    unos = np.ones((gz3.rows(),1))
    a3 = np.concatenate(unos,gz3,axis=1)
    h0x = a3

    #coste J
    J = np.sum((np.log(h0x)*yh) + (1 - yh) * np.log(1-h0x))
    J = - J/m 
    J += lambda1/(2*m)*(np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))

    # =========================================================================
    return J



def computeGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1):    
    # computeGradient implements the backpropagation algorithm to compute the gradients
    # Theta1_grad and Theta2_grad. You should return the partial derivatives of
    # the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    # Theta2_grad, respectively. 
    #
    # Note: The vector y passed into the function is a vector of labels containing
    # values from 0 to num_labels-1. You need to map this vector into a 
    # binary vector of 1's and 0's.
    m=y.shape[0]
    yh = np.zeros([m,len(np.unique(y))])
    yh[np.arange(m),y]=1    
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network
    nn=hidden_layer_size*(input_layer_size+1)
    Theta1 = nn_params[:nn].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[nn:].reshape(num_labels, (hidden_layer_size + 1))

    # You need to compute the following variables correctly
    Theta1_grad = np.zeros((Theta1.shape))
    Theta2_grad = np.zeros((Theta2.shape))
    
    # ====================== YOUR CODE HERE ======================
    # Implement backpropagation and return the partial derivatives of
    # the cost function with respect to Theta1 and Theta2 in Theta1_grad
    # and Theta2_grad, respectively
    # Setup some useful variables

    
    # =========================================================================
    # Unroll gradients
    grad = np.concatenate((Theta1_grad.ravel(),Theta2_grad.ravel()),axis=None)
    return grad


def computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1, epsilon):
    # computeNumericalGradient computes the numerical gradient of the
    # cost function around theta, providing a numerical estimate of the gradient.

    # You need to compute the following variables correctly
    numgrad=np.zeros((nn_params.shape))
    # ====================== YOUR CODE HERE ======================
    # You should set numgrad(i) to (a numerical approximation of) the partial
    # derivative of the cost function with respect to the 
    # i-th input argument, evaluated at theta.
    # Hint: use a shallow copy of nn_params


    # =========================================================================

    return numgrad

def randInitializeWeights(l_in,l_out,epsilon):
    # randInitializeWeights randomly initializes the weights of a layer with l_in
    # incoming connections and l_out outgoing connections
    # such that each value is in the (-epsilon,+epsilon) interval
    # Note that w should be set to a numpy array of size(l_out, 1 + l_in) as
    # the first row of w handles the "bias" terms

    # You need to return the following variable correctly 
    w = np.zeros((l_out,1+l_in))

    # ====================== YOUR CODE HERE ======================
    # Initialize w randomly so that we break the symmetry while
    # training the neural network.
    # Hint: you may find numpy.random.rand() useful
    

    # =========================================================================
    return w

