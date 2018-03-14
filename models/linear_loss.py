import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    y_predict = [sum([k*p for k,p in zip(x,W)]) for x in X]
    loss = sum([(_y-y)**2 for _y,y in zip(y_predict,y)]) / (2*len(y))

    # N,D = X.shape

    # for i in range(D):
    #     for j in range(N):
    #         dW[i] += X[j][i] * (y_predict[j] - y[j]) / (len(y))
    
    dW = sum([(_y-y_true)*x  for x,y_true,_y in zip(X,y,y_predict)]) / len(y) 
    

    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    y_predict = X.dot(W)
    loss = np.mean((y_predict-y)**2) / 2 
    dW = X.transpose().dot(y_predict-y) / len(y)
    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW