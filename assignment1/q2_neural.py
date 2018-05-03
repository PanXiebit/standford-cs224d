import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    data: (N, Dx)
    W1: (Dx, H)
    b1: (1, H)
    W2: (H, Dy)
    b2: (1, Dy)
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:(ofs+ Dx * H)], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:(ofs + H)], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:(ofs + H * Dy)], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:(ofs + Dy)], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h1 = sigmoid(np.dot(data, W1) + b1)  # (N ,H)
    score = np.dot(h1, W2) + b2
    yhat = softmax(score)
    cost = -np.sum(labels * np.log(yhat)) ## label is one-hot vector
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    gradscore = yhat - labels  # (N, Dy)
    gradb2 = np.sum(gradscore, axis=0, keepdims=True)  # (1,Dy)
    gradW2 = h1.T.dot(gradscore)  # (H, Dy)
    gradh1 = gradscore.dot(W2.T) * sigmoid_grad(h1)
    gradb1 = np.sum(gradh1, axis=0, keepdims=True)
    gradW1 = data.T.dot(gradh1)
    ### END YOUR CODE
    assert gradb2.shape == b2.shape
    assert gradW2.shape == W2.shape
    assert gradb1.shape == b1.shape
    assert gradW1.shape == W1.shape
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()