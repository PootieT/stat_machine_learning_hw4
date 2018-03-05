import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  C = float(C)
  J = 1/(2*m)*np.sum(theta**2) + C/m*np.sum(np.maximum(np.zeros([m]),1 - np.multiply(y, np.dot(X, theta.T))))
  grad = 1/m * theta
  index = np.multiply(y, np.dot(X, theta.T)) < 1.0
  grad += C/m * (-np.dot(y[index],X[index]))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
	that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples

  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
  temps = np.zeros((m,K,K))
  temp = np.matmul(X,theta)
  for i in range(m):
    for j in range(K):
      for k in range(K):
        temps[i,j,k] = temp[i,j] - temp[i,k] + delta

  for i in range(m):
    for j in range(K):
      if j != y[i] and temps[i,j,y[i]] > 0.0:
        J += 1.0 / float(m) * temps[i,j,y[i]]
        dtheta[:,j] += 1.0 / float(m) * X[i,:]
        dtheta[:,y[i]] -= 1.0 / float(m) * X[i,:]

  for i in range(K):
    for j in range(theta.shape[0]):
      J += reg / (2.0) * (theta[j,i] ** 2)
  dtheta += reg * theta



  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  K = theta.shape[1] # number of classes
  m = X.shape[0]     # number of examples
  d = X.shape[1]
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  # 8-10 lines of code                                                        #
  #############################################################################

  temp = np.matmul(X, theta).T
  temp_three = (temp - temp[y, range(m)]).T + delta
  
  temp_four = np.maximum(np.zeros(temp_three.shape), temp_three)
  temp_four[range(m), y] = 0
  
  temp_five = np.sum(temp_four, axis = 1)
  
  J += 1.0 / float(m) * np.sum(temp_five)
  J += reg / (2.0) * np.sum(np.square(theta))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  temp_six = np.sign(temp_four)
  temp_seven = -1.0 * np.sum(temp_six,axis=1).T
  temp_six[range(m),y] = temp_seven
  dtheta += np.matmul(X.T, temp_six)
  dtheta *= 1.0 / float(m)
  
  dtheta += reg * theta
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
