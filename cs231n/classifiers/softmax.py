import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    # exponentiate the values
    denominator = 0
    for j in range(num_classes):
      denominator += np.exp(scores[j])
      dW[:, j] += (np.exp(scores[j]) * X[i]) / np.sum(np.exp(scores))

    dW[:, y[i]] -= X[i]

    loss += -np.log(np.exp(correct_class_score) / denominator)
  
  loss = loss / num_train
  dW = dW / num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = np.zeros_like(W)

  scores = np.dot(X, W)
  # solves the numeric stability problem
  scores = scores - scores.max(axis=1)[:, np.newaxis]
  correct_class_scores = np.choose(y, scores.T)[:, np.newaxis]

  correct_class_scores_exp = np.exp(correct_class_scores)
  scores_exp_sum = np.exp(scores).sum(axis=1)[:, np.newaxis]
  # scores_exp_minus_correct_class_scores = scores_exp_sum
  # now remove the correct_class_scores
  loss_step = -np.log(correct_class_scores_exp / scores_exp_sum)
  loss += np.sum(loss_step)

  # Compute the gradient
  derivate = np.exp(scores) / scores_exp_sum
  derivate[np.arange(num_train), y] -= 1
  dW = X.T.dot(derivate)

  # average
  loss = loss / num_train
  dW = dW / num_train

  # regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

