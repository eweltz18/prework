import numpy as np
import sys
import tensorflow as tf


def list_to_index_map(x):
    """Convert a list to a dictionary mapping each list element to its index

    Args:
        x: a list with all unique elements

    Lines: 1

    Returns: a dictionary mapping each list element to its index
    """
    raise NotImplementedError()


def list_to_index_map_test():
    x = list('tensor')
    d = list_to_index_map(x)
    for j in x:
        if x.index(j) != d.get(j, None):
            raise Exception("TEST FAILED! d[{0}] = {1}\tx[{2}] = {0}".format(
                j, d[j], x.index(j)
            ))
    print("TEST PASSED!")


def softmax(h):
    """Computes the softmax function

    Args:
        X: a batch_size x n_classes Tensor

    Calls:
        tf.exp(...)
        tf.reduce_sum(...)

    Lines: 1-3

    Returns: a batch_size x n_classes Tensor with softmax values
    """
    raise NotImplementedError()


def softmax_test():
    X = tf.random_uniform((10, 5))
    X_tf, X_mn = tf.Session().run([tf.nn.softmax(X), softmax(X)])
    if not np.allclose(X_tf, X_mn):
        raise Exception("TEST FAILED!\n\n{0}\n\n{1}".format(X_tf, X_mn))
    print("TEST PASSED!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("No problem number provided")
    N = sys.argv[1]
    if N == '3':
        list_to_index_map_test()
    elif N == '6':
        softmax_test()
    else:
        raise Exception("No problem <{0}>. Options: [3, 6]".format(N))
