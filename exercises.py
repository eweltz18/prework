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
    print "TEST PASSED!"


def vectorized():
    """Implements the code snippet below in a vectorized fashion

    ```
    X = []
    for _ in xrange(10):
        row = np.zeros(5)
        if np.random.rand() > 0.5:
            for i in xrange(len(row)):
                row[i] = 1
        X.append(row)
    X = np.vstack(X)
    ```

    Args: None

    Lines: 3-4

    Returns: a 2D NumPy array (X in the snippet)
    """
    raise NotImplementedError()


def vectorized_test():
    def slow():
        X = []
        for _ in xrange(10):
            row = np.zeros(5)
            if np.random.rand() > 0.5:
                for i in xrange(len(row)):
                    row[i] = 1
            X.append(row)
        X = np.vstack(X)    
        return X
    np.random.seed(1701)
    X_slow = slow()
    np.random.seed(1701)
    X_fast = vectorized()
    if not np.array_equal(X_slow, X_fast):
        raise Exception("TEST FAILED!\n\n{0}\n\n{1}".format(X_slow, X_fast))
    print "TEST PASSED!"


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
    print "TEST PASSED!"


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("No problem number provided")
    N = sys.argv[1]
    if N == '2':
        list_to_index_map_test()
    elif N == '3':
        vectorized_test()
    elif N == '8':
        softmax_test()
    else:
        raise Exception("No problem <{0}>. Options: [2, 3, 8]".format(N))
