import numpy as np
import tensorflow as tf

from util import sigmoid


class LogisticRegressionBase(object):
    def __init__(self, dim):
        """Abstract class for a logistic regression model

        Args:
            dim: dimension of model weights
        """
        self.d = dim

    def _update_weights(self, X, y, alpha):
        """Applies an SGD step by updating self.w and self.b

        Args:
            X: training batch data matrix (2D NumPy array)
            y: training batch labels (NumPy vector)
            alpha: learning rate

        Returns: batch loss after update
        """
        raise NotImplementedError()

    def _initialize(self):
        """Initialize model variables"""
        pass        

    def _message(self, t, avg_loss):
        return "Epoch {0:<2}\tAverage loss = {1:.6f}".format(t, avg_loss)

    def train(self, X, y, lr=0.5, n_epochs=10, batch_size=10, seed=1701):
        """Train a logistic regression model

        ** IMPLEMENT ME! **
        NOTE: THIS IS THE ONLY FUNCTION YOU HAVE TO IMPLEMENT

        Args:
            X: training data matrix (2D NumPy array)
            y: training data labels (NumPy vector)
            lr: learning rate
            n_epochs: number of epochs
            batch_size: size of each SGD batch
            seed: random seed; if None, no random seeding

        Calls:
            np.random.seed(...)
            _initialize(...)
            np.random.permutation(...)
            _update_weights(...)
            _message(...)

        Lines: 12-20

        Returns: None
        """
        raise NotImplementedError()

    def predict(self, X):
        """Get positive class probabilities for test data

        Args:
            X: test data matrix (2D NumPy array)

        Returns: NumPy vector of positive class probabilities for test data
        """
        raise NotImplementedError()

    def accuracy(self, X, y, b=0.5):
        """Get model accuracy on test data

        Args:
            X: test data matrix (2D NumPy array)
            y: test data labels (NumPy vector)

        Returns: fraction of correct predictions
        """
        y_hat = self.predict(X)
        return np.mean((y_hat > b) == (y > b))


class LogisticRegression(LogisticRegressionBase):

    def __init__(self, dim):
        super(LogisticRegression, self).__init__(dim)
        self.w = None
        self.b = None


    def _initialize(self):
        self.w = np.random.normal(scale=0.1, size=self.d)
        self.b = 0.
    
    def _score(self, X):
        """Get positive class score for data

        Args:
            X: data matrix (2D NumPy array)

        Returns: NumPy vector of positive class scores for data
        """
        return np.ravel(X.dot(self.w) + self.b)

    def _loss(self, X, y):
        """Get logistic loss for data with respect to labels

        Args:
            X: data matrix (2D NumPy array)
            y: data labels (NumPy vector)

        Returns: average logistic loss
        """
        y = (2*y - 1).copy()
        z = y * self._score(X)
        pos = z > 0
        q = np.empty(z.size, dtype=np.float)
        q[pos] = np.log1p(np.exp(-z[pos]))
        q[~pos] = (-z[~pos] + np.log1p(np.exp(z[~pos])))
        return np.mean(q)

    def _grad(self, X, y):
        """Compute gradient of logistic loss for batch

        Args:
            X: training batch data matrix (2D NumPy array)
            y: training batch labels (NumPy vector)

        Returns: tuple of a scalar and a self.d dimensional vector which are
            - gradient of log loss with respect to biases (self.b)
            - gradient of log loss with respect to weights (self.w)
        """
        h = sigmoid(self._score(X))
        return np.mean(h - y), np.ravel(X.T.dot(h - y)) / len(y)

    def _update_weights(self, X, y, alpha):
        db, dw = self._grad(X, y)
        self.w -= alpha * dw
        self.b -= alpha * db
        return self._loss(X, y)

    def predict(self, X):
        return sigmoid(self._score(X))


class TFLogisticRegression(LogisticRegressionBase):
    def __init__(self, dim):
        super(TFLogisticRegression, self).__init__(dim)
        self.session = tf.Session()
        self._build()

    def _initialize(self):
        self.session.run(tf.global_variables_initializer())

    def _build(self):
        """Build TensorFlow model graph

        Populates input placeholders (self.X, self.y, self.lr),
        model variables (self.w, self.b),
        and core ops (self.predict_op, self.loss_op, self.train_op)
        """
        # Data placeholders
        self.X = tf.placeholder(tf.float32, (None, self.d))
        self.y = tf.placeholder(tf.float32, (None,))
        self.lr = tf.placeholder(tf.float32)
        # Compute linear "layer"
        with tf.variable_scope('linear'):
            self.w = tf.get_variable('w', (self.d, 1), dtype=tf.float32)
            self.b = tf.get_variable('b', (1,), dtype=tf.float32)
        h = tf.squeeze(tf.nn.bias_add(tf.matmul(self.X, self.w), self.b))
        # Prediction op
        self.predict_op = tf.sigmoid(h)
        # Train op
        self.loss_op = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=h)
        )
        trainer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = trainer.minimize(self.loss_op)

    def _update_weights(self, X, y, lr):
        feed = {self.X: X, self.y: y, self.lr: lr}
        return self.session.run([self.loss_op, self.train_op], feed)[0]

    def predict(self, X):
        return self.session.run(self.predict_op, {self.X: X})
