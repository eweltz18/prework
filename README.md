# Prework assignment for Applied AI (iX 2017)

In this assignment, we'll look at some questions related to the readings you
did, as well as train our first machine learning model.

## Setup

If you already know what you're doing with Git, clone this repository.
If not, download it as a zip folder using the green "Clone or download" button
near the top of the page.

## Deliverables

1. Copy `README.md` (this file) to a file called `readings.md`. Add answers
(right after the question text) to questions 4, 5, 6, 7, and 9 from Part I.
All answers should be three sentences or less, and none require
equations (maybe some multiplication).
2. Add solutions to `exercises.py` as described in questions 2, 3, and 8 from
Part I. You can run tests for each question by running `python exercises.py N`
from the command line, replacing `N` with the question number.
3. Complete Part II in `logistic_regression.py`, and run the model as
described.
4. Create a zip file called `MYGITHUBUSERNAME_prework.zip` containing
`readings.md`, `exercises.py`, `logistic_regression.py`, `pred.pkl`, and
`pred_tf.pkl`. Replace `MYGITHUBUSERNAME` with your username. Upload this to
Canvas using the instructions provided in the email.

And that's it! Good luck on your first assignment.


## Part I: Reading response questions

#### Python

1. Write PEP 8-compliant code for this assignment.
2. Use a dictionary comprehension to complete the function
`list_to_index_map(...)` in `exercises.py`.

#### NumPy

3. Write a vectorized version of the code snippet below as `vectorized(...)`
in `exercises.py`.

```python
import numpy as np

X = []
for _ in xrange(10):
    row = np.zeros(5)
    if np.random.rand() > 0.5:
        for i in xrange(len(row)):
            row[i] = 1
    X.append(row)
X = np.vstack(X)
```

#### Basics of machine learning

[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf)

4. Gradient descent - more specifically, stochastic gradient descent - is 
the primary algorithm used to train machine learning models.
At the end of Section I.1, the SGD algorithm listing says to loop *m* times.
But how do we choose *m*? One idea is to keep looping until none of the model
parameters change by more than, say, 0.000001. But this isn't always a good
idea. Look up **overfitting** on the world wide web. This is an essential
concept in machine learning. Give one way to prevent overfitting involving the
number of training epochs *m*.

5. In Part II, we discuss classification with logistic regression.
I just had a great idea! Instead of using this weird logistic loss function,
let's just use 0-1 loss (given below) to train our model. Afterall, that's the exact
metric I want my classifier to be optimized for. Spoiler: this isn't a great idea. Why not?

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\ell(\theta)&space;=&space;\left\lbrace\begin{array}{ll}&space;0&space;&&space;\text{sign}(\theta^Tx^{(i)})&space;=&space;y^{(i)}&space;\\&space;1&space;&&space;\text{o.w.}&space;\end{array}&space;\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\ell(\theta)&space;=&space;\left\lbrace\begin{array}{ll}&space;0&space;&&space;\text{sign}(\theta^Tx^{(i)})&space;=&space;y^{(i)}&space;\\&space;1&space;&&space;\text{o.w.}&space;\end{array}&space;\right." title="\ell(\theta) = \left\lbrace\begin{array}{ll} 0 & \text{sign}(\theta^Tx^{(i)}) = y^{(i)} \\ 1 & \text{o.w.} \end{array} \right." /></a>

(Here, we'll assume our labels are -1's and +1's, instead of 0's and 1's).

#### Linear algebra for machine learning

[CS229 linear algebra notes](http://cs229.stanford.edu/section/cs229-linalg.pdf)

Say I have a matrix *A* which is *n* x *p* and a matrix *B* which is *p* x *d*.

6. Suppose I also have a *d*-dimensional vector *y*, and I want to compute the
product *z = ABy*. Under what circumstances should I compute *AB* first?
When should I compute *By* first? Your answer should be in terms of the 
number of multiplications you need to make in the two cases.
Big-*O* notation might be useful.

7. Now suppose I have *k* different *d*-dimensional vectors
*y*<sub>1</sub>,..., *y*<sub>k</sub> and I want to compute all of the products
*z*<sub>i</sub> = *ABy*<sub>i</sub> for *i*=1,...,*k*.
Under what circumstances should I precompute *AB* before computing the
products *z*<sub>i</sub>?

#### TensorFlow

8. The **softmax** function is given in equation (8) of Section III.9.3 in the
[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf).
It's an important function in machine learning, so TensorFlow implements it in
`tf.nn.softmax(...)`. But unfortunately someone pushed a bad change to 
TensorFlow's C implementation of the softmax function, and now it's broken.
Implement the `softmax(...)` function in `exercises.py` using base
TensorFlow functions - like `tf.exp(...)` and `tf.reduce_sum(...)` - so that
the code chunk below works.

```python
import tensorflow as tf
from exercises import softmax

d = 10
n_classes = 5
X = tf.placeholder(tf.float32, (None, d))
theta = tf.get_variable('theta', (d, n_classes), dtype=tf.float32)
h = tf.matmul(X, theta)
predictions = softmax(h)
```

9. TensorFlow uses **automatic differentiation** (autodiff) in order to
train complex machine learning models with gradient descent using an algorithm
called **backpropagation** (which we'll learn more about later).
TensorFlow uses a type of autodiff called **symbolic differentiation** in which
every operator knows its own gradient. There's another type of autodiff called
**numerical differentiation**. Look up symbolic and numerical differentiation
on the world wide web. Describe one advantage and one disadvantage for each.
Why did TensorFlow go with symbolic differentiation?

## Part II: Sentiment analysis warmup

Let's get our hands dirty. We're going to train a model
to predict whether movie reviews are positive or negative. For example,
given the sentence

```
a trashy , exploitative , thoroughly unpleasant experience .
```

we would want to predict that it has a low probability of being positive.

Luckily, we have access to a bunch of movie reviews and someone has taken the
time to mark whether or not they were positive. Check out `data/train.text`.
We're going to train a logistic regression model (which you read about)
over a bag-of-words representation of the sentences.
This is a very simple vector representation of words and sentences. We'll take
a fixed vocabulary of size *d* where every word has an integer index.

* To represent a word, we simply take a *d* dimensional vector of zeros and put
a one at the word's index
* To represent a sentence, we sum the *d* dimensional vectors of all the words
in the sentence

These sentence representations lose all word order information, but we can
generalize the notion of "word" to an n-gram, which is a (short) sequence of
words. So, we would count `not good` as a single entry in the vocabulary,
as well as both `not` and `good`. We'll assemble all of these sentence vectors
into a matrix so that we can train a logistic regression model as normal,
using the provided labels.

We have almost all of the code we need train this model.
The goal is primarily to understand what a basic machine learning application
looks like in code. Review the code described below.

#### Creating word features

The n-gram bag-of-words featurizer is implemented in the functions

* `words_to_ngrams(...)` in `util.py` which converts a list of
words to a list of n-grams that pass a filter
* `sentence_to_ngram_counts(...)` in `util.py` which converts
a raw sentence string to a [Counter](https://docs.python.org/2/library/collections.html#collections.Counter)
of the unique n-grams in it
* `SymbolTable` in `util.py` which converts n-gram counts to a data matrix

#### Logistic regression implemented with NumPy

The `LogisticRegression` class in `logistic_regression.py` implements the loss
function, gradients, weight update step, and prediction function for a 
logistic regression model, all using base NumPy functions.
Here, we can see the "inner-workings" of training a linear model.

* `_loss(...)` computes the log loss for a minibatch of data
* `_grad(...)` computes the gradients of the log loss with
respect to the model parameters. The gradients are covered in the
[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
* `_update_weights(...)` updates the parameters of the model
by calling `_grad(...)`
* `predict(...)` gets probabilistic predictions for test data

#### Level up: machine learning in TensorFlow

NumPy is great, but it isn't a good fit for large-scale models and deep
neural networks. That's where TensorFlow comes in. TensorFlow is overkill for
our simple model and on small dataset, but let's see what a model looks like.
The `TFLogisticRegression` class implements a logistic regression in TensorFlow.

* `_build(...)` builds the computation graph and computes the logistic loss
(equivalent to cross entropy loss here) of a minibatch of data.
We don't have to compute any derivatives
here by hand. TensorFlow does that for us if we just tell it the loss.
* `_update_weights(...)` runs a batch of data through the model
graph and updates the weights with gradient descent
* `predict(...)` gets probabilistic predictions for test data

#### Completing the implementation and running the model

All that we have to do is implement a simple minibatch SGD training loop in
`LogisticRegression.train(...)`. This will let us train both the NumPy and 
TensorFlow models. Implement the pseudo-code algorithm below, which is a
standard SGD algorithm like we saw at the end of Section I.1 in the
[CS229 intro notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf).
The doc string for the function gives some hints for function calls and 
a rough estimate of the number of lines needed.

```
Input: X ([m x d] data matrix), y (label array of length m), T (number of epochs), b (batch size), α (learning rate)

Set the random seed
Initialize model weights
for t = 1,2,...,T:
    Create a permutation P of 1,2,...,m
    Permute y and the rows of X according to P
    for each batch of b rows of X and corresponding b entries of y:
        Update the model weights using the batch and the learning rate α
        Record the batch loss
    Print a status message using the average of the batch losses from the epoch
```


You're ready to train your model and get some results! Run

```bash
python train.py
```

to train your model. It will print the loss at each training epoch, which
will decrease if the model is implemented correctly. You should get a
test accuracy of at least 71%. Feel free to change the hyperparameters
(such as the learning rate or number of epochs). Can you get the model
to overfit (very high training accuracy, low test accuracy)?

To train the TensorFlow implementation, run

```bash
python train.py tf
```

You should get similar results. Predictions will be saved to `pred.pkl` and 
`pred_tf.pkl`.
