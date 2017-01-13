from mnist import MNIST
import random
import sklearn.metrics as metrics
import numpy as np
import scipy
import pdb

NUM_CLASSES = 10
dim = 784
d = 4000
w = np.random.normal(0, 0.1, (dim, d))
b = np.random.uniform(0, 2*np.pi, d)

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    xtx = X_train.T.dot(X_train)
    return scipy.linalg.solve(xtx + reg*np.eye(xtx.shape[0]), X_train.T.dot(y_train), sym_pos=True)

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    xtx = X_train.T.dot(X_train)
    xty = X_train.T.dot(y_train)
    xtx /= float(X_train.shape[0])
    xty /= float(X_train.shape[0])
    W = np.random.normal(0, 0.3, (X_train.shape[1], NUM_CLASSES))
    for i in range(num_iter):
        gradient = (xtx + reg*np.eye(xtx.shape[0])).dot(W) - xty
        W -= alpha * gradient
    return W

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    W = np.random.normal(0, 0.3, (X_train.shape[1], NUM_CLASSES))
    for i in range(num_iter):
        sample_index = random.randint(0, X_train.shape[0] - 1)
        x_i, y_i = X_train[sample_index], y_train[sample_index]
        xtx = np.outer(x_i, x_i.T)
        xty = np.outer(x_i, y_i.T)
        gradient = (xtx + reg*np.eye(xtx.shape[0])).dot(W) - xty
        W -= alpha * gradient
    return W

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(X.dot(model), axis=1)

def phi(X):
    ''' Multiply the 784-dimensional MNIST vectors by unit normal '''
    return np.cos(X.dot(w) + b)


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    pdb.set_trace()
    X_train, X_test = phi(X_train), phi(X_test)

    model = train(X_train, y_train, reg=0.1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=20000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    model = train_sgd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=100000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Stochastic gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
