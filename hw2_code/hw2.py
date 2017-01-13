from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import pdb

NUM_CLASSES = 10
NUM_FEATURES = 5000
memory = {}

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    d = X_train.shape[1]
    A = np.dot(X_train.T, X_train) + reg * np.identity(d)
    L = np.dot(X_train.T, y_train)
    W = np.dot(np.linalg.inv(A), L)
    return W

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    return np.zeros((X_train.shape[1], NUM_CLASSES))

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    return np.zeros((X_train.shape[1], NUM_CLASSES))

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    y = np.zeros((X_train.shape[0], NUM_CLASSES))
    for i, x in enumerate(labels_train):
        y[i][x] = 1
    return y

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    pred = np.dot(X, model)
    y = np.empty(pred.shape[0])
    for i in range(pred.shape[0]):
        y[i] = pred[i].argmax()
    return y

def phi(X):
    ''' Featurize the inputs using random Fourier features '''
    d = NUM_FEATURES
    var = 4

    if 'G' not in memory:
        mean = np.zeros(X.shape[1])
        cov = var * np.identity(X.shape[1])
        G = np.random.multivariate_normal(mean, cov, d)
        memory['G'] = G
    else:
        G = memory['G']
    if 'b' not in memory:
        b = np.random.uniform(0, 2*np.pi, (d,1))
        memory['b'] = b
    else:
        b = memory['b']

    T = np.dot(G, X.T) + b
    P = np.cos(T)
    return P


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    print("X_train's shape is: " + str(X_train.shape))
    print("X_test's shape is: " + str(X_test.shape))
    y_train = one_hot(labels_train)
    print("y_train's shape is: " + str(y_train.shape))
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train), phi(X_test)
    print("X_train's shape after phi is: " + str(X_train.shape))
    print("X_test's shape after phi is: " + str(X_test.shape))

    d = X_train.shape[1]
    A = np.dot(X_train.T, X_train) + reg * np.identity(d)
    L = np.dot(X_train.T, y_train)
    W = np.dot(np.linalg.inv(A), L)

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
