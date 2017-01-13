from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy

NUM_CLASSES = 10

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

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]


def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(X.dot(model), axis=1)

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    model = train(X_train, y_train, reg=0.1)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)

    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
