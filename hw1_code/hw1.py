from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import pdb

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
    y = one_hot(y_train)
    d = X_train.shape[1]
    A = np.dot(X_train.transpose(), X_train) + reg * np.identity(d)
    L = np.dot(X_train.transpose(), y)
    W = np.dot(np.linalg.inv(A), L)
    return W

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    y = np.zeros((len(labels_train), NUM_CLASSES))
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

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    model = train(X_train, labels_train, 1) # move down and change to y_train
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    print("X_train's shape is: " + str(X_train.shape))
    print("y_train's shape is: " + str(y_train.shape))
    print("model's shape is: " + str(model.shape))

    print("X_train looks like: " + str(X_train))
    print("labels_train looks like: " + str(labels_train))

    print("y_train looks like: " + str(y_train))
    print("model looks like: " + str(model))

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)


    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
