from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import pdb
import csv

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
    ### From HW1
    d = len(X_train[0])
    identity_xx = np.identity(d)
    inv = np.linalg.inv(np.dot(np.transpose(X_train), X_train) + reg * identity_xx)
    outer_y = np.dot(np.transpose(X_train), y_train)
    w = np.dot(inv, outer_y)
    return w

def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    p = len(X_train[0])
    W = np.random.rand(p, y_train.shape[1])
    identity_xx = np.identity(p)
    x_t_x_reg = np.dot(np.transpose(X_train), X_train) + reg * identity_xx
    outer_y = np.dot(np.transpose(X_train), y_train)

    for i in range(num_iter):
      grad_f = -2.0/p * (np.dot(x_t_x_reg, W) - outer_y)
      W = W + alpha * grad_f
    return W


def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    p = len(X_train[0])
    W = np.random.rand(p, y_train.shape[1])
    identity_p = np.identity(p)
    for i in range(num_iter):
      # if i % 1000 == 0:
      #   print(str(i))
      rand_index = np.random.randint(0, y_train.shape[0])
      x_t_x_reg = np.dot(np.atleast_2d(X_train[rand_index]).T, np.atleast_2d(X_train[rand_index])) + reg * identity_p
      outer_y = np.dot(np.atleast_2d(X_train[rand_index]).T, np.atleast_2d(y_train[rand_index]))
      grad_f = -2.0/p * (np.dot(x_t_x_reg, W) - outer_y)
      W = W + alpha * grad_f
    return W



def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    ### From HW1
    result = np.zeros((len(labels_train), NUM_CLASSES))
    for i in range(len(labels_train)):
      result[i][labels_train[i]] = 1.0
    return result

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    ### From HW 1
    predicted = np.dot(X, model)
    y_result = np.zeros(len(predicted))
    for i in range(len(y_result)):
      y_result[i] = np.argmax(predicted[i])
    return y_result

def generate_W_B(n_train, n_test, d):
    pdb.set_trace()
    mean = np.zeros(d)
    cov = variance * np.identity(d)
    W = np.random.multivariate_normal(mean, cov, p)
    b = np.random.uniform(0.0, 2 * np.pi, p)
    B_train = np.tile(b, (n_train,1)).transpose()
    B_test = np.tile(b, (n_test,1)).transpose()
    return W, B_train, B_test

def phi(X, W, B):
    ''' Featurize the inputs using random Fourier features '''
    result = np.dot(W, X.transpose()) + B
    return np.sqrt(2.0/p) * np.cos(result.transpose())


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    ### Tuneable parameters #97% with p=3000 and var =.01
    p = 3000 ## how many GR features we want
    variance = .01 ## variance of the GR variables
    print('p: ' + str(p) + ' var: '  + str(variance))
    W, B_train, B_test = generate_W_B(X_train.shape[0], X_test.shape[0], X_train.shape[1])
    X_train, X_test = phi(X_train, W, B_train), phi(X_test, W, B_test)

    print("calculating closed form sol")
    model = train(X_train, y_train, reg=0.1)
    model.shape
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    # CSV STUFF
    # c = csv.writer(open("kaggle.csv", "wt"))
    # c.writerow(['Id', 'Category'])
    # for i in range(len(pred_labels_test)):
    #   c.writerow( (i, int(pred_labels_test[i])))


    print('starting gradient descent')
    model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=100000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    print('starting stochastic gradient descent')
    model = train_sgd(X_train, y_train, alpha=1e-1, reg=0.1, num_iter=100000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Stochastic gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
