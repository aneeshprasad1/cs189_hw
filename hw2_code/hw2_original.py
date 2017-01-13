from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pdb

#%matplotlib inline

NUM_CLASSES = 10

### Tuneable parameters #97% with p=3000 and var =.01
VARIANCE = 0.01 ## variance of the Gaussian Random variables
P = 3000 ## how many Gaussian Random features we want

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
    xtxreg = X_train.T.dot(X_train)+ reg*np.eye(P)
    xty = X_train.T.dot(y_train)
    W = np.random.rand(P, NUM_CLASSES)
    for i in range(num_iter):
        gradient = (xtxreg.dot(W) - xty)
        W = W - alpha*(1.0/P)*gradient
    return W

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    return np.zeros((X_train.shape[1], NUM_CLASSES))

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.eye(NUM_CLASSES)[labels_train]

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(X.dot(model), axis=1)

def phi(X, G, b):
    ''' Featurize the inputs using random Fourier features '''
    return np.sqrt(2.0/P) * np.cos(np.add(G.dot(X.T), b).T)

def gaussian_random_lift(d):
    mean = np.zeros(d)
    cov = VARIANCE * np.identity(d)
    G = np.random.multivariate_normal(mean, cov, P)
    b = np.random.uniform(0.0, 2 * np.pi, P).reshape((P, 1))
    return G, b

def show_image(X, i):
    im = X[i].reshape(28, 28)*255 #Image.fromarray(X[i].reshape(28, 28)*255)
    plt.gray()
    plt.imshow(im)

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    G, b = gaussian_random_lift(X_train.shape[1])
    X_train, X_test = phi(X_train, G, b), phi(X_test, G, b)

    # model = train(X_train, y_train, reg=0.1)
    # pred_labels_train = predict(model, X_train)
    # pred_labels_test = predict(model, X_test)
    # print("Closed form solution")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

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

    """WTF is going on here:
    ### Tuneable Random Gaussian Feature Lift Variables ###
    #FEATURE_VAR = 0.01
    #NUM_FEATURES = 3000

    # def phi_other(X, W, b):
    #     ''' Featurize the inputs using random Fourier features '''
    #     return np.sqrt(2.0/NUM_FEATURES)*np.cos(np.add(W.dot(X.T), b).T)
    #
    # def gaussian_random_lift_other(d):
    #     mean = np.zeros(d)
    #     cov = FEATURE_VAR*np.eye(d)
    #     W = np.cos(np.random.multivariate_normal(mean, cov, NUM_FEATURES))
    #     b = np.random.uniform(0.0, 2*np.pi, NUM_FEATURES).reshape((NUM_FEATURES, 1))
    #     return W, b

    # print("starting Varun's closed form")
    # W_varun, b_varun = gaussian_random_lift_other(X_train.shape[1])
    # X_train_varun, X_test_varun = phi_other(X_train, W_varun, b_varun), phi_varun(X_test, W_varun, b_varun)
    #
    # # so training funciton is not the issue, because both used the same train, but got different results
    # model_varun = train(X_train_varun, y_train, reg=0.1)
    # pred_labels_train = predict(model_varun, X_train_varun)
    # pred_labels_test = predict(model_varun, X_test_varun)
    # print("Varun's Closed form solution")
    # print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    # print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
    #
    # pdb.set_trace()
    HUH"""

    # ##### PLAYGROUND #####
    # d = X_train.shape[1]
    # mean = np.zeros(d)
    # cov = FEATURE_VAR*np.eye(d)
    # W = np.random.multivariate_normal(mean, cov, NUM_FEATURES).T
    # b = np.random.uniform(0, 2*np.pi, NUM_FEATURES)
    # X_train.T.shape
    # W.T.shape
    #
    # l = W.T.dot(X_train.T)
    # m = X_train.dot(W)
    # m.shape
    # l.shape
    # b = b.reshape(NUM_FEATURES, 1)
    # res = np.add(b, l)
    # res.shape
    #
    # mean = [0, 0]
    # cov = [[1, 0], [0, 100]]  # diagonal covariance
    # np.random.multivariate_normal(mean, cov, 5000).T
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()
    # ##### END PLAYGROUND #####
