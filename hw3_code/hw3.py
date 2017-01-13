import sklearn.metrics as metrics
import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import random
import csv
import pdb

# %matplotlib inline

clamped = 0.000001
iterations = np.arange(0, 200000, 20000)

def load_dataset(filename):
    mat = scipy.io.loadmat(filename)
    return mat['Xtrain'], mat['Xtest'], mat['ytrain']

def train_gd(X_train, y_train, alpha=1e-7, reg=0.1, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    W = np.zeros((X_train.shape[1], 1))+clamped
    for i in range(num_iter):
        mu = scipy.special.expit(X_train.dot(W))
        gradient = 2*reg*W - X_train.T.dot(y_train - mu)
        W -= alpha * gradient
    return W

def train_sgd(X_train, y_train, alpha=1e-4, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    W = np.zeros((X_train.shape[1], 1))+clamped
    for i in range(num_iter):
        sample_index = random.randint(0, X_train.shape[0] - 1)
        x_i, y_i = X_train[sample_index].reshape(X_train.shape[1], 1), y_train[sample_index]
        mu = scipy.special.expit(W.T.dot(x_i))
        gradient = 2*reg*W - x_i.dot(y_i - mu)
        W -= alpha * gradient
    return W

def train_sgd_decay(X_train, y_train, alpha=1e-4, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    W = np.zeros((X_train.shape[1], 1))+clamped
    for i in range(num_iter):
        sample_index = random.randint(0, X_train.shape[0] - 1)
        x_i, y_i = X_train[sample_index].reshape(X_train.shape[1], 1), y_train[sample_index]
        mu = scipy.special.expit(W.T.dot(x_i))
        gradient = 2*reg*W - x_i.dot(y_i - mu)
        W -= (alpha/(i+1)) * gradient
    return W

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    y_pred = scipy.special.expit(X.dot(model))
    return np.round(y_pred)

### Preprocessing Techniques ###
def standardize(X):
    ''' Standardize columns to have mean 0 and unit variance '''
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def log_lift(X):
    ''' Transform the features using log lift '''
    return np.log(X + 0.01)

def binarize(X):
    ''' Binarize features into negative (0) and nonnegative (1) '''
    return np.where(X>0.0, 1.0, 0.0)

def phi(X, G, b):
    ''' Featurize the inputs using random Fourier features '''
    return np.sqrt(2.0/P) * np.cos(np.add(G.dot(X.T), b).T)

def gaussian_random_lift(d):
    mean = np.zeros(d)
    cov = VARIANCE * np.identity(d)
    G = np.random.multivariate_normal(mean, cov, P)
    b = np.random.uniform(0.0, 2 * np.pi, P).reshape((P, 1))
    return G, b


### Post processing analysis ###
def training_loss(X_train, y_train, trainer, alpha=1e-7, reg=0.1):
    training_loss = []
    for num_iter in iterations:
        model = trainer(X_train, y_train, alpha, reg, num_iter)
        pred_y_train = predict(model, X_train)
        accuracy = metrics.accuracy_score(y_train, pred_y_train)
        print("Train accuracy with " + str(num_iter) + " iterations: {0}".format(accuracy))
        training_loss.append(accuracy)
    return training_loss

def plot_training_loss(training_loss, title):
    plt.plot(iterations, training_loss, 'r-')
    plt.axis([0, 105000, .6, 1])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train = load_dataset('spam')
    X, Xt, yt = load_dataset('spam_kaggle')
    pdb.set_trace()


    """ Training losses for gradient descenet """
    # print("Batch gradient descent on log lifted data")
    # log_training_loss = training_loss(log_lift(X_train), y_train, train_gd)
    log = [0.60521739130434782, 0.93275362318840582, 0.94260869565217387, 0.94376811594202903, 0.94492753623188408, 0.94666666666666666, 0.94608695652173913, 0.94608695652173913, 0.94666666666666666, 0.94608695652173913]
    # print("Batch gradient descent on standardized data")
    # standardized_training_loss = training_loss(standardize(X_train), y_train, train_gd)
    standardized = [0.65884057971014498, 0.90927536231884054, 0.91130434782608694, 0.9127536231884058, 0.91449275362318838, 0.91565217391304343, 0.9147826086956522, 0.91594202898550725, 0.91594202898550725, 0.91623188405797107]
    # print("Batch gradient descent on binarized data")
    # binarized_training_loss = training_loss(binarize(X_train), y_train, train_gd)
    binarized = [0.39478260869565218, 0.87971014492753619, 0.89246376811594208, 0.89507246376811589, 0.90000000000000002, 0.90318840579710147, 0.90550724637681157, 0.90666666666666662, 0.91188405797101446, 0.91246376811594199]

    """ Training losses for stochastic gradient descenet """
    # print("Stochastic gradient descent on log lifted data")
    # sgd_log_training_loss = training_loss(log_lift(X_train), y_train, train_sgd, alpha=1e-4)
    sgd_log = [0.60521739130434782, 0.92695652173913046, 0.92376811594202901, 0.9301449275362319, 0.93710144927536232, 0.92985507246376808, 0.93275362318840582, 0.92985507246376808, 0.93391304347826087, 0.93652173913043479]
    # print("Stochastic gradient descent on standardized data")
    # sgd_standardized_training_loss = training_loss(standardize(X_train), y_train, train_sgd, alpha=1e-4)
    sgd_standardized = [0.65884057971014498, 0.9040579710144927, 0.90869565217391302, 0.90927536231884054, 0.91130434782608694, 0.91130434782608694, 0.9104347826086957, 0.91130434782608694, 0.91072463768115941, 0.91130434782608694]
    # print("Stochastic gradient descent on binarized data")
    # sgd_binarized_training_loss = training_loss(binarize(X_train), y_train, train_sgd, alpha=1e-3)
    sgd_binarized = [0.39478260869565218, 0.89681159420289858, 0.888695652173913, 0.89217391304347826, 0.8863768115942029, 0.88579710144927537, 0.89478260869565218, 0.89101449275362321, 0.89130434782608692, 0.88115942028985506]

    """ Training losses for stochastic gradient descenet with decaying alpha """
    # print("Decaying Alpha Stochastic gradient descent on log lifted data")
    # sgda_log_training_loss = training_loss(log_lift(X_train), y_train, train_sgd_decay, alpha=1e-3, reg=0.01)
    # print("Training Losses: " + str(sgda_log_training_loss))
    sgda_log = [0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782, 0.60521739130434782]
    # print("Decaying Alpha Stochastic gradient descent on standardized data")
    # sgda_standardized_training_loss = training_loss(standardize(X_train), y_train, train_sgd_decay, alpha=1e-3, reg=0.01)
    # print("Training Losses: " + str(sgda_standardized_training_loss))
    sgda_standardized = [0.65884057971014498, 0.88144927536231887, 0.89275362318840579, 0.87449275362318846, 0.88144927536231887, 0.89420289855072466, 0.87855072463768114, 0.888695652173913, 0.88550724637681155, 0.89507246376811589]
    # print("Decaying Alpha Stochastic gradient descent on binarized data")
    # sgda_binarized_training_loss = training_loss(binarize(X_train), y_train, train_sgd_decay, alpha=1e-3, reg=0.01)
    # print("Training Losses: " + str(sgda_binarized_training_loss))
    sgda_binarized = [0.39478260869565218, 0.70028985507246377, 0.86405797101449278, 0.87333333333333329, 0.60521739130434782, 0.87855072463768114, 0.83043478260869563, 0.71739130434782605, 0.82521739130434779, 0.7185507246376811]

    """ Plotting """
    plot_training_loss(log, "Batch gradient descent on log lifted data")
    plot_training_loss(standardized, "Batch gradient descent on standardized data")
    plot_training_loss(binarized, "Batch gradient descent on binarized data")
    plot_training_loss(sgd_log, "Stochastic gradient descent on log lifted data")
    plot_training_loss(sgd_standardized, "Stochastic gradient descent on standardized data")
    plot_training_loss(sgd_binarized, "Stochastic gradient descent on binarized data")
    plot_training_loss(sgda_log, "Stochastic gradient descent with decaying alpha on log lifted data")
    plot_training_loss(sgda_standardized, "Stochastic gradient descent with decaying alpha on standardized data")
    plot_training_loss(sgda_binarized, "Stochastic gradient descent with decaying alpha on binarized data")

    # """ Kaggle Test """
    # model = train_gd(log_lift(X_train), y_train, alpha=1e-7, reg=0.1, num_iter=100000)
    # pred_y_test = predict(model, log_lift(X_test))
    #
    # c = csv.writer(open("hw3_kaggle.csv", "wt"))
    # c.writerow(['Id', 'Category'])
    # for i, val in enumerate(pred_y_test):
    #   c.writerow((i+1, int(val)))
