import scipy.io
import scipy.sparse
import scipy.linalg
import numpy as np
import sklearn.metrics as metrics
import pdb
import matplotlib.pyplot as plt
import csv

%matplotlib inline

def svd(X, k):
    X_nan = np.nan_to_num(X)
    u,s,v = scipy.sparse.linalg.svds(X_nan, k)
    return u.dot(np.diag(s)).dot(v)

def mse(X, model):
    error = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not np.isnan(X[i][j]):
                error += (model[i][j]-X[i][j])**2
    return error

def predict(X_test, model):
    pred = np.empty(X_test.shape[0])
    for i, row in enumerate(X_test):
        user, joke = row-1
        mean = np.mean(model[user])
        if model[user][joke] >= 0:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred


def train(X_train, k=5, reg=100, threshold=1e-4, num_iter=1000):
    R = np.nan_to_num(X_train)
    U = np.random.rand(k, R.shape[0])
    V = np.random.rand(k, R.shape[1])
    for x in range(num_iter):
        old_U, old_V = U, V
        U = scipy.linalg.solve(V.dot(V.T) + reg*np.eye(k), V.dot(R.T))
        V = scipy.linalg.solve(U.dot(U.T) + reg*np.eye(k), U.dot(R))
        diff_U = np.linalg.norm(old_U-U)
        diff_V = np.linalg.norm(old_V-V)
        print("U diff is: " + str(diff_U))
        print("V diff is: " + str(diff_V))
        if diff_U < threshold or diff_V < threshold:
            break
        # U, V = new_U, new_V
    print("iteration: " + str(x))
    return U.T.dot(V)


if __name__ == '__main__':
    X = scipy.io.loadmat('joke_data/joke_train')
    X_train = X['train']
    X_validate = np.loadtxt('joke_data/validation.txt', delimiter=',', dtype=int)
    X_test = np.loadtxt('joke_data/query.txt', dtype=int, delimiter=',')
    X_test_ids, X_test = X_test[:,0], X_test[:,1:]
    X_validate, labels_validate = X_validate[:,:2], X_validate[:,2]

    dimensions = [2, 5, 10, 20]

    np.random.seed(45)

    svd_accuracies = []
    for k in dimensions:
        model = svd(X_train, k)
        error = mse(X_train, model)
        print(error)
        pred_labels = predict(X_validate, model)
        accuracy = metrics.accuracy_score(labels_validate, pred_labels)
        print("Validation accuracy: {0}".format(accuracy))
        svd_accuracies.append(accuracy)

    # np.save('svd_accuracies.npy', svd_accuracies)
    plt.plot(dimensions, svd_accuracies)
    plt.show()

    closed_accuracies = []
    for k in dimensions:
        model = train(np.sign(X_train), k)
        error = mse(X_train, model)
        print(error)
        pred_labels_closed = predict(X_validate, model)
        accuracy = metrics.accuracy_score(labels_validate, pred_labels_closed)
        print("Validation accuracy: {0}".format(accuracy))
        closed_accuracies.append(accuracy)
    # np.save("closed_accuracies.npy", closed_accuracies)
    plt.plot(dimensions, closed_accuracies)
    plt.show()


    model = train(np.sign(X_train), 20)
    pred_labels_test = predict(X_test, model)

    c = csv.writer(open("kaggle.csv", "wt"))
    c.writerow(['Id', 'Category'])
    for i in range(len(pred)):
      c.writerow((i+1, int(pred_labels_test[i])))
