import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy import linalg
import pdb
# %matplotlib inline

def generate_data():
    theta = np.random.uniform(0, 2*np.pi, 100)
    w1, w2 = np.random.normal(0, 1, 100), np.random.normal(0, 1, 100)
    first_x1, first_x2 = 8*np.cos(theta) + w1, 8*np.sin(theta) + w2
    second_x1, second_x2 = np.random.normal(0, 1, 100), np.random.normal(0, 1, 100)
    data = np.r_[np.c_[first_x1, first_x2, np.ones(100)], np.c_[second_x1, second_x2, -1*np.ones(100)]]
    np.random.shuffle(data)
    X, y = data[:,:2], data[:,2]
    return X, y

def plot(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x1 = X[:,0]
    x2 = X[:,1]
    rgb = plt.get_cmap('jet')(y)
    ax.scatter(x1, x2, color=rgb)
    plt.show()
    # plt.savefig('/tmp/out.png')    # to save the figure to a file

def split_dataset(data, size=50000):
    return data[:size], data[size:]

def kernelize_poly(x, z, p=2):
    return np.power(x.T.dot(z) + 1, p)

def kernelize_gauss(x, z, gamma=10):
    return np.exp(-gamma*(x-z).T.dot(x-z))

def train(X_train, y_train, reg=0, kernelize=kernelize_poly):
    ''' Build a model from X_train -> y_train '''
    K = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[0]):
            K[i][j] = kernelize(X_train[i], X_train[j])
    return linalg.solve(K + reg*np.eye(K.shape[0]), y_train, sym_pos=True)

def predict(model, X_train, X_test, kernelize=kernelize_poly):
    ''' From model and data points, output prediction vectors '''
    pred = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        h = 0
        for j in range(X_train.shape[0]):
            h += model[j]*kernelize(X_train[j], X_test[i])
        pred[i] = h
    return np.where(pred>0.0, 1.0, -1.0)

def plot_contour(X_train, y_train, kernelize=kernelize_poly, reg=1e-6, h=0.1, title='polynomial kernel'):
    model = train(X_train, y_train, reg, kernelize)
    x_min, x_max = X_train[:,0].min() - 1, X_train[:,0].max() + 1
    y_min, y_max = X_train[:,1].min() - 1, X_train[:,1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    pred_grid = predict(model, X_train, grid, kernelize).reshape(xx.shape)

    plt.contourf(xx, yy, pred_grid, cmap=plt.cm.Paired)
    plt.axis('off')
    plt.title('Decision boundary with ' + title)

    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.Paired)

if __name__ == "__main__":
    X, y = generate_data()
    X_train, y_train = X, y#X[:-50], y[:-50]
    X_test, y_test = X, y#X[-50:], y[-50:]

    plot(X, y)

    model_poly = train(X_train, y_train, 1e-6)
    pred_poly_train = predict(model_poly, X_train, X_train)
    pred_poly_test = predict(model_poly, X_train, X_test)

    model_gauss = train(X_train, y_train, reg=1e-4, kernelize=kernelize_gauss)
    pred_gauss_train = predict(model_gauss, X_train, X_train, kernelize=kernelize_gauss)
    pred_gauss_test = predict(model_gauss, X_train, X_test, kernelize=kernelize_gauss)

    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_poly_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(y_test, pred_poly_test)))

    print("Train accuracy: {0}".format(metrics.accuracy_score(y_train, pred_gauss_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(y_test, pred_gauss_test)))

    # Polynomial Kernel with reg=1e-6
    plot_contour(X_train, y_train)

    # Gaussian Kernel with reg=1e-4, gamma=10
    plot_contour(X_train, y_train, kernelize=kernelize_gauss, reg=1e-4, title='gaussian_kernel gamma=10')

    # Gaussian Kernel with reg=1e-4, gamma=0.1
    kernelize_gauss_1 = lambda x,z: kernelize_gauss(x, z, gamma=0.1)
    plot_contour(X_train, y_train, kernelize=kernelize_gauss_1, reg=1e-4, title='gaussian_kernel gamma=0.1')

    # Gaussian Kernel with reg=1e-4, gamma=1e-4
    kernelize_gauss_2 = lambda x,z: kernelize_gauss(x, z, gamma=1e-4)
    plot_contour(X_train, y_train, kernelize=kernelize_gauss_2, reg=1e-4, title='gaussian_kernel gamma=1e-4')
