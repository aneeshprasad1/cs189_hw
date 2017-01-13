import numpy as np
import scipy.stats
import scipy.io
import sklearn.metrics as metrics
import pdb
import pandas as pd
import sys
import csv
import math
import matplotlib.pyplot as plt
import random
import scipy.spatial


# %matplotlib inline

# np.set_printoptions(threshold=10)

### preprocessing ###
def load_dataset(filename):
    mat = scipy.io.loadmat(filename)
    return mat['images']

def flatten(mat):
    data = np.zeros((mat.shape[2], mat.shape[0]*mat.shape[1]))
    for i in range(mat.shape[2]):
        data[i] = mat[:,:,i].flatten()
    return data

def split_dataset(data, prop=0.5):
    size = int(data.shape[0]*prop)
    return data[:size], data[size:]

# def standardize(X, data_mean):
#     ''' Standardize columns to have mean 0 and unit variance
#         Geometrically whitens everything but the important region'''
#     return (X - data_mean)/255.0
#
# def normalize(X):
#     ''' Normalize so columns sum to one '''
#     return X / np.sum(X)

def show_image(X):
    im = X.reshape(28, 28)*255 #Image.fromarray(X[i].reshape(28, 28)*255)
    plt.gray()
    plt.imshow(im)

# # Step 2 - update y, fix mu
# def update_y(X, mu):
#     y = np.zeros(X.shape[0], dtype=int)
#     for i in range(X.shape[0]):
#         y[i] = np.argmin(mu.dot(X[i]))
#     return y
#
# # Step 1 - update mu, fix y
# def update_mu(X, y, k):
#     mu = np.zeros((k, X.shape[1]))
#     count = np.zeros(k)
#     for i in range(X.shape[0]):
#         count[y[i]] += 1
#         mu[y[i]] += X[i]
#     return np.divide(mu, count.reshape((1, k)).T)

def kmeans(X, k=5, num_iter=20, init='kmeans++'):
    labels = np.zeros(X.shape[0], dtype=int)
    if init == 'kmeans++':
        centroids = kmeans_plus_init(X, k)
    else:
        centroids = lloyd_init(X, k)
    for i in range(num_iter):
        print("iteration" + str(i))
        labels = update_labels(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids

def lloyd_init(X, k):
    return X[np.random.choice(X.shape[0], k)]

def kmeans_plus_init(X, k):
    first_centroid = X[np.random.choice(X.shape[0], 1)]
    centroids = first_centroid
    for i in range(k):
        sqdists = scipy.spatial.distance.cdist(centroids, X, 'sqeuclidean')
        mins = np.argmin(sqdists, axis=0)

        prob = np.empty(sqdists.shape[1])
        for pt in range(sqdists.shape[1]):
            prob[pt] = sqdists[:,pt][mins[pt]]
        prob /= np.sum(prob)

        new_centroid = X[np.random.choice(X.shape[0], 1, p=prob)]
        centroids = np.r_[centroids, new_centroid]
    return centroids

def update_labels(X, centroids):
    sqdists = scipy.spatial.distance.cdist(centroids, X, 'sqeuclidean')
    return np.argmin(sqdists, axis=0)

def update_centroids(X, labels, k):
    centroids = np.empty((k, X.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(X[labels == i], axis=0)
    return centroids

if __name__ == "__main__":
    np.random.seed(3)
    mat = load_dataset('mnist_data/images')
    # reshape
    X = flatten(mat)
    # shuffle
    np.random.shuffle(X)

    labels_5, centroids_5 = kmeans(X, init='kmeans++', k=5)
    labels_10, centroids_10 = kmeans(X, init='kmeans++', k=10)
    labels_20, centroids_20 = kmeans(X, init='kmeans++', k=20)

    np.save("labels_5", labels_5)
    np.save("centroids_5", centroids_5)
    np.save("labels_10", labels_10)
    np.save("centroids_10", centroids_10)
    np.save("labels_20", labels_20)
    np.save("centroids_20", centroids_20)

    c_5 = np.load("centroids_5.npy")
    c_10 = np.load("centroids_10.npy")
    c_20 = np.load("centroids_20.npy")
