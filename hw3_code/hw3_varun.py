import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pdb
import scipy.io
import sklearn.metrics as metrics
import csv

num_iterations = 100000
alpha = 1e-3 ### learning rate
reg = .01 ## lambda
starting_weight = .000001

mat = scipy.io.loadmat('spam')
ytrain = mat['ytrain']
Xtrain = mat['Xtrain']
Xtest = mat['Xtest']
# weights = np.atleast_2d(np.ones(Xtrain.shape[1]) * starting_weight).T

########### Preprocessing
## Standardize Data
# Xtrain = Xtrain - np.mean(Xtrain)
# Xtrain = Xtrain / np.std(Xtrain)

## Log Data
def log_lift(X):
    return np.log(X + .01)
Xtrain = log_lift(Xtrain)

## Binarize Data
# Xtrain = Xtrain > 0


# ########### BATCH GRADIENT DESCENT
# def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
#     weights = np.zeros((X_train.shape[1], 1)) + starting_weight
#     for i in range(num_iterations):
#         u_vec = scipy.special.expit(X_train.dot(weights))
#         weights = weights - alpha * (2*reg*weights - X_train.T.dot(y_train - u_vec))
#     return weights
#
# def predict(model, X):
#     train_pred = scipy.special.expit(Xtrain.dot(weights))
#     return np.round(train_pred)
#
# model = train_gd(log_lift(Xtrain), ytrain, alpha, reg, num_iterations)
# pred_y_train = predict(model, log_lift())
# print("Batch gradient descent")
# print("Train accuracy: {0}".format(metrics.accuracy_score(ytrain, pred_y_train)))
#
#
# ### info for batch accuracy
num_iter = [100, 500, 1000, 5000, 10000, 50000, 100000]
### standardize data alpha = .00001, reg=.01
train_accurary = [.7089, .728, .7301, .7286, .7286, .7409, .7507]

### Log Data alpha = 1e-7, reg = .0001
train_accurary = [.6757, .8797, .9002, .9258, .9328, .94667, .9461]

### Binarize Data alpha = 1e-4, reg = .01
train_accurary = [.9128, .93101, .9324, .93188, .93217, .9333, .9336]

weights = np.zeros((Xtrain.shape[1], 1)) + starting_weight

######### STOCHASTIC GRADIENT DESCENT
for i in range(num_iterations):
  rand = int(np.random.uniform(0, Xtrain.shape[0]))
  curr_x = np.atleast_2d(Xtrain[rand]).T
  curr_y = ytrain[rand]
  curr_u = 1 / (1 + np.exp(-1 * np.dot(weights.T, curr_x)))
  weights = weights - alpha * (2 * reg * weights - np.dot(curr_x,  curr_y - curr_u))

train_pred = 1 / (1 + np.exp(-1 * np.dot(weights.T, Xtrain.T)))
train_pred = np.round(train_pred.T)
print("Stochastic gradient descent")
print("Train accuracy: {0}".format(metrics.accuracy_score(ytrain, train_pred)))


# ### info for stochastic accuracy
num_iter = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
### standardize data alpha = 1e-3, reg=.01
train_accurary = [.5011, .607, .6408, .7057, .7028, .7060, .7257, .7255]

### Log Data alpha = 1e-3, reg = .01
train_accurary = [.62231, .903, .882, .9336, .9223, .9391, .94521, .9455]

### Binarize Data alpha = 1e-4, reg = .01
train_accurary = [.875, .795, .8869, .8689, .8692, .8826, .8949, .9026]


######### STOCHASTIC GRADIENT DESCENT DECAYING ALPHA
for i in range(num_iterations):
  rand = int(np.random.uniform(0, Xtrain.shape[0]))
  curr_x = np.atleast_2d(Xtrain[rand]).T
  curr_y = ytrain[rand]
  curr_u = 1 / (1 + np.exp(-1 * np.dot(weights.T, curr_x)))
  weights = weights - (alpha / (i+1)) * (2 * reg * weights - np.dot(curr_x,  curr_y - curr_u))

train_pred = 1 / (1 + np.exp(-1 * np.dot(weights.T, Xtrain.T)))
train_pred = np.round(train_pred.T)
print("Stochastic gradient descent- decaying alpha")
print("Train accuracy: {0}".format(metrics.accuracy_score(ytrain, train_pred)))


# ### info for stochastic accuracy
num_iter = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
### standardize data alpha = 1e-8, reg=.01
train_accurary = [.605, .6808, .6727, .6649, .6742, .6646, .6521, .65212]

### Log Data alpha = 1e-3, reg = .01
train_accurary = [.8144, .8742, .8985, .8823, .8898, .9136, .9055, .8956]


### Binarize Data alpha = 1e-4, reg = .01
train_accurary = [.6437, .9107, .8973, .9165, .9162, .9165, .9174, .91652]




###### PLOT ACCURACY
plt.plot(num_iter, train_accurary, 'r-')
plt.axis([0, 105000, .6, 1])
plt.title('Stochastic Gradient Descent Decaying Alpha- Binarize Data')
plt.show()

######## PREDICT AND WRITE TO CSV
Xtest = np.log(Xtest + .01)
test_pred = 1 / (1 + np.exp(-1 * np.dot(weights.T, Xtest.T)))
test_pred = np.round(test_pred.T)

c = csv.writer(open("kaggle.csv", "wt"))
c.writerow(['Id', 'Category'])
for i in range(len(test_pred)):
  c.writerow((i+1, int(test_pred[i])))
