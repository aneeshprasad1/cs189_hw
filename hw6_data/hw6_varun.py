from mnist import MNIST
import numpy as np
import sklearn.metrics as metrics
import scipy.io
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import csv

def load_data(file_name):
  mat = scipy.io.loadmat(file_name)
  data = mat['train']
  return data

def mean_sq_error(U, V, R):
  error = 0
  for i in range(U.shape[0]):
    for j in range(V.shape[1]):
      u = U[i]
      v = V[:,j]
      r_ij = R[i][j]
      if (not np.isnan(r_ij)):
        error += (u.dot(v) - r_ij) ** 2
  return error

def predict(U, V, valid):
  labels = []
  pred = []
  for entry in valid:
    i, j, label = list(map(int, entry.split(',')))
    i, j = i-1, j-1
    u, v = U[i], V[:,j]
    labels.append(label)
    pred.append(u.dot(v))
  pred, labels = np.array(pred), np.array(labels)
  pred[pred <= 0] = 0
  pred[pred > 0] = 1
  return labels, pred

def predict_kaggle(U, V, valid):
  labels = []
  pred = []
  for entry in valid:
    label, i, j = list(map(int, entry.split(',')))
    i, j = i-1, j-1
    u, v = U[i], V[:,j]
    labels.append(label)
    pred.append(u.dot(v))
  pred, labels = np.array(pred), np.array(labels)
  pred[pred <= 0] = 0
  pred[pred > 0] = 1
  return labels, pred

def trainQ3(data, d, reg):
  n, m = data.shape
  U = np.random.rand(d, n)
  V = np.random.rand(d, m)
  diff_U, diff_V = np.inf, np.inf
  cutoff = 1e-4
  iteration = 0
  while diff_U > cutoff or diff_V > cutoff:
    iteration += 1
    old_U, old_V = U, V
    U = np.linalg.inv(V.dot(V.T) + reg*np.identity(d)).dot(V).dot(data.T)
    V = np.linalg.inv(U.dot(U.T) + reg*np.identity(d)).dot(U.dot(data))
    diff_U, diff_V = np.linalg.norm(old_U - U), np.linalg.norm(old_V - V)
    print("U diff is: " + str(diff_U))
    print("V diff is: " + str(diff_V))
  print("iteration: " + str(iteration))
  return U, V



train = load_data('./joke_data/joke_train.mat')
train = np.nan_to_num(train)
train = np.sign(train) # not necessary
ranks = [2, 5, 10, 20]
reg = 100
np.random.seed(seed=45)


# ##### Question 2
# ##### Part A
# errors = []
# for d in ranks:
#   U, s, V = scipy.sparse.linalg.svds(train, d)
#   error = mean_sq_error(U, V, train)
#   errors.append(error)
# print(errors)
# plt.plot(ranks, errors)
# plt.title('MSE Error vs d')
# plt.show()
#
# #### Part B
# valid_accuracy = []
# with open('./joke_data/validation.txt') as f:
#   valid = f.read()
# valid = valid.split('\n')[0:-1]
#
# for d in ranks:
#   U, s, V = scipy.sparse.linalg.svds(train, d)
#   labels, pred = predict(U, V, valid)
#   valid_accuracy.append(metrics.accuracy_score(labels, pred))
# print(valid_accuracy)
# plt.plot(ranks, valid_accuracy)
# plt.title('Validation Accuracy vs d (including Nan=0)')
# plt.axis([2, 20, 0, 1])
# plt.show()

#### Question 3
# ##### Part A
errors = []
for d in ranks:
  U, V = trainQ3(train, d, reg)
  error = mean_sq_error(U.T, V, train)
  errors.append(error)
print(errors)
plt.plot(ranks, errors)
plt.title('MSE Error vs d')
plt.show()

#### Part B
valid_accuracy = []
with open('./joke_data/validation.txt') as f:
  valid = f.read()
valid = valid.split('\n')[0:-1]


for d in ranks:
  U, V = trainQ3(train, d, reg)
  labels, pred = predict(U.T, V, valid)
  valid_accuracy.append(metrics.accuracy_score(labels, pred))
print(valid_accuracy)
plt.plot(ranks, valid_accuracy)
plt.title('Validation Accuracy vs d')
plt.axis([2, 20, 0, 1])
plt.show()


#### KAGGLE
with open('./joke_data/query.txt') as f:
  query = f.read()
query = query.split('\n')[0:-1]

query
d = 20
U, V = trainQ3(train, d, reg)
U = U.T
valid = query

labels = []
pred = []
for entry in valid:
    label, i, j = list(map(int, entry.split(',')))
    i, j = i-1, j-1
    u, v = U[i], V[:,j]
    labels.append(label)
    pred.append(u.dot(v))
    pred, labels = np.array(pred), np.array(labels)
    pred[pred <= 0] = 0
    pred[pred > 0] = 1

ids, pred = predict_kaggle(U.T, V, query)

c = csv.writer(open("jokes.csv", "wt"))
c.writerow(['Id', 'Category'])
for i in range(len(pred)):
  c.writerow((i+1, int(pred[i])))
