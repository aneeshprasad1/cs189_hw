import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, Imputer
import math, random
from scipy.io import loadmat
import matplotlib.pyplot as plt
import heapq

lm = loadmat('data/mnist_data/images.mat')
images = lm['images'].transpose(2, 0, 1)
images = images.reshape(60000,-1)
lm = loadmat('data/joke_data/joke_train.mat')
joke_train = lm['train']
joke_indices, joke_labels = [],[]
with open('data/joke_data/validation.txt','rb') as txtfile:
	for line in txtfile:
		line_list = line.split(',')
		joke_indices += [(int(line_list[0]) - 1, int(line_list[1]) - 1)]
		joke_labels += [int(line_list[-1])]
joke_labels = np.array(joke_labels)

class KMeansCluster:

	def __init__(self, k, data):
		self.k = k
		print str(k) + "-Means Clustering"
		self.data = data
		self.n, self.d = self.data.shape
		indices = [random.randint(0, self.n) for i in range(self.k)]
		self.means = self.data[indices]
		self.Y = np.zeros(self.n)
		self.cutoff = 50

	def cluster(self):
		err = 1000
		for i in range(self.cutoff):
			print "Round " + str(i)
			self.updateLabels()
			newErr = self.error()
			print "Loss: " + str(newErr)
			if newErr == err:
				break
			else:
				err = newErr
			self.updateMeans()	

	def updateMeans(self):
		count = np.zeros(self.k)
		newMeans = np.zeros((self.k, self.d))
		for i in range(self.n):
			label, point = self.Y[i], self.data[i]
			count[label] += 1
			newMeans[label] += point
		for label in range(self.k):
			newMeans[label] *= 1 / float(count[label])
		self.means = newMeans

	def updateLabels(self):
		newY = np.zeros(self.n)
		for i in range(self.n):
			point = self.data[i]
			distances = [np.linalg.norm(point - self.means[j]) for j in range(self.k)]
			newY[i] = np.argmin(distances)
		self.Y = newY

	def error(self):
		result = 0
		for i in range(self.n):
			point, label = self.data[i], self.Y[i]
			result += np.linalg.norm(point - self.means[label]) ** 2
		return result


# 1 Part I
def viewClusterCenters():
	for k in [5, 10, 20]:
		kmclust = KMeansCluster(k, images)
		kmclust.cluster()
		print kmclust.error()
		for i in range(k):
			centerToPlot = kmclust.means[i]
			centerToPlot = centerToPlot.reshape(28, 28)
			plt.imshow(centerToPlot)
			plt.show()

# 1 Part II
def observeKMeansError():
	for i in range(3):
		kmc = KMeansCluster(10, images)
		kmc.cluster()
		print kmc.error()

# 2 Warm-up Part I
def averagePredictor():
	fillBlanks = Imputer(missing_values='NaN', strategy='mean')
	averageMatrix = fillBlanks.fit_transform(joke_train)

	prediction = []
	for tup in joke_indices:
		val = averageMatrix[tup]
		if val > 0:
			prediction += [1]
		else:
			prediction += [0]

	prediction = np.array(prediction)
	errors = prediction != joke_labels
	print sum(errors) / float(len(joke_labels))

# 2 Warm-up Part II
def kNearestNeighbors():
	n, d = joke_train.shape
	data = np.nan_to_num(joke_train)
	for k in [10, 100, 1000]:
		print "k=" + str(k)
		prediction = []
		for i in range(len(joke_indices)):
			print "tuple " + str(i)
			user, joke = joke_indices[i]
			hp = []
			for j in range(n):
				if j == user:
					continue
				point = data[j]
				dist = -np.linalg.norm(data[user] - point)
				if len(hp) < k:
					heapq.heappush(hp, (dist, j))
				elif dist > hp[0][0]:
					heapq.heapreplace(hp, (dist, j))

			vals = np.array([data[(pair[1], joke)] for pair in hp])
			if np.count_nonzero(vals):
				avgRating = sum(vals) / float(np.count_nonzero(vals))
			else:
				avgRating = 0

			if avgRating > 0:
				prediction += [1]
			else:
				prediction += [0]

		prediction = np.array(prediction)
		errors = prediction != joke_labels
		print sum(errors) / float(len(joke_labels))

# 2 Latent Factor Model Part I, II
def latentZero():
	n, d = joke_train.shape
	data = np.nan_to_num(joke_train)
	U, D, V = np.linalg.svd(data, full_matrices=False)
	D = np.diagflat(D)
	for r in [2, 5, 10, 20]:
		Ur, Dr, Vr = U[:,:r], D[:r, :r], V[:r]
		Xprime = np.dot(np.dot(Ur, Dr), Vr)
		mse = 0
		for i in range(n):
			for j in range(d):
				if not np.isnan(joke_train[(i,j)]):
					mse += (Xprime[(i,j)] - joke_train[(i,j)]) ** 2
		print "For d=" + str(r) + ", mse is " + str(mse)

		prediction = []
		for i in range(len(joke_indices)):
			user, joke = joke_indices[i]
			guess = Xprime[(user,joke)]
			if guess > 0:
				prediction += [1]
			else:
				prediction += [0]

		prediction = np.array(prediction)
		errors = prediction != joke_labels
		print "For d=" + str(r) + ", error is " + str(sum(errors) / float(len(joke_labels)))

# 2 Latent Factor Model Part III, IV
def alternateUandV(l=10, epsilon=.0001, cutoff=400):
	n, d = joke_train.shape
	U, V = np.random.rand(n, 9), np.random.rand(9,d)
	for z in range(cutoff):
		i = random.randint(0, n-1)
		print "U_" +str(i)
		# gradient descent for u_i
		incr = np.zeros(9)
		for j in range(d):
			if not np.isnan(joke_train[(i, j)]):
				print joke_train[(i,j)], np.dot(U[i],V[:,j])
				incr += 1 / float(l) * (joke_train[(i,j)] - np.dot(U[i], V[:,j]))
		U[i] = U[i] + epsilon * incr

		j = random.randint(0,d-1)
		print "V_" + str(j)
		# gradient descent for v_j
		incr = np.zeros(9)
		for i in range(n):
			if not np.isnan(joke_train[(i, j)]):
				incr += 1 / float(l) * (joke_train[(i,j)] - np.dot(U[i], V[:,j]))
		V[:,j] = V[:,j] + epsilon * incr

	prediction = []
	for i in range(len(joke_indices)):
		user, joke = joke_indices[i]
		guess = np.dot(U[user], V[:, joke])
		if guess > 0:
			prediction += [1]
		else:
			prediction += [0]

	prediction = np.array(prediction)
	errors = prediction != joke_labels
	print "For d=" + str(9) + ", error is " + str(sum(errors) / float(len(joke_labels)))
	mse = 0
	for i in range(n):
		for j in range(d):
			if not np.isnan(joke_train[(i,j)]):
				mse += (np.dot(U[i],V[:,j]) - joke_train[(i,j)]) ** 2
	print "mse was "+str(mse)


# 2 Recommending Jokes 
def kagglePortion():
	indexes = []
	with open('data/joke_data/query.txt','rb') as txtfile:
		for line in txtfile:
			line_list = line.split(',')
			indexes += [(int(line_list[1]) - 1, int(line_list[2]) - 1)]
	indexes = np.array(indexes)
	data = np.nan_to_num(joke_train)
	U, D, V = np.linalg.svd(data, full_matrices=False)
	D = np.diagflat(D)
	Ur, Dr, Vr = U[:, :9], D[:9, :9], V[:9]
	Xprime = np.dot(np.dot(Ur, Dr), Vr)
	prediction = []
	for i in range(len(indexes)):
		user, joke = indexes[i]
		guess = Xprime[(user, joke)]
		if guess > 0:
			prediction += [1]
		else:
			prediction += [0]

	prediction = np.array(prediction)
	with open('problem7jokes.csv','wb') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['Id','Category'])
		for i in range(len(prediction)):
			writer.writerow([i+1, prediction[i]])