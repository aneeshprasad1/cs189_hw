import numpy as np
import scipy.stats
import scipy.io
import sklearn.metrics as metrics
import pdb
import pandas as pd
import sys
import csv
import math
import random

class RandomForest:
    def __init__(self, size=10, depth=6, subset=0.4):
        self.size = size
        self.depth = depth
        self.subset = subset
        self.trees = []

    def train(self, train_data, train_labels, categorical=[], bag_attributes=True):
        for i in range(self.size):
            print("Training tree: ", i)
            data = np.c_[train_data, train_labels]
            rows = random.sample(range(data.shape[0]), int(self.subset*data.shape[0]))
            tree = DecisionTree(self.depth, bag_attributes)
            tree.train(data[rows][:,:-1], data[rows][:,-1], categorical)
            self.trees.append(tree)
        return self.trees

    def predict(self, test_data):
        pred = np.zeros(test_data.shape[0])
        for i, tree in enumerate(self.trees):
            print("Predicting from tree: ", i)
            pred += tree.predict(test_data)
        return np.round(pred/self.size)

class DecisionTree:
    def __init__(self, depth=10, bag_attributes=False):
        self.root = Node()
        self.depth = depth
        self.bag_attributes = bag_attributes

    def entropy(self, hist):
        if np.count_nonzero(hist) < 2:
            return 0
        prob = hist/np.sum(hist)
        return -(prob).dot(np.log2(prob))

    def impurity(self, left_label_hist, right_label_hist):
        left_entropy, left_tot = self.entropy(left_label_hist), np.sum(left_label_hist)
        right_entropy, right_tot = self.entropy(right_label_hist), np.sum(right_label_hist)
        return (left_entropy*left_tot + right_entropy*right_tot)/(left_tot+right_tot)

    def counter(self, combined):
        sort = combined[np.argsort(combined[:,0])]
        freq = np.array([int(sort[0][0]), 0, 0]).reshape((1, 3))
        for i in range(sort.shape[0]):
            val = int(sort[i][0])
            label = int(sort[i][1])
            if val != freq[-1][0]:
                freq = np.r_[freq, np.array([val, 0, 0]).reshape((1, 3))]
            freq[-1][label+1] += 1
        return freq

    def segmenter(self, data, labels, categorical):
        ''' Finds split rule (feature and threshold) with lowest impurity '''
        split_rule = (0, 0, '')
        min_impurity = sys.maxsize

        if self.bag_attributes:
            subset = random.sample(range(data.shape[1]), int(np.sqrt(data.shape[1])))
        else:
            subset = np.arange(data.shape[1])
        data = data[:,subset]
        for i, col in enumerate(data.T):
            freq = self.counter(np.c_[col, labels])
            if i in categorical:
                for j, row in enumerate(freq):
                    in_set = freq[j,1:]
                    not_in_set = np.sum(freq[:j,1:], axis=0) + np.sum(freq[j+1:,1:],axis=0)
                    imp = self.impurity(in_set, not_in_set)
                    if imp < min_impurity:
                        min_impurity = imp
                        split_rule = (subset[i], row[0], 'categorical')
            else:
                for j, row in enumerate(freq):
                    left = np.sum(freq[:j,1:], axis=0)
                    right = np.sum(freq[j:,1:], axis=0)
                    imp = self.impurity(left, right)
                    if imp < min_impurity:
                        min_impurity = imp
                        split_rule = (subset[i], row[0], 'numerical') # (feature i, threshold_value row[0]) [can add left in to indicate index of split]
        return split_rule

    def train(self, train_data, train_labels, categorical=[]):
        ''' Grows decision tree to find best splits of input data '''
        def trainer(train_data, train_labels, depth):
            node = Node()
            if depth == 0:
                node.label = np.round(np.sum(train_labels)/train_labels.shape[0])
                return node

            feature, threshold, var_type = self.segmenter(train_data, train_labels, categorical)

            data = np.c_[train_data, train_labels]
            sort = data[np.argsort(data[:,feature])]
            threshold_ind = np.where(sort[:,feature]==threshold)[0] #if too slow, can return index from segmenter

            if threshold_ind[0] == 0:
                node.label = np.round(np.sum(train_labels)/train_labels.shape[0])
                return node
            if var_type == 'categorical':
                left_data = sort[threshold_ind[0]:threshold_ind[-1]+1] # left contains category
                right_data = np.r_[sort[:threshold_ind[0]],sort[threshold_ind[-1]+1:]] # right does not contain category
            else:
                left_data = sort[:threshold_ind[0]] # discludes threshold val and less
                right_data = sort[threshold_ind[0]:] # includes threshold val and greater

            if left_data.size == 0 or right_data.size == 0:
                pdb.set_trace()

            num_ones = np.sum(sort[:,-1])
            num_zeros = sort.shape[0] - num_ones
            node.split_rule = (feature, threshold, var_type, num_zeros, num_ones)

            node.left = trainer(left_data[:,:-1], left_data[:,-1], depth-1)
            node.right = trainer(right_data[:,:-1], right_data[:,-1], depth-1)
            return node
        self.root = trainer(train_data, train_labels, self.depth)
        return self.root

    def predict(self, test_data):
        ''' Traverse tree to find best label to classify test data'''
        pred = np.zeros(test_data.shape[0])
        for i, row in enumerate(test_data):
            node = self.root
            while node.split_rule != None:
                feature, threshold, var_type, __, __ = node.split_rule
                if var_type == 'categorical':
                    if row[feature] == threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if row[feature] < threshold:
                        node = node.left
                    else:
                        node = node.right
            pred[i] = node.label
        return pred

    def __str__(self):
        return str(self.root)

class Node:
    def __init__(self, split_rule=None, left=None, right=None, label=None):
        self.split_rule = split_rule # (feature_index, threshold_value, var_type)
        self.left = left
        self.right = right
        self.label = label # set iff leaf node

    def __str__(self, level=0):
        ret = "\t"*level+repr(self)+"\n"
        if self.label != None:
            return ret
        # ret = "\t"*level+repr(self.value)+"\n"
        ret += self.left.__str__(level+1)
        ret += self.right.__str__(level+1)
        # for child in self.children:
            # ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

    def __repr__(self):
        if self.split_rule != None:
            return "Branch: " + str(self.split_rule)
        else:
            return "Leaf: " + str(self.label)

# Pre-processing
def load_dataset(filename):
    mat = scipy.io.loadmat(filename)
    return mat['training_data'], mat['test_data'], mat['training_labels']

def split_dataset(data, prop=0.5):
    size = int(data.shape[0]*prop)
    return data[:size], data[size:]

def classify(input_file, output_file, depth,  size=10, forest=False, categories=[]):
    X_train, X_test, y_train = load_dataset(input_file)
    data = np.c_[X_train, y_train.T]
    np.random.shuffle(data)
    X_train, X_validate = split_dataset(data[:,:-1])
    y_train, y_validate = split_dataset(data[:,-1:])

    if forest:
        classifier = RandomForest(size, depth)
    else:
        classifier = DecisionTree(depth)
    classifier.train(X_train, y_train, categories)

    # print(classifier)

    pred_y_train = classifier.predict(X_train)
    pred_y_validate = classifier.predict(X_validate)

    train_accuracy = metrics.accuracy_score(y_train, pred_y_train)
    validate_accuracy = metrics.accuracy_score(y_validate, pred_y_validate)
    print("Train accuracy: {0}".format(train_accuracy))
    print("Validation accuracy: {0}".format(validate_accuracy))

    pred_y_test = classifier.predict(X_test)

    c = csv.writer(open(output_file, "wt"))
    c.writerow(['Id', 'Category'])
    for i in range(len(pred_y_test)):
        c.writerow((i+1, int(pred_y_test[i])))

    return classifier

if __name__ == "__main__":

    spam_classifier = classify('spam_data/spam_data','spam_data/kaggle.csv', 8)

    categories = [1, 3, 5, 6, 7, 8 ,9 ,13]
    classifier = classify('census_data/census_data','census_data/kaggle.csv', 7, categories=categories)
