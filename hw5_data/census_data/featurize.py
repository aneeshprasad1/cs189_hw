import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, Imputer
import scipy.io

def transform_categorical(filename, categories):
    with open(filename, "r",encoding='utf-8', errors='ignore') as f:
        X = pd.read_csv(filename)
        for category in categories:
            le = LabelEncoder()
            X[category] = le.fit_transform(X[category])
        return X

if __name__ == '__main__':
    categories = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    train_file, test_file = 'train_data.csv', 'test_data.csv'

    train_data = transform_categorical(train_file, categories)
    test_data = transform_categorical(test_file, categories)

    X_train, X_test = train_data.as_matrix(), test_data.as_matrix()
    X_train, y_train = X_train[:,:-1], X_train[:,-1]

    file_dict = {}
    file_dict['training_data'] = X_train
    file_dict['training_labels'] = y_train
    file_dict['test_data'] = test
    scipy.io.savemat('census_data.mat', file_dict)
