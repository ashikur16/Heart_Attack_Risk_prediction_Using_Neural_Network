

# Importing

import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix



# Read The Dataset
def get_dataset(filename):
    return pd.read_csv(filename)



# Printing the dataset
def print_data(data,n):
    print(data.head(n))




# Cleaning The Dataset

### Detecting The None/Missing Values




def count(a):
    count1 = 0
    for i in a:
        if (i >= -np.inf and i <= np.inf):
            count1 = count1 + 1

    return count1

# Removing The Duplicate Values

def data_cleaning(data):
    # Total rows
    total_rows = len(data.index)
    print("Total Rows of The Dataset", total_rows)

    # Removing Duplicate Rows
    data.drop_duplicates(keep=False, inplace=True)

    # Total rows after removing duplicates
    total_rows_after_remove_duplicates = len(data.index)

    print("Total Rows of The Dataset After Removing Duplicates", total_rows_after_remove_duplicates)
    print("There are {} duplicates in the dataset".format(total_rows - total_rows_after_remove_duplicates))
    return data

def data_is_anyMissingData(data):
    for j in data.columns:
        c = count(data[j])
        if (c == len(data.index)):
            print("There is no missing value for column", j)
        else:
            print("None/Missing values detected for column", j)

# There is no missing value in the whole dataset



# Setting up the features with the columns of dataset except 'output'

def get_features(data):
    features = data.columns
    features = [i for i in features if i != 'output']
    # display columns of the feature
    print(features)
    return features



# Splitting the dataset into train and test dataset
def data_split_test_train(data):
    train, test = train_test_split(data, test_size=0.25)  # Train dataset is 75% of actual dataset
    print("Length Of Actual Heart Dataset: ", len(data))
    print("Length Of Train Dataset: ", len(train))
    print("Length Of Test Dataset: ", len(test))
    return train,test



# Setting Up The Train and Test Dataset Into x And y

def set_XY(train,test,features):
    x_train = train[features]
    y_train = train["output"]

    x_test = test[features]
    y_test = test["output"]

    return x_train,x_test,y_train,y_test


### Creating the neural network model

def NNModel():
    mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), max_iter=900, activation='relu')
    return mlp


### Fitting the train and test dataset in the model

def fitModel(x_train,y_train,mlp):
    mlp = mlp.fit(x_train, y_train)
    return mlp



### Prediction according to neural network algorithm

def get_prediction(mlp,x_test):
    y_pred = mlp.predict(x_test)
    print(y_pred)
    return y_pred

### Determining the Accuracy for Neural Network

def model_accuracy(y_test, y_pred):
    score_NN = accuracy_score(y_test, y_pred) * 100
    print("Accuracy using Neural Network: ", round(score_NN, 4), "%")
    return score_NN


if __name__ == '__main__':




    filename="heart.csv"

    data = get_dataset(filename)
    print()
    print_data(data,11)
    print()
    data_is_anyMissingData(data)
    print()
    data=data_cleaning(data)
    print()

    features = get_features(data)
    print()
    train, test = data_split_test_train(data)
    print()
    x_train, x_test, y_train, y_test = set_XY(train, test, features)
    print()
    mlp = NNModel()
    mlp = fitModel(x_train, y_train, mlp)
    print()
    y_pred = get_prediction(mlp, x_test)
    print()
    score_NN = model_accuracy(y_test, y_pred)
    print()
