

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
