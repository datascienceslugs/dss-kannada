# base_nn.py
#
# Anders Poirel
# Baseline neural network model for the Kannada MNIST competition
# this first one use a simple train-test split to train then evaluate the model's 
# performance without doing any more sophisticated cross-val.

import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

train = pd.read_csv('../../data/raw/train.csv')
X_pred = pd.read_csv('../../data/raw/test.csv').values

labels = train.iloc[:, 0].values
features = train.iloc[:, 1:-1].values

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size = 0.2,
                                                    random_state = 69)






