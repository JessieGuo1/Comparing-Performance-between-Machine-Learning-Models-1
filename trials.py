from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score 

import numpy as np

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

NUM_EXPERIMENTS = 30

def split_data(i, data_inputx, data_inputy):

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=i)

    return x_train, x_test, y_train, y_test

def neural(x_train, x_test, y_train, y_test, neurons, layers, learning): #compare with best approach from prev qu.
    transformer = Normalizer().fit(x_train.iloc[:,1:])
    x_train.iloc[:, 1:] = transformer.transform(x_train.iloc[:, 1:])
    x_test.iloc[:, 1:] = transformer.transform(x_test.iloc[:, 1:])
  
    mlp = MLPRegressor(hidden_layer_sizes=(neurons, ), activation='identity', solver='sgd', learning_rate_init = learning, max_iter=1000)
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rsquared = r2_score(y_test, y_pred)

    return rmse, rsquared

def process_data(df):
    df.iloc[:, 0] = df.iloc[:, 0].replace({"M": 0, "F": 1, "I": 2})


def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    process_data(df)
    # gen_plots(df)
    print(df)

    rmse_nn = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn = np.zeros(NUM_EXPERIMENTS)

    data_inputx = df.iloc[:, 0:-1]
    data_inputy = df.iloc[:, -1]
    print(data_inputx)
    print(data_inputy)

    learning = range(0.01, 0.1, 0.01)
    layers = range()
    neurons = range(3, 9, 1)

    for i in range(learning):
        for j in range(layers):
            for k in range(neurons):
                rmse_val, rsquared_val = neural(x_train, x_test, y_train, y_test, k, j, i)
                rmse_nn[i], rsquared_nn[i] = rmse_val, rsquared_val

    print(f"RMSE of Neural Network Mean: {rmse_nn.mean()}")
    print(f"RMSE of Neural Network SD: {rmse_nn.sd()}")
    print(f"Rsquared of Neural Network Mean: {rsquared_nn.mean()}")
    print(f"Rsquared of Neural Network SD: {rsquared_nn.sd()}")

#Ask about reporting for training set
if __name__ =='__main__':
    main() 
