from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

def split_data(i, data_inputx, data_inputy):

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=i)

    return x_train, x_test, y_train, y_test

def neural(x_train, x_test, y_train, y_test, layers, learning): #compare with best approach from prev qu.
    mlp = MLPRegressor(hidden_layer_sizes=layers, activation='identity', solver='sgd', learning_rate_init = learning, max_iter=1000)
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

    data_inputx = df.iloc[:, 0:-1]
    data_inputy = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = split_data(1, data_inputx, data_inputy)

    transformer = MinMaxScaler().fit(x_train.iloc[:,1:])
    x_train.iloc[:, 1:] = transformer.transform(x_train.iloc[:, 1:])
    x_test.iloc[:, 1:] = transformer.transform(x_test.iloc[:, 1:])
  
    learning = np.arange(0.01, 0.1, 0.01)

    layers = [(10, ), (20, ), (30, ), 
          (10, 10), (10, 20), (10, 30),
          (10, 10), (20, 10), (30, 10),
          (10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10, 10)]
    
    print(df.describe().T)

    for j in layers:
        rmse_val = []
        for i in learning:
            rmse, rsquared = neural(x_train, x_test, y_train, y_test, j, i)
            rmse_val = np.append(rmse_val, rmse)
       
        plt.plot(learning, rmse_val, linestyle='--', label=f'{j}')
        
    plt.xlabel('learning rate')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('plot.png')
    plt.close()
   
    for j in layers:
          rsquared_val = []
            for i in learning:
                rmse, rsquared = neural(x_train, x_test, y_train, y_test, j, i)
                rsquared_val = np.append(rsquared_val, rsquared)
        
            plt.plot(learning, rsquared_val, linestyle='--', label=f'{j}')
            
    plt.xlabel('learning rate')
    plt.ylabel('R^2')
    plt.legend()
    plt.savefig('plot.png')
    plt.close()  
    
if __name__ =='__main__':
    main() 
