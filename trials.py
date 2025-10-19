from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def split_data(i, data_inputx, data_inputy):

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=i)

    return x_train, x_test, y_train, y_test

def neural(x_train, x_test, y_train, y_test, layers, learning): #compare with best approach from prev qu.
    mlp = MLPRegressor(hidden_layer_sizes=layers, activation='identity', solver='sgd', learning_rate_init = learning, random_state = 1, max_iter=5000)
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rsquared = r2_score(y_test, y_pred)

    return rmse, rsquared

def process_data(df):
    values = np.array(df.iloc[:, 0])
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = pd.DataFrame(onehot_encoded, columns = ["F", "I", "M"])
    print(onehot_encoded)
    df = df.drop(columns = "sex")
    df = pd.concat([onehot_encoded, df], axis=1)

    return df

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df = process_data(df)

    data_inputx = df.iloc[:, 0:-1]
    data_inputy = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = split_data(1, data_inputx, data_inputy)

    transformer = MinMaxScaler()
    x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
    x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])
  
    learning = np.array([0.001, 0.005, 0.01, 0.012, 0.015])

    layers = [(5, ), (5, 5), (10, ), (20, ), (30, ), 
          (10, 10), (10, 20), 
            (10, 30),(16, 8),
          (20, 10), (10, 10, 10, 10)]

    for j in layers:
        rmse_val = []
        for i in learning:
            try:
                rmse, rsquared = neural(x_train, x_test, y_train, y_test, j, i)
            except Exception:
                rmse = np.nan
            rmse_val.append(rmse)
        plt.plot(learning, rmse_val, linestyle='--', label=f'{j}')
        
    plt.xlabel('learning rate')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('trials_rmse_learnrate.png')
    plt.close()
   
    for j in layers:
        rsquared_val = []
        for i in learning:
            try:
                rmse, rsquared = neural(x_train, x_test, y_train, y_test, j, i)
            except Exception:
                rsquared = np.nan
            rsquared_val.append(rsquared)
        plt.plot(learning, rsquared_val, linestyle='--', label=f'{j}')
            
    plt.xlabel('learning rate')
    plt.ylabel('R^2')
    plt.legend()
    plt.savefig('trials_rsquared_learnrate.png')
    plt.close()  
    
if __name__ =='__main__':
    main() 
