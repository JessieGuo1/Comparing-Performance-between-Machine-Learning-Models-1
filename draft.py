from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

import numpy as np

import pandas as pd
from pandas import DataFrame

NUM_EXPERIMENTS = 30

def scatter():
    # scatter plot, dots colored by class value
    colors = {'corr_feat_1':'red', 'corr.feat_2':'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', CHANGE='x', ring_age='y', label=key, color=colors[key])
    plt.savefig('scatter.png')
    plt.clf()
    plt.close()



def split_data(i, data_inputx, data_inputy):

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=i)

    return x_train, x_test, y_train, y_test


def scipy_linear_mod(x_train, x_test, y_train, y_test, transform, model):
    if transform == True:
        transformer = Normalizer().fit(data_inputx)
        data_inputx = transformer.transform(data_inputx)

    if model == 0:
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)

        y_pred = regr.predict(x_test)
        y_test[:, ] = np.where(y_test[:, ]>=7, 1, 0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        return rmse, rsquared

    elif model == 1:
        y_train[:, ] = np.where(y_train[:, ]>=7, 1, 0)
        regr = linear_model.LogisticRegression()
        regr.fit(x_train, y_train)

        y_pred = regr.predict_proba(x_test)[:, -1]
        y_test[:, ] = np.where(y_test[:, ]>=7, 1, 0)
        acc = accuracy_score(y_pred, y_test)
        auc = roc_auc_score(y_pred, y_test)
        
        return acc, auc
#Ask about way of comparison

def neural(x_train, x_test, y_train, y_test, transform, model): #compare with best approach from prev qu.
    if model == 0: 
        mlp = MLPRegressor(hidden_layer_sizes=(3, ), activation='identity', solver='sgd', learning_rate_init = 0.001, max_iter=1000)
        mlp.fit(x_train, y_train)
    elif model == 1:
        y_train[:, ] = np.where(y_train[:, ]>=7, 1, 0)
        mlp = MLPClassifier(hidden_layer_sizes = (3, ), activation='logistic', solver='sgd', learning_rate_init = 0.001, max_iter=1000)
        mlp.fit(x_train, y_train)

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    df.iloc[:, 0] = df.iloc[:, 0].replace({"M": 0, "F": 1})
    print(df)

    rmse = np.zeros(NUM_EXPERIMENTS)
    rsquared = np.zeros(NUM_EXPERIMENTS)
    acc = np.zeros(NUM_EXPERIMENTS)
    auc = np.zeros(NUM_EXPERIMENTS)

    data_inputx = df[:, 0:8].values
    data_inputy = df[:, -1].values

    for i in range(NUM_EXPERIMENTS):
        x_train, x_test, y_train, y_test = split_data(i, data_inputx, data_inputy)
        rmse, rsquared = scipy_linear_mod(x_train, x_test, y_train, y_test, transform, 0)
        acc, auc = scipy_linear_mod(x_train, x_test, y_train, y_test, transform, 1)
        rmse[i] = rmse
        rsquared[i] = rsquared
        acc[i] = acc
        auc[i] = auc
    
    print("RMSE_ALL: %.2f" %rmse)
    print("Rsquared: %.2f" %rsquared)
    print("Accuracy: %.2f" %acc)
    print("AUC: %.2f" %auc)

    print("RMSE_mean: %.2f" %np.mean(rmse))
    print("RMSE_sd: %.2f" %np.sd(rmse))
    print("Rsquared_mean: %.2f" %np.mean(rsquared))
    print("Rsquared_sd: %.2f" %np.sd(rsquared))
   
#Ask about reporting for training set
if __name__ =='__main__':
    main() 
