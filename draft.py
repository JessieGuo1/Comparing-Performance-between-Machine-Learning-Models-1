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

def scatter(df, feature, target):
    plt.figure()
    plt.scatter(df[target], df[feature])
    plt.xlabel(target)
    plt.ylabel(feature)
    plt.title(f"scatter_{feature}")
    plt.savefig(f"scatter_{feature}.png")
    plt.close()

def hist(df, feature, bins=30):
    plt.figure()
    plt.hist(df[feature], bins=bins)
    plt.xlabel(feature)
    plt.ylabel("amount")
    plt.title(f"hist_{feature}")
    plt.savefig(f"hist_{feature}.png")
    plt.close()

def gen_plots(df):
    # Heatmap
    corr = df.corr()
    print(corr)
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig("heatmap.png")
    plt.close()

    ringcorrs = df.drop(columns=["sex", "rings"]).corrwith(df["rings"])
    max_corr = ringcorrs.idxmax()
    min_corr = ringcorrs.idxmin()

    scatter(df, max_corr, "rings")
    scatter(df, min_corr, "rings")

    hist(df, max_corr)
    hist(df, min_corr)
    hist(df, "rings")

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

def process_data(df):
    df.iloc[:, 0] = df.iloc[:, 0].replace({"M": 0, "F": 1, "I": 2})

def regression_stats(xtrain, x_test, y_train, ytest):
    return


def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    process_data(df)
    # gen_plots(df)
    print(df)

    rmse = np.zeros(NUM_EXPERIMENTS)
    rsquared = np.zeros(NUM_EXPERIMENTS)
    acc = np.zeros(NUM_EXPERIMENTS)
    auc = np.zeros(NUM_EXPERIMENTS)

    rmse_norm = np.zeros(NUM_EXPERIMENTS)
    rsquared_norm = np.zeros(NUM_EXPERIMENTS)
    acc_norm = np.zeros(NUM_EXPERIMENTS)
    auc_norm = np.zeros(NUM_EXPERIMENTS)

    data_inputx = df.iloc[:, 0:-1]
    data_inputy = df.iloc[:, -1]
    print(data_inputx)
    print(data_inputy)

    for i in range(NUM_EXPERIMENTS):
        x_train, x_test, y_train, y_test = split_data(i, data_inputx, data_inputy)
        rmse, rsquared = scipy_linear_mod(x_train, x_test, y_train, y_test, False, 0)
        acc, auc = scipy_linear_mod(x_train, x_test, y_train, y_test, False, 1)
        rmse[i] = rmse
        rsquared[i] = rsquared
        acc[i] = acc
        auc[i] = auc

        rmse, rsquared = scipy_linear_mod(x_train, x_test, y_train, y_test, True, 0)
        acc, auc = scipy_linear_mod(x_train, x_test, y_train, y_test, True, 1)
        rmse_norm[i] = rmse
        rsquared_norm[i] = rsquared
        acc_norm[i] = acc
        auc_norm[i] = auc
    
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
