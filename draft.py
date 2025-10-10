from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve 

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


def scipy_linear_mod(x_train_real, x_test_real, y_train, y_test, transform, model):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = Normalizer().fit(x_train.iloc[:,1:])
        x_train.iloc[:, 1:] = transformer.transform(x_train.iloc[:, 1:])
        x_test.iloc[:, 1:] = transformer.transform(x_test.iloc[:, 1:])
    else:
        x_train = x_train_real
        x_test = x_test_real

    type_str = "norm" if transform else "no_norm"

    if model == 0:
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)

        y_pred = regr.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("True rings")
        plt.ylabel("Predicted rings")
        plt.title(f"Linear: y_true vs y_pred ({type_str})")
        plt.savefig(f"linreg_scatter_{type_str}.png")
        plt.close()

        # residuals histogram
        res = np.asarray(y_test) - np.asarray(y_pred)
        plt.figure()
        plt.hist(res, bins=30)
        plt.xlabel("Residual = y_true - y_pred"); plt.ylabel("count")
        plt.title(f"Linear: residuals hist ({type_str})")
        plt.savefig(f"regression resid hist {type_str}.png");
        plt.close()

        # residuals vs fitted
        plt.figure()
        plt.scatter(y_pred, res)
        plt.xlabel("Fitted")
        plt.ylabel("Residual")
        plt.title(f"Linear: residuals vs fitted ({type_str})")
        plt.savefig(f"regression resid vs fit {type_str}.png")
        plt.close()

        return rmse, rsquared

    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7) # note this is y true
        regr = linear_model.LogisticRegression(max_iter=1000)
        regr.fit(x_train, y_train_binary)

        y_probs = regr.predict_proba(x_test)[:, 1]
        y_pred = (y_probs >= 0.5)
        acc = accuracy_score(y_test_binary, y_pred)
        auc = roc_auc_score(y_test_binary, y_probs)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve ({type_str})")
        plt.tight_layout()
        plt.savefig(f"logistic roc {type_str}.png")
        plt.close()

        return acc, auc

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
        rmse_val, rsquared_val = scipy_linear_mod(x_train, x_test, y_train, y_test, False, 0)
        acc_val, auc_val = scipy_linear_mod(x_train, x_test, y_train, y_test, False, 1)
        rmse[i], rsquared[i], acc[i], auc[i] = rmse_val, rsquared_val, acc_val, auc_val

        rmse_val, rsquared_val = scipy_linear_mod(x_train, x_test, y_train, y_test, True, 0)
        acc_val, auc_val = scipy_linear_mod(x_train, x_test, y_train, y_test, True, 1)
        rmse_norm[i], rsquared_norm[i], acc_norm[i], auc_norm[i] = rmse_val, rsquared_val, acc_val, auc_val
    
    print(f"RMSE no norm: {rmse.mean()}, with norm: {rmse_norm.mean()}")
    print(f"RMSE no norm: {rsquared.mean()}, with norm: {rsquared_norm.mean()}")
    print(f"RMSE no norm: {acc.mean()}, with norm: {acc_norm.mean()}")
    print(f"RMSE no norm: {auc.mean()}, with norm: {auc_norm.mean()}")
    
#Ask about reporting for training set
if __name__ =='__main__':
    main() 
