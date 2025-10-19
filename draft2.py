from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve 
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

NUM_EXPERIMENTS = 2

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

    ringcorrs = df.drop(columns=["rings", "M", "I", "F"]).corrwith(df["rings"])
    max_corr = ringcorrs.idxmax()
    min_corr = ringcorrs.idxmin()

    scatter(df, max_corr, "rings")
    scatter(df, min_corr, "rings")

    hist(df, max_corr)
    hist(df, min_corr)
    hist(df, "rings")

    return [max_corr, min_corr]

def split_data(i, data_inputx, data_inputy):

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=i)

    return x_train, x_test, y_train, y_test


def scipy_linear_mod(x_train_real, x_test_real, y_train, y_test, transform, model):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler()
        x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
        x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])

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

        y_pred_train = regr.predict(x_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rsquared_train = r2_score(y_train, y_pred_train)

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

        return rmse, rsquared, rmse_train, rsquared_train

    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7) # note this is y true
        regr = linear_model.LogisticRegression(max_iter=1000)
        regr.fit(x_train, y_train_binary)

        y_probs = regr.predict_proba(x_test)[:, 1]
        y_pred = (y_probs >= 0.5)
        ypred = regr.predict(x_test)
        acc = accuracy_score(y_test_binary, ypred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = regr.predict_proba(x_train)[:, 1]
        y_pred_train  = (y_probs_train >= 0.5)
        ypred_train = regr.predict(x_train)
        acc_train = accuracy_score(y_train_binary, ypred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({type_str})")
        plt.tight_layout()
        plt.savefig(f"logistic roc {type_str}.png")
        plt.close()

        return acc, auc, acc_train, auc_train

def selected(x_train_real, x_test_real, y_train, y_test, transform, model, selected_feats):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler()
        x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
        x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])
    else:
        x_train = x_train_real
        x_test = x_test_real

    type_str = "norm" if transform else "no_norm"

    if model == 0:
        regr = linear_model.LinearRegression()
        regr.fit(x_train[selected_feats], y_train)

        y_pred = regr.predict(x_test[selected_feats])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        y_pred_train = regr.predict(x_train[selected_feats])
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rsquared_train = r2_score(y_train, y_pred_train)

        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("True rings")
        plt.ylabel("Predicted rings")
        plt.title(f"Linear: y_true vs y_pred ({type_str}) {str(selected_feats)}")
        plt.savefig(f"linreg_scatter_{type_str}_selected_features.png")
        plt.close()

        # residuals histogram
        res = np.asarray(y_test) - np.asarray(y_pred)
        plt.figure()
        plt.hist(res, bins=30)
        plt.xlabel("Residual = y_true - y_pred"); plt.ylabel("count")
        plt.title(f"Linear: residuals hist ({type_str}) {str(selected_feats)}")
        plt.savefig(f"regression_resid_hist_{type_str}_selected_features.png");
        plt.close()

        # residuals vs fitted
        plt.figure()
        plt.scatter(y_pred, res)
        plt.xlabel("Fitted")
        plt.ylabel("Residual")
        plt.title(f"Linear: residuals vs fitted ({type_str}) {str(selected_feats)}")
        plt.savefig(f"regression resid vs fit {type_str}_selected_features.png")
        plt.close()

        return rmse, rsquared, rmse_train, rsquared_train

    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7) # note this is y true
        regr = linear_model.LogisticRegression(max_iter=1000)
        regr.fit(x_train[selected_feats], y_train_binary)

        y_probs = regr.predict_proba(x_test[selected_feats])[:, 1]
        y_pred = (y_probs >= 0.5)
        ypred = regr.predict(x_test[selected_feats])
        acc = accuracy_score(y_test_binary, ypred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = regr.predict_proba(x_train[selected_feats])[:, 1]
        y_pred_train = (y_probs_train >= 0.5)
        ypred_train = regr.predict(x_train[selected_feats])
        acc_train = accuracy_score(y_train_binary, ypred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve ({type_str}) {str(selected_feats)}")
        plt.tight_layout()
        plt.savefig(f"logistic roc {type_str} selected features.png")
        plt.close()

        return acc, auc, acc_train, auc_train

def neural(x_train_real, x_test_real, y_train, y_test, transform, model):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler()
        x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
        x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])
    else:
        x_train = x_train_real
        x_test = x_test_real

    type_str = "norm" if transform else "no_norm"

    if model == 0:
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='identity', solver='sgd', learning_rate_init = 0.01, max_iter=1000, random_state = 1)
        mlp.fit(x_train, y_train)

        y_pred = mlp.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        y_pred_train = mlp.predict(x_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rsquared_train = r2_score(y_train, y_pred_train)

        return rmse, rsquared, rmse_train, rsquared_train
        
    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7)
        mlp = MLPClassifier(hidden_layer_sizes=(20, ), activation='logistic', solver='sgd', learning_rate_init = 0.015, max_iter=1000, random_state = 1)
        mlp.fit(x_train, y_train_binary)

        y_probs = mlp.predict_proba(x_test)[:, 1]
        y_pred = (y_probs >= 0.5)
        ypred = mlp.predict(x_test)
        acc = accuracy_score(y_test_binary, ypred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = mlp.predict_proba(x_train)[:, 1]
        y_pred_train = (y_probs_train >= 0.5)
        ypred_train = mlp.predict(x_train)
        acc_train = accuracy_score(y_train_binary, ypred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve Neural Network ({type_str})")
        plt.tight_layout()
        plt.savefig(f"logistic roc Neural Network {type_str}.png")
        plt.close()

        return acc, auc, acc_train, auc_train

def neural_selected(x_train_real, x_test_real, y_train, y_test, transform, model, selected_feats):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler()
        x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
        x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])
    else:
        x_train = x_train_real
        x_test = x_test_real

    type_str = "norm" if transform else "no_norm"

    if model == 0:
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='identity', solver='sgd', learning_rate_init = 0.01, max_iter=1000, random_state = 1)
        mlp.fit(x_train[selected_feats], y_train)

        y_pred = mlp.predict(x_test[selected_feats])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        y_pred_train = mlp.predict(x_train[selected_feats])
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rsquared_train = r2_score(y_train, y_pred_train)

        return rmse, rsquared, rmse_train, rsquared_train

    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7) # note this is y true
        mlp = MLPClassifier(hidden_layer_sizes=(20, ), activation='logistic', solver='sgd', learning_rate_init = 0.015, max_iter=1000, random_state = 1)
        mlp.fit(x_train[selected_feats], y_train_binary)

        y_probs = mlp.predict_proba(x_test[selected_feats])[:, 1]
        y_pred = (y_probs >= 0.5)
        ypred = mlp.predict(x_test[selected_feats])
        acc = accuracy_score(y_test_binary, ypred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = mlp.predict_proba(x_train[selected_feats])[:, 1]
        y_pred_train = (y_probs_train >= 0.5)
        ypred_train = mlp.predict(x_train[selected_feats])
        acc_train = accuracy_score(y_train_binary, ypred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve Neural Network ({type_str}) {str(selected_feats)}")
        plt.tight_layout()
        plt.savefig(f"logistic roc Neural Network {type_str} Selected Features.png")
        plt.close()

        return acc, auc, acc_train, auc_train

def neural_relu(x_train_real, x_test_real, y_train, y_test, transform, model):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler()
        x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
        x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])
    else:
        x_train = x_train_real
        x_test = x_test_real

    type_str = "norm" if transform else "no_norm"

    if model == 0:
        # Regression with Relu hidden layers
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='sgd',
                           learning_rate_init=0.01, max_iter=1000, random_state=1)
        mlp.fit(x_train, y_train)

        y_pred = mlp.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        y_pred_train = mlp.predict(x_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rsquared_train = r2_score(y_train, y_pred_train)

        return rmse, rsquared, rmse_train, rsquared_train

    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7)
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='sgd',
                            learning_rate_init=0.003, max_iter=1000, random_state=1)
        mlp.fit(x_train, y_train_binary)

        y_probs = mlp.predict_proba(x_test)[:, 1]
        ypred = mlp.predict(x_test)
        acc = accuracy_score(y_test_binary, ypred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = mlp.predict_proba(x_train)[:, 1]
        ypred_train = mlp.predict(x_train)
        acc_train = accuracy_score(y_train_binary, ypred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve Neural Network ReLU ({type_str})")
        plt.tight_layout()
        plt.savefig(f"logistic roc Neural Network ReLU {type_str}.png")
        plt.close()

        return acc, auc, acc_train, auc_train

def neural_selected_relu(x_train_real, x_test_real, y_train, y_test, transform, model, selected_feats):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler()
        x_train.iloc[:, 3:] = transformer.fit_transform(x_train.iloc[:, 3:])
        x_test.iloc[:, 3:] = transformer.transform(x_test.iloc[:, 3:])
    else:
        x_train = x_train_real
        x_test = x_test_real

    type_str = "norm" if transform else "no_norm"

    if model == 0:
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='sgd',
                           learning_rate_init=0.01, max_iter=1000, random_state=1)
        mlp.fit(x_train[selected_feats], y_train)

        y_pred = mlp.predict(x_test[selected_feats])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rsquared = r2_score(y_test, y_pred)

        y_pred_train = mlp.predict(x_train[selected_feats])
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rsquared_train = r2_score(y_train, y_pred_train)

        return rmse, rsquared, rmse_train, rsquared_train

    elif model == 1:
        y_train_binary = (np.asarray(y_train) >= 7)
        y_test_binary = (np.asarray(y_test) >= 7)
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='sgd',
                            learning_rate_init=0.003, max_iter=1000, random_state=1)
        mlp.fit(x_train[selected_feats], y_train_binary)

        y_probs = mlp.predict_proba(x_test[selected_feats])[:, 1]
        ypred = mlp.predict(x_test[selected_feats])
        acc = accuracy_score(y_test_binary, ypred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = mlp.predict_proba(x_train[selected_feats])[:, 1]
        ypred_train = mlp.predict(x_train[selected_feats])
        acc_train = accuracy_score(y_train_binary, ypred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve Neural Network ReLU ({type_str}) {str(selected_feats)}")
        plt.tight_layout()
        plt.savefig(f"logistic roc Neural Network ReLU {type_str} Selected Features.png")
        plt.close()

        return acc, auc, acc_train, auc_train


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
    selected_features = gen_plots(df)
    print(df)

    # BELOW ARE RELU EXPERIMENTS: DO NOT RUN

    # data_inputx_relu = df.iloc[:, 0:-1]
    # data_inputy_relu = df.iloc[:, -1]

    # rmse_nn_relu = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu = np.zeros(NUM_EXPERIMENTS)

    # rmse_nn_relu_norm = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_norm = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_norm = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_norm = np.zeros(NUM_EXPERIMENTS)

    # rmse_nn_relu_sel = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_sel = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_sel = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_sel = np.zeros(NUM_EXPERIMENTS)

    # rmse_nn_relu_sel_norm = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_sel_norm = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_sel_norm = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_sel_norm = np.zeros(NUM_EXPERIMENTS)

    # # train metrics
    # rmse_nn_relu_tr = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_tr = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_tr = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_tr = np.zeros(NUM_EXPERIMENTS)

    # rmse_nn_relu_norm_tr = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_norm_tr = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_norm_tr = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_norm_tr = np.zeros(NUM_EXPERIMENTS)

    # rmse_nn_relu_sel_tr = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_sel_tr = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_sel_tr = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_sel_tr = np.zeros(NUM_EXPERIMENTS)

    # rmse_nn_relu_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    # rsquared_nn_relu_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    # acc_nn_relu_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    # auc_nn_relu_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)

    # for i in range(NUM_EXPERIMENTS):
    #     x_train, x_test, y_train, y_test = split_data(i, data_inputx_relu, data_inputy_relu)

    #     rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural_relu(x_train, x_test, y_train, y_test, False, 0)
    #     acc_te, auc_te, acc_tr_i, auc_tr_i = neural_relu(x_train, x_test, y_train, y_test, False, 1)
    #     rmse_nn_relu[i], rsquared_nn_relu[i], acc_nn_relu[i], auc_nn_relu[i] = rmse_te, r2_te, acc_te, auc_te
    #     rmse_nn_relu_tr[i], rsquared_nn_relu_tr[i], acc_nn_relu_tr[i], auc_nn_relu_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

    #     rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural_relu(x_train, x_test, y_train, y_test, True, 0)
    #     acc_te, auc_te, acc_tr_i, auc_tr_i = neural_relu(x_train, x_test, y_train, y_test, True, 1)
    #     rmse_nn_relu_norm[i], rsquared_nn_relu_norm[i], acc_nn_relu_norm[i], auc_nn_relu_norm[i] = rmse_te, r2_te, acc_te, auc_te
    #     rmse_nn_relu_norm_tr[i], rsquared_nn_relu_norm_tr[i], acc_nn_relu_norm_tr[i], auc_nn_relu_norm_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

    #     rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural_selected_relu(x_train, x_test, y_train, y_test, False, 0, selected_features)
    #     acc_te, auc_te, acc_tr_i, auc_tr_i = neural_selected_relu(x_train, x_test, y_train, y_test, False, 1, selected_features)
    #     rmse_nn_relu_sel[i], rsquared_nn_relu_sel[i], acc_nn_relu_sel[i], auc_nn_relu_sel[i] = rmse_te, r2_te, acc_te, auc_te
    #     rmse_nn_relu_sel_tr[i], rsquared_nn_relu_sel_tr[i], acc_nn_relu_sel_tr[i], auc_nn_relu_sel_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

    #     rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural_selected_relu(x_train, x_test, y_train, y_test, True, 0, selected_features)
    #     acc_te, auc_te, acc_tr_i, auc_tr_i = neural_selected_relu(x_train, x_test, y_train, y_test, True, 1, selected_features)
    #     rmse_nn_relu_sel_norm[i], rsquared_nn_relu_sel_norm[i], acc_nn_relu_sel_norm[i], auc_nn_relu_sel_norm[i] = rmse_te, r2_te, acc_te, auc_te
    #     rmse_nn_relu_sel_norm_tr[i], rsquared_nn_relu_sel_norm_tr[i], acc_nn_relu_sel_norm_tr[i], auc_nn_relu_sel_norm_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

    # print(f"RMSE of Neural Network ReLU no norm Mean: {rmse_nn_relu.mean():.3f}, with norm: {rmse_nn_relu_norm.mean():.3f}")
    # print(f"RMSE of Neural Network ReLU no norm SD: {rmse_nn_relu.std():.3f}, with norm: {rmse_nn_relu_norm.std():.3f}")
    # print(f"Rsquared of Neural Network ReLU no norm Mean: {rsquared_nn_relu.mean():.3f}, with norm: {rsquared_nn_relu_norm.mean():.3f}")
    # print(f"Rsquared of Neural Network ReLU no norm SD: {rsquared_nn_relu.std():.3f}, with norm: {rsquared_nn_relu_norm.std():.3f}")

    # print(f"Accuracy of Neural Network ReLU no norm Mean: {acc_nn_relu.mean():.3f}, with norm: {acc_nn_relu_norm.mean():.3f}")
    # print(f"Accuracy of Neural Network ReLU no norm SD: {acc_nn_relu.std():.3f}, with norm: {acc_nn_relu_norm.std():.3f}")
    # print(f"AUC of Neural Network ReLU no norm Mean: {auc_nn_relu.mean():.3f}, with norm: {auc_nn_relu_norm.mean():.3f}")
    # print(f"AUC of Neural Network ReLU no norm SD: {auc_nn_relu.std():.3f}, with norm: {auc_nn_relu_norm.std():.3f}")

    # print(f"RMSE of Neural Network ReLU Selected Features no norm Mean: {rmse_nn_relu_sel.mean():.3f}, with norm: {rmse_nn_relu_sel_norm.mean():.3f}")
    # print(f"RMSE of Neural Network ReLU Selected Features no norm SD: {rmse_nn_relu_sel.std():.3f}, with norm: {rmse_nn_relu_sel_norm.std():.3f}")
    # print(f"Rsquared of Neural Network ReLU Selected Features no norm Mean: {rsquared_nn_relu_sel.mean():.3f}, with norm: {rsquared_nn_relu_sel_norm.mean():.3f}")
    # print(f"Rsquared of Neural Network ReLU Selected Features no norm SD: {rsquared_nn_relu_sel.std():.3f}, with norm: {rsquared_nn_relu_sel_norm.std():.3f}")

    # print(f"Accuracy of Neural Network ReLU Selected Features no norm Mean: {acc_nn_relu_sel.mean():.3f}, with norm: {acc_nn_relu_sel_norm.mean():.3f}")
    # print(f"Accuracy of Neural Network ReLU Selected Features no norm SD: {acc_nn_relu_sel.std():.3f}, with norm: {acc_nn_relu_sel_norm.std():.3f}")
    # print(f"AUC of Neural Network ReLU Selected Features no norm Mean: {auc_nn_relu_sel.mean():.3f}, with norm: {auc_nn_relu_sel_norm.mean():.3f}")
    # print(f"AUC of Neural Network ReLU Selected Features no norm SD: {auc_nn_relu_sel.std():.3f}, with norm: {auc_nn_relu_sel_norm.std():.3f}")

    # print(f"TRAIN RMSE of Neural Network ReLU Mean: {rmse_nn_relu_tr.mean():.3f}, with norm: {rmse_nn_relu_norm_tr.mean():.3f}")
    # print(f"TRAIN RMSE of Neural Network ReLU SD: {rmse_nn_relu_tr.std():.3f}, with norm: {rmse_nn_relu_norm_tr.std():.3f}")
    # print(f"TRAIN Rsquared of Neural Network ReLU Mean: {rsquared_nn_relu_tr.mean():.3f}, with norm: {rsquared_nn_relu_norm_tr.mean():.3f}")
    # print(f"TRAIN Rsquared of Neural Network ReLU SD: {rsquared_nn_relu_tr.std():.3f}, with norm: {rsquared_nn_relu_norm_tr.std():.3f}")

    # print(f"TRAIN Accuracy of Neural Network ReLU no norm Mean: {acc_nn_relu_tr.mean():.3f}, with norm: {acc_nn_relu_norm_tr.mean():.3f}")
    # print(f"TRAIN Accuracy of Neural Network ReLU no norm SD: {acc_nn_relu_tr.std():.3f}, with norm: {acc_nn_relu_norm_tr.std():.3f}")
    # print(f"TRAIN AUC of Neural Network ReLU no norm Mean: {auc_nn_relu_tr.mean():.3f}, with norm: {auc_nn_relu_norm_tr.mean():.3f}")
    # print(f"TRAIN AUC of Neural Network ReLU no norm SD: {auc_nn_relu_tr.std():.3f}, with norm: {auc_nn_relu_norm_tr.std():.3f}")

    # print(f"TRAIN RMSE of Neural Network ReLU Selected Features no norm Mean: {rmse_nn_relu_sel_tr.mean():.3f}, with norm: {rmse_nn_relu_sel_norm_tr.mean():.3f}")
    # print(f"TRAIN RMSE of Neural Network ReLU Selected Features no norm SD: {rmse_nn_relu_sel_tr.std():.3f}, with norm: {rmse_nn_relu_sel_norm_tr.std():.3f}")
    # print(f"TRAIN Rsquared of Neural Network ReLU Selected Features no norm Mean: {rsquared_nn_relu_sel_tr.mean():.3f}, with norm: {rsquared_nn_relu_sel_norm_tr.mean():.3f}")
    # print(f"TRAIN Rsquared of Neural Network ReLU Selected Features no norm SD: {rsquared_nn_relu_sel_tr.std():.3f}, with norm: {rsquared_nn_relu_sel_norm_tr.std():.3f}")

    # print(f"TRAIN Accuracy of Neural Network ReLU Selected Features no norm Mean: {acc_nn_relu_sel_tr.mean():.3f}, with norm: {acc_nn_relu_sel_norm_tr.mean():.3f}")
    # print(f"TRAIN Accuracy of Neural Network ReLU Selected Features no norm SD: {acc_nn_relu_sel_tr.std():.3f}, with norm: {acc_nn_relu_sel_norm_tr.std():.3f}")
    # print(f"TRAIN AUC of Neural Network ReLU Selected Features no norm Mean: {auc_nn_relu_sel_tr.mean():.3f}, with norm: {auc_nn_relu_sel_norm_tr.mean():.3f}")
    # print(f"TRAIN AUC of Neural Network ReLU Selected Features no norm SD: {auc_nn_relu_sel_tr.std():.3f}, with norm: {auc_nn_relu_sel_norm_tr.std():.3f}")


    rmse = np.zeros(NUM_EXPERIMENTS)
    rsquared = np.zeros(NUM_EXPERIMENTS)
    acc = np.zeros(NUM_EXPERIMENTS)
    auc = np.zeros(NUM_EXPERIMENTS)

    rmse_norm = np.zeros(NUM_EXPERIMENTS)
    rsquared_norm = np.zeros(NUM_EXPERIMENTS)
    acc_norm = np.zeros(NUM_EXPERIMENTS)
    auc_norm = np.zeros(NUM_EXPERIMENTS)
    
    rmse_sel = np.zeros(NUM_EXPERIMENTS)
    rsquared_sel = np.zeros(NUM_EXPERIMENTS)
    acc_sel = np.zeros(NUM_EXPERIMENTS)
    auc_sel = np.zeros(NUM_EXPERIMENTS)

    rmse_sel_norm = np.zeros(NUM_EXPERIMENTS)
    rsquared_sel_norm = np.zeros(NUM_EXPERIMENTS)
    acc_sel_norm = np.zeros(NUM_EXPERIMENTS)
    auc_sel_norm = np.zeros(NUM_EXPERIMENTS)

    rmse_nn = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn = np.zeros(NUM_EXPERIMENTS)
    acc_nn = np.zeros(NUM_EXPERIMENTS)
    auc_nn = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_norm = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_norm = np.zeros(NUM_EXPERIMENTS)
    acc_nn_norm = np.zeros(NUM_EXPERIMENTS)
    auc_nn_norm = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_sel = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_sel = np.zeros(NUM_EXPERIMENTS)
    acc_nn_sel = np.zeros(NUM_EXPERIMENTS)
    auc_nn_sel = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_sel_norm = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_sel_norm = np.zeros(NUM_EXPERIMENTS)
    acc_nn_sel_norm = np.zeros(NUM_EXPERIMENTS)
    auc_nn_sel_norm = np.zeros(NUM_EXPERIMENTS)


    rmse_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_tr = np.zeros(NUM_EXPERIMENTS)
    acc_tr = np.zeros(NUM_EXPERIMENTS)
    auc_tr = np.zeros(NUM_EXPERIMENTS)

    rmse_norm_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_norm_tr = np.zeros(NUM_EXPERIMENTS)
    acc_norm_tr = np.zeros(NUM_EXPERIMENTS)
    auc_norm_tr = np.zeros(NUM_EXPERIMENTS)

    rmse_sel_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_sel_tr = np.zeros(NUM_EXPERIMENTS)
    acc_sel_tr = np.zeros(NUM_EXPERIMENTS)
    auc_sel_tr = np.zeros(NUM_EXPERIMENTS)
    
    rmse_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    acc_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    auc_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_tr = np.zeros(NUM_EXPERIMENTS)
    acc_nn_tr = np.zeros(NUM_EXPERIMENTS)
    auc_nn_tr = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_norm_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_norm_tr = np.zeros(NUM_EXPERIMENTS)
    acc_nn_norm_tr = np.zeros(NUM_EXPERIMENTS)
    auc_nn_norm_tr = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_sel_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_sel_tr = np.zeros(NUM_EXPERIMENTS)
    acc_nn_sel_tr = np.zeros(NUM_EXPERIMENTS)
    auc_nn_sel_tr = np.zeros(NUM_EXPERIMENTS)

    rmse_nn_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    acc_nn_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)
    auc_nn_sel_norm_tr = np.zeros(NUM_EXPERIMENTS)


    data_inputx = df.iloc[:, 0:-1]
    data_inputy = df.iloc[:, -1]
    print(data_inputx)
    print(data_inputy)

    for i in range(NUM_EXPERIMENTS):
        x_train, x_test, y_train, y_test = split_data(i, data_inputx, data_inputy)

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = scipy_linear_mod(x_train, x_test, y_train, y_test, False, 0)
        acc_te, auc_te, acc_tr_i, auc_tr_i = scipy_linear_mod(x_train, x_test, y_train, y_test, False, 1)
        rmse[i], rsquared[i], acc[i], auc[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_tr[i], rsquared_tr[i], acc_tr[i], auc_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = scipy_linear_mod(x_train, x_test, y_train, y_test, True, 0)
        acc_te, auc_te, acc_tr_i, auc_tr_i = scipy_linear_mod(x_train, x_test, y_train, y_test, True, 1)
        rmse_norm[i], rsquared_norm[i], acc_norm[i], auc_norm[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_norm_tr[i], rsquared_norm_tr[i], acc_norm_tr[i], auc_norm_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = selected(x_train, x_test, y_train, y_test, False, 0, selected_features)
        acc_te, auc_te, acc_tr_i, auc_tr_i = selected(x_train, x_test, y_train, y_test, False, 1, selected_features)
        rmse_sel[i], rsquared_sel[i], acc_sel[i], auc_sel[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_sel_tr[i], rsquared_sel_tr[i], acc_sel_tr[i], auc_sel_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = selected(x_train, x_test, y_train, y_test, True, 0, selected_features)
        acc_te, auc_te, acc_tr_i, auc_tr_i = selected(x_train, x_test, y_train, y_test, True, 1, selected_features)
        rmse_sel_norm[i], rsquared_sel_norm[i], acc_sel_norm[i], auc_sel_norm[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_sel_norm_tr[i], rsquared_sel_norm_tr[i], acc_sel_norm_tr[i], auc_sel_norm_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural(x_train, x_test, y_train, y_test, False, 0)
        acc_te, auc_te, acc_tr_i, auc_tr_i = neural(x_train, x_test, y_train, y_test, False, 1)
        rmse_nn[i], rsquared_nn[i], acc_nn[i], auc_nn[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_nn_tr[i], rsquared_nn_tr[i], acc_nn_tr[i], auc_nn_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural(x_train, x_test, y_train, y_test, True, 0)
        acc_te, auc_te, acc_tr_i, auc_tr_i = neural(x_train, x_test, y_train, y_test, True, 1)
        rmse_nn_norm[i], rsquared_nn_norm[i], acc_nn_norm[i], auc_nn_norm[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_nn_norm_tr[i], rsquared_nn_norm_tr[i], acc_nn_norm_tr[i], auc_nn_norm_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural_selected(x_train, x_test, y_train, y_test, False, 0, selected_features)
        acc_te, auc_te, acc_tr_i, auc_tr_i = neural_selected(x_train, x_test, y_train, y_test, False, 1, selected_features)
        rmse_nn_sel[i], rsquared_nn_sel[i], acc_nn_sel[i], auc_nn_sel[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_nn_sel_tr[i], rsquared_nn_sel_tr[i], acc_nn_sel_tr[i], auc_nn_sel_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural_selected(x_train, x_test, y_train, y_test, True, 0, selected_features)
        acc_te, auc_te, acc_tr_i, auc_tr_i = neural_selected(x_train, x_test, y_train, y_test, True, 1, selected_features)
        rmse_nn_sel_norm[i], rsquared_nn_sel_norm[i], acc_nn_sel_norm[i], auc_nn_sel_norm[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_nn_sel_norm_tr[i], rsquared_nn_sel_norm_tr[i], acc_nn_sel_norm_tr[i], auc_nn_sel_norm_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

    print(f"RMSE no norm Mean: {rmse.mean():.3f}, with norm: {rmse_norm.mean():.3f}")
    print(f"RMSE no norm SD: {rmse.std():.3f}, with norm: {rmse_norm.std():.3f}")
    print(f"Rsquared no norm Mean: {rsquared.mean():.3f}, with norm: {rsquared_norm.mean():.3f}")
    print(f"Rsquared no norm SD: {rsquared.std():.3f}, with norm: {rsquared_norm.std():.3f}")

    print(f"Accuracy no norm Mean: {acc.mean():.3f}, with norm: {acc_norm.mean():.3f}")
    print(f"Accuracy no norm SD: {acc.std():.3f} with norm: {acc_norm.std():.3f}")
    print(f"AUC no norm Mean: {auc.mean():.3f}, with norm: {auc_norm.mean():.3f}")
    print(f"AUC no norm SD: {auc.std():.3f} with norm: {auc_norm.std():.3f}")
    
    print(f"RMSE of Selected Features no norm Mean: {rmse_sel.mean():.3f}, with norm: {rmse_sel_norm.mean():.3f}")
    print(f"RMSE of Selected Features no norm SD: {rmse_sel.std():.3f}, with norm: {rmse_sel_norm.std():.3f}")
    print(f"Rsquared of Selected Features no norm Mean: {rsquared_sel.mean():.3f}, with norm: {rsquared_sel_norm.mean():.3f}")
    print(f"Rsquared of Selected Features no norm SD: {rsquared_sel.std():.3f}, with norm: {rsquared_sel_norm.std():.3f}")

    print(f"Accuracy of Selected Features no norm Mean: {acc_sel.mean():.3f}, with norm: {acc_sel_norm.mean():.3f}")
    print(f"Accuracy of Selected Features no norm SD: {acc_sel.std():.3f}, with norm: {acc_sel_norm.std():.3f}")
    print(f"AUC of Selected Features no norm Mean: {auc_sel.mean():.3f}, with norm: {auc_sel_norm.mean():.3f}")
    print(f"AUC of Selected Features no norm SD: {auc_sel.std():.3f}, with norm: {auc_sel_norm.std():.3f}")
    
    print(f"RMSE of Neural Network no norm Mean: {rmse_nn.mean():.3f}, with norm: {rmse_nn_norm.mean():.3f}")
    print(f"RMSE of Neural Network no norm SD: {rmse_nn.std():.3f}, with norm: {rmse_nn_norm.std():.3f}")
    print(f"Rsquared of Neural Network no norm Mean: {rsquared_nn.mean():.3f}, with norm: {rsquared_nn_norm.mean():.3f}")
    print(f"Rsquared of Neural Network no norm SD: {rsquared_nn.std():.3f}, with norm: {rsquared_nn_norm.std():.3f}")
 
    print(f"Accuracy of Neural Network no norm Mean: {acc_nn.mean():.3f}, with norm: {acc_nn_norm.mean():.3f}")
    print(f"Accuracy of Neural Network no norm SD: {acc_nn.std():.3f}, with norm: {acc_nn_norm.std():.3f}")
    print(f"AUC of Neural Network no norm Mean: {auc_nn.mean():.3f}, with norm: {auc_nn_norm.mean():.3f}")
    print(f"AUC of Neural Network no norm SD: {auc_nn.std():.3f}, with norm: {auc_nn_norm.std():.3f}")
 
    print(f"RMSE of Neural Network Selected Features no norm Mean: {rmse_nn_sel.mean():.3f}, with norm: {rmse_nn_sel_norm.mean():.3f}")
    print(f"RMSE of Neural Network Selected Features no norm SD: {rmse_nn_sel.std():.3f}, with norm: {rmse_nn_sel_norm.std():.3f}")
    print(f"Rsquared of Neural Selected Features Network no norm Mean: {rsquared_nn_sel.mean():.3f}, with norm: {rsquared_nn_sel_norm.mean():.3f}")
    print(f"Rsquared of Neural Network Selected Features no norm SD: {rsquared_nn_sel.std():.3f}, with norm: {rsquared_nn_sel_norm.std():.3f}")

    print(f"Accuracy of Neural Network Selected Features no norm Mean: {acc_nn_sel.mean():.3f}, with norm: {acc_nn_sel_norm.mean():.3f}")
    print(f"Accuracy of Neural Network Selected Features no norm SD: {acc_nn_sel.std():.3f}, with norm: {acc_nn_sel_norm.std():.3f}")
    print(f"AUC of Neural Network Selected Features no norm Mean: {auc_nn_sel.mean():.3f}, with norm: {auc_nn_sel_norm.mean():.3f}")
    print(f"AUC of Neural Network Selected Features no norm SD: {auc_nn_sel.std():.3f}, with norm: {auc_nn_sel_norm.std():.3f}")
 

    print(f"TRAIN RMSE no norm Mean: {rmse_tr.mean():.3f}, with norm: {rmse_norm_tr.mean():.3f}")
    print(f"TRAIN RMSE no norm SD: {rmse_tr.std():.3f}, with norm: {rmse_norm_tr.std():.3f}")
    print(f"TRAIN Rsquared no norm Mean: {rsquared_tr.mean():.3f}, with norm: {rsquared_norm_tr.mean():.3f}")
    print(f"TRAIN Rsquared no norm SD: {rsquared_tr.std():.3f}, with norm: {rsquared_norm_tr.std():.3f}")

    print(f"TRAIN Accuracy no norm Mean: {acc_tr.mean():.3f}, with norm: {acc_norm_tr.mean():.3f}")
    print(f"TRAIN Accuracy no norm SD: {acc_tr.std():.3f} with norm: {acc_norm_tr.std():.3f}")
    print(f"TRAIN AUC no norm Mean: {auc_tr.mean():.3f}, with norm: {auc_norm_tr.mean():.3f}")
    print(f"TRAIN AUC no norm SD: {auc_tr.std():.3f} with norm: {auc_norm_tr.std():.3f}")

    print(f"TRAIN RMSE of Selected Features no norm Mean: {rmse_sel_tr.mean():.3f}, with norm: {rmse_sel_norm_tr.mean():.3f}")
    print(f"TRAIN RMSE of Selected Features no norm SD: {rmse_sel_tr.std():.3f}, with norm: {rmse_sel_norm_tr.std():.3f}")
    print(f"TRAIN Rsquared of Selected Features no norm Mean: {rsquared_sel_tr.mean():.3f}, with norm: {rsquared_sel_norm_tr.mean():.3f}")
    print(f"TRAIN Rsquared of Selected Features no norm SD: {rsquared_sel_tr.std():.3f}, with norm: {rsquared_sel_norm_tr.std():.3f}")

    print(f"TRAIN Accuracy of Selected Features no norm Mean: {acc_sel_tr.mean():.3f}, with norm: {acc_sel_norm_tr.mean():.3f}")
    print(f"TRAIN Accuracy of Selected Features no norm SD: {acc_sel_tr.std():.3f}, with norm: {acc_sel_norm_tr.std():.3f}")
    print(f"TRAIN AUC of Selected Features no norm Mean: {auc_sel_tr.mean():.3f}, with norm: {auc_sel_norm_tr.mean():.3f}")
    print(f"TRAIN AUC of Selected Features no norm SD: {auc_sel_tr.std():.3f}, with norm: {auc_sel_norm_tr.std():.3f}")

    print(f"TRAIN RMSE of Neural Network Mean: {rmse_nn_tr.mean():.3f}, with norm: {rmse_nn_norm_tr.mean():.3f}")
    print(f"TRAIN RMSE of Neural Network SD: {rmse_nn_tr.std():.3f}, with norm: {rmse_nn_norm_tr.std():.3f}")
    print(f"TRAIN Rsquared of Neural Network Mean: {rsquared_nn_tr.mean():.3f}, with norm: {rsquared_nn_norm_tr.mean():.3f}")
    print(f"TRAIN Rsquared of Neural Network SD: {rsquared_nn_tr.std():.3f}, with norm: {rsquared_nn_norm_tr.std():.3f}")

    print(f"TRAIN Accuracy of Neural Network no norm Mean: {acc_nn_tr.mean():.3f}, with norm: {acc_nn_norm_tr.mean():.3f}")
    print(f"TRAIN Accuracy of Neural Network no norm SD: {acc_nn_tr.std():.3f}, with norm: {acc_nn_norm_tr.std():.3f}")
    print(f"TRAIN AUC of Neural Network no norm Mean: {auc_nn_tr.mean():.3f}, with norm: {auc_nn_norm_tr.mean():.3f}")
    print(f"TRAIN AUC of Neural Network no norm SD: {auc_nn_tr.std():.3f}, with norm: {auc_nn_norm_tr.std():.3f}")
 
    print(f"TRAIN RMSE of Neural Network Selected Features no norm Mean: {rmse_nn_sel_tr.mean():.3f}, with norm: {rmse_nn_sel_norm_tr.mean():.3f}")
    print(f"TRAIN RMSE of Neural Network Selected Features no norm SD: {rmse_nn_sel_tr.std():.3f}, with norm: {rmse_nn_sel_norm_tr.std():.3f}")
    print(f"TRAIN Rsquared of Neural Network Selected Features no norm Mean: {rsquared_nn_sel_tr.mean():.3f}, with norm: {rsquared_nn_sel_norm_tr.mean():.3f}")
    print(f"TRAIN Rsquared of Neural Network Selected Features no norm SD: {rsquared_nn_sel_tr.std():.3f}, with norm: {rsquared_nn_sel_norm_tr.std():.3f}")
   
    print(f"TRAIN Accuracy of Neural Network Selected Features no norm Mean: {acc_nn_sel_tr.mean():.3f}, with norm: {acc_nn_sel_norm_tr.mean():.3f}")
    print(f"TRAIN Accuracy of Neural Network Selected Features no norm SD: {acc_nn_sel_tr.std():.3f}, with norm: {acc_nn_sel_norm_tr.std():.3f}")
    print(f"TRAIN AUC of Neural Network Selected Features no norm Mean: {auc_nn_sel_tr.mean():.3f}, with norm: {auc_nn_sel_norm_tr.mean():.3f}")
    print(f"TRAIN AUC of Neural Network Selected Features no norm SD: {auc_nn_sel_tr.std():.3f}, with norm: {auc_nn_sel_norm_tr.std():.3f}")
 

if __name__ =='__main__':
    main()
