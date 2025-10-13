from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

    return [max_corr, min_corr]

def split_data(i, data_inputx, data_inputy):

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=i)

    return x_train, x_test, y_train, y_test


def scipy_linear_mod(x_train_real, x_test_real, y_train, y_test, transform, model):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler().fit(x_train.iloc[:,1:])
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
        acc = accuracy_score(y_test_binary, y_pred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = regr.predict_proba(x_train)[:, 1]
        y_pred_train  = (y_probs_train >= 0.5)
        acc_train = accuracy_score(y_train_binary, y_pred_train)
        auc_train = roc_auc_score(y_train_binary, y_probs_train)

        fpr, tpr, thresh = roc_curve(y_test_binary, y_probs)
        plt.figure()
        plt.plot(fpr, tpr); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve ({type_str})")
        plt.tight_layout()
        plt.savefig(f"logistic roc {type_str}.png")
        plt.close()

        return acc, auc, acc_train, auc_train

def selected(x_train_real, x_test_real, y_train, y_test, transform, model, selected_feats):
    if transform == True:
        x_train = x_train_real.copy()
        x_test = x_test_real.copy()
        transformer = MinMaxScaler().fit(x_train.iloc[:,1:])
        x_train.iloc[:, 1:] = transformer.transform(x_train.iloc[:, 1:])
        x_test.iloc[:, 1:] = transformer.transform(x_test.iloc[:, 1:])
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
        plt.savefig(f"linreg_scatter_{type_str}_selected_Features.png")
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
        acc = accuracy_score(y_test_binary, y_pred)
        auc = roc_auc_score(y_test_binary, y_probs)

        y_probs_train = regr.predict_proba(x_train[selected_feats])[:, 1]
        y_pred_train = (y_probs_train >= 0.5)
        acc_train = accuracy_score(y_train_binary, y_pred_train)
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

def neural(x_train, x_test, y_train, y_test): #refer to trial.py
    transformer = MinMaxScaler().fit(x_train.iloc[:,1:])
    x_train.iloc[:, 1:] = transformer.transform(x_train.iloc[:, 1:])
    x_test.iloc[:, 1:] = transformer.transform(x_test.iloc[:, 1:])
  
    mlp = MLPRegressor(hidden_layer_sizes=(3, ), activation='identity', solver='sgd', learning_rate_init = 0.001, max_iter=1000)
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rsquared = r2_score(y_test, y_pred)

    y_pred_train = mlp.predict(x_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rsquared_train = r2_score(y_train, y_pred_train)

    return rmse, rsquared, rmse_train, rsquared_train
  
def process_data(df):
    df.iloc[:, 0] = df.iloc[:, 0].replace({"M": 0, "F": 1, "I": 2})

def main():
    df = pd.read_csv('abalone.data', header=None, names=["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"])
    process_data(df)
    selected_features = gen_plots(df)
    print(df)

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

    rmse_nn = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn = np.zeros(NUM_EXPERIMENTS)

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

    rmse_nn_tr = np.zeros(NUM_EXPERIMENTS)
    rsquared_nn_tr = np.zeros(NUM_EXPERIMENTS)
    
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

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = selected(x_train, x_test, y_train, y_test, True, 0, selected_features)
        acc_te, auc_te, acc_tr_i, auc_tr_i = selected(x_train, x_test, y_train, y_test, True, 1, selected_features)
        rmse_sel[i], rsquared_sel[i], acc_sel[i], auc_sel[i] = rmse_te, r2_te, acc_te, auc_te
        rmse_sel_tr[i], rsquared_sel_tr[i], acc_sel_tr[i], auc_sel_tr[i] = rmse_tr_i, r2_tr_i, acc_tr_i, auc_tr_i

        rmse_te, r2_te, rmse_tr_i, r2_tr_i = neural(x_train, x_test, y_train, y_test)
        rmse_nn[i], rsquared_nn[i] = rmse_te, r2_te
        rmse_nn_tr[i], rsquared_nn_tr[i] = rmse_tr_i, r2_tr_i

    print(f"RMSE no norm Mean: {rmse.mean()}, with norm: {rmse_norm.mean()}")
    print(f"RMSE no norm SD: {rmse.std()}, with norm: {rmse_norm.std()}")
    print(f"Rsquared no norm Mean: {rsquared.mean()}, with norm: {rsquared_norm.mean()}")
    print(f"Rsquared no norm SD: {rsquared.std()}, with norm: {rsquared_norm.std()}")

    print(f"Accuracy no norm Mean: {acc.mean()}, with norm: {acc_norm.mean()}")
    print(f"Accuracy no norm SD: {acc.std()} with norm: {acc_norm.std()}")
    print(f"AUC no norm Mean: {auc.mean()}, with norm: {auc_norm.mean()}")
    print(f"AUC no norm SD: {auc.std()} with norm: {auc_norm.std()}")
    
    print(f"RMSE of Selected Features Mean: {rmse_sel.mean()}")
    print(f"RMSE of Selected Features SD: {rmse_sel.std()}")
    print(f"Rsquared of Selected Features Mean: {rsquared_sel.mean()}")
    print(f"Rsquared of Selected Features SD: {rsquared_sel.std()}")

    print(f"Accuracy of Selected Features Mean: {acc_sel.mean()}")
    print(f"Accuracy of Selected Features SD: {acc_sel.std()}")
    print(f"AUC of Selected Features Mean: {auc_sel.mean()}")
    print(f"AUC of Selected Features SD: {auc_sel.std()}")
    
    print(f"RMSE of Neural Network Mean: {rmse_nn.mean()}")
    print(f"RMSE of Neural Network SD: {rmse_nn.std()}")
    print(f"Rsquared of Neural Network Mean: {rsquared_nn.mean()}")
    print(f"Rsquared of Neural Network SD: {rsquared_nn.std()}")

    print(f"TRAIN RMSE no norm Mean: {rmse_tr.mean()}, with norm: {rmse_norm_tr.mean()}")
    print(f"TRAIN RMSE no norm SD: {rmse_tr.std()}, with norm: {rmse_norm_tr.std()}")
    print(f"TRAIN Rsquared no norm Mean: {rsquared_tr.mean()}, with norm: {rsquared_norm_tr.mean()}")
    print(f"TRAIN Rsquared no norm SD: {rsquared_tr.std()}, with norm: {rsquared_norm_tr.std()}")

    print(f"TRAIN Accuracy no norm Mean: {acc_tr.mean()}, with norm: {acc_norm_tr.mean()}")
    print(f"TRAIN Accuracy no norm SD: {acc_tr.std()} with norm: {acc_norm_tr.std()}")
    print(f"TRAIN AUC no norm Mean: {auc_tr.mean()}, with norm: {auc_norm_tr.mean()}")
    print(f"TRAIN AUC no norm SD: {auc_tr.std()} with norm: {auc_norm_tr.std()}")

    print(f"TRAIN RMSE of Selected Features Mean: {rmse_sel_tr.mean()}")
    print(f"TRAIN RMSE of Selected Features SD: {rmse_sel_tr.std()}")
    print(f"TRAIN Rsquared of Selected Features Mean: {rsquared_sel_tr.mean()}")
    print(f"TRAIN Rsquared of Selected Features SD: {rsquared_sel_tr.std()}")

    print(f"TRAIN Accuracy of Selected Features Mean: {acc_sel_tr.mean()}")
    print(f"TRAIN Accuracy of Selected Features SD: {acc_sel_tr.std()}")
    print(f"TRAIN AUC of Selected Features Mean: {auc_sel_tr.mean()}")
    print(f"TRAIN AUC of Selected Features SD: {auc_sel_tr.std()}")

    print(f"TRAIN RMSE of Neural Network Mean: {rmse_nn_tr.mean()}")
    print(f"TRAIN RMSE of Neural Network SD: {rmse_nn_tr.std()}")
    print(f"TRAIN Rsquared of Neural Network Mean: {rsquared_nn_tr.mean()}")
    print(f"TRAIN Rsquared of Neural Network SD: {rsquared_nn_tr.std()}")


if __name__ =='__main__':
    main() 
