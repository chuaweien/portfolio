import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("Numpy", numpy.__version__)
import scipy; print("Scipy", scipy.__version__)

import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import inference
import warnings

def train():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    file='Ames_Housing_Sales.csv'
    data=pd.read_csv(file)

    #finding columns with string values for one-hot encoding later
    mask = data.dtypes == np.object
    categorical_cols = data.columns[mask]
    num_ohc_cols = (data[categorical_cols]
                    .apply(lambda x: x.nunique())
                    .sort_values(ascending=False))

    # one-hot encoding
    le = LabelEncoder()
    ohc = OneHotEncoder(drop='first')

    data_ohc = data.copy()

    for col in num_ohc_cols.index:
        dat = le.fit_transform(data_ohc[col]).astype(np.int)
        data_ohc = data_ohc.drop(col, axis=1)
        new_dat = ohc.fit_transform(dat.reshape(-1, 1))
        n_cols = new_dat.shape[1]
        col_names = ['_'.join([col, str(x)]) for x in range(n_cols)]
        new_df = pd.DataFrame(new_dat.toarray(),
                              index=data_ohc.index,
                              columns=col_names)

        data_ohc = pd.concat([data_ohc, new_df], axis=1)

    #split train test set
    X_data = data_ohc.drop('SalePrice', axis=1)
    Y_data = data_ohc['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data,
                                                        test_size=0.3, random_state=42)

    #standard scaler
    s = StandardScaler()
    X_train_ss = s.fit_transform(X_train)
    X_train_ss = pd.DataFrame(X_train_ss, columns = X_train.columns)
    y_train.reset_index(inplace=True, drop=True)

    X_test_ss = s.transform(X_test)
    X_test_ss = pd.DataFrame(X_test_ss, columns = X_test.columns)
    y_test.reset_index(inplace=True, drop=True)

    print(X_train_ss.shape)
    print(y_train.shape)

    #output to csv
    trainset = pd.concat([X_train_ss, y_train], axis= 1)
    testset = pd.concat([X_test_ss, y_test], axis= 1)

    trainset.to_csv("train.csv", index=False)
    testset.to_csv("test.csv", index=False)

    #RidgeCV
    alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]

    rr = RidgeCV(alphas = alphas,
                cv=4)
    rr_fitted = rr.fit(X_train_ss, y_train)

    #dump model to joblib
    dump(rr_fitted, "rr_fitted.joblib")
    print("Training completed")


if __name__ == "__main__":
    train()
    inference.inference()

