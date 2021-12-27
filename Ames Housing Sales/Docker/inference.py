import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("Numpy", numpy.__version__)
import scipy; print("Scipy", scipy.__version__)

import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings


def rmse (ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue,ypredicted))

def inference():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    #import model
    rr_fitted = load("rr_fitted.joblib")

    #import test data
    test_df = pd.read_csv("test.csv")

    X_test = test_df.drop(columns=["SalePrice"], axis=1)
    y_test = test_df["SalePrice"]

    print(X_test.shape)
    print(y_test.shape)

    #predict test data with model
    y_pred_rr = rr_fitted.predict(X_test)

    rr_rmse = rmse(y_test, y_pred_rr)

    print(rr_rmse)
    print("Prediction completed")


if __name__ == "__main__":
    inference()