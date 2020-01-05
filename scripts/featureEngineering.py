import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import scale
from category_encoders import BinaryEncoder, OrdinalEncoder

import utils

def categoryEncode(df, cols=None, mode="binary"):
    if(mode == "ordinal"):
        encoder = OrdinalEncoder(cols=cols, handle_missing="return_nan", handle_unknown="return_nan")
    elif(mode == "binary"):
        encoder = BinaryEncoder(cols=cols)
    df_new = encoder.fit_transform(df)
    return df_new

def splitListsToColumn(df, cols, elementsPerList=3):
    df_new = df.copy() 
    for col in cols:
        col_old = df.loc[:, col]
        col_new = [[np.nan for _ in range(len(col_old))] for _ in range(elementsPerList)]
        for i in range(elementsPerList):
            for rowInd in range(len(col_old)):
                if(type(col_old[rowInd]) == list and i < len(col_old[rowInd])):
                    col_new[i][rowInd] = col_old[rowInd][i]
        for i in range(elementsPerList):
            df_new.insert(df.columns.get_loc(col) + i, col + str(i), col_new[i])
        df_new = df_new.drop(labels=[col], axis=1)
    return df_new

def imputeMissingValues(df):
    columns = df.columns
    X, y = utils.getXandy(df)
    X_pred = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(X)
    df_new = pd.DataFrame(columns=columns[:-1], data=X_pred)
    df_new["oscar"] = y
    return df_new

    
def scaleAndCenterValues(df, cols):
    df_new = df.copy() 
    for col in cols:
        df_new[col] = scale(X=df.loc[:, col])
    return df_new

def undersample(df):
    df_new = pd.DataFrame()
    count = list(df["oscar"].value_counts())
    lessLabel = 0 if count[0] < count[1] else 1
    df_new = pd.concat([df[df["oscar"] == 0].sample(count[lessLabel]), df[df["oscar"] == 1].sample(count[lessLabel])])
    return df_new

''' FIXARE '''
def dropRowsWithMissingValues(df, cols=None, thresh=None, mode="rows"):
    axis = 0 if mode == "rows" else 1
    if(thresh == None):
        return df.dropna(subset=cols, axis=axis)
    else:
        threshold = int(df.shape[1] * thresh if mode == "rows" else df.shape[0] * thresh)
        return df.dropna(subset=cols, thresh=threshold, axis=axis)





