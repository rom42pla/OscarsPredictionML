import pandas
import category_encoders
import numpy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def categoryEncode(dataframe, cols):
    encoder = category_encoders.BinaryEncoder(cols=cols)
    newDataframe = encoder.fit_transform(dataframe)
    return newDataframe

def dropRowsWithMissingValues(dataframe, cols):
    return dataframe.dropna(subset=cols)

def imputeMissingValues(dataframe, cols):
    newDataframe = dataframe.copy()
    imp = IterativeImputer(max_iter=10, random_state=0)
    for col in cols:
        newDataframe[col] = imp.fit_transform([newDataframe.loc[:, col]])
    return newDataframe 

'''
D A
R I V E D E R E
'''
def listsToColumns(dataframe, cols):
    newDataframe = dataframe.copy()
    for col in cols:
        newCols = set()
        for cell in dataframe[col].tolist():
            if(cell != None):
                newCols = newCols | set(cell)
        for newCol in newCols:
            newDataframe.insert(dataframe.columns.get_loc(col) + 1, f"{col}_{newCol}", 0)
        for row in range(0, newDataframe.shape[0]):
            cell = newDataframe.loc[row, col]
            print(col, row)
            if (cell == None):
                continue
            for newCol in cell:
                newDataframe.at[row, f"{col}_{newCol}"] = 1
    return newDataframe

