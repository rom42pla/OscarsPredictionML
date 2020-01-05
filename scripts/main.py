#!python3

import sys
import os
import pandas as pd

import utils, parsing, featureEngineering, images, classification, evaluation

def getXandy(df):
    return df.loc[:, :"otherPrizes"].to_numpy(), df.loc[:, "oscar"].to_numpy()

def splitDataframe(dataframe, trainingPercentage=0.8):
    lastTrainingRow = int(dataframe.shape[0] * trainingPercentage)
    trainingSet = dataframe.loc[:lastTrainingRow , :]
    testSet = dataframe.loc[lastTrainingRow+1: , :]
    return (trainingSet, testSet)

'''
P A R A M E T E R S
'''
# paths
datasetPath = "../data/dataset.tsv"
preprocessedDatasetPath = "../data/dataset_preprocessed.tsv"
# how many movies of the dataset should the program analyze
linesCap = None

# reads the preprocessed file
readPreprocessedFile = True
# number of columns in which split list columns
elementsPerList = 5
# minimum percentage of non-null values for each row
threshold = 0.85

# size of X_train and y_train in respect to X and y
testSize = 0.2

# number of most important features to show in the evaluation phase
importantFeaturesNo = 10

if __name__ == "__main__":
    totalTimer, sectionTimer = utils.Timer(), utils.Timer()
    totalTimer.restartTimer()

    '''
    P A R S I N G
    '''
    print(f"====\t####\t====\t####\t====\t####\t====\t####\t====\nP A R S I N G\n====\t####\t====\t####\t====\t####\t====\t####\t====")
    df = pd.DataFrame()
    if(not readPreprocessedFile):
        print(f"Parsing and preprocessing the dataset\n")
        # parses the file to a dataframe
        df = parsing.parseFileToDataframe(filepath=datasetPath, printDataframe=True, rows=linesCap)
        print()

        '''
        F E A T U R E
        E N G I N E E R I N G
        '''
        print(f"====\t####\t====\t####\t====\t####\t====\t####\t====\nF E A T U R E\nE N G I N E E R I N G\n====\t####\t====\t####\t====\t####\t====\t####\t====")
        print(f"Removing useless columns and rows...")
        sectionTimer.restartTimer()
        # deletes some unnecessary features
        df = df.drop(labels=["ID", "title", "storyline"], axis=1).reset_index(drop=True)
        df = featureEngineering.dropRowsWithMissingValues(df=df, cols=["genre", "director", "cast", "otherPrizes", "oscar"], mode="rows").reset_index(drop=True)
        df = featureEngineering.dropRowsWithMissingValues(df=df, thresh=threshold, mode="rows").reset_index(drop=True)
        # removes columns with more than 50% of null values
        nullDistribution = df.isna().sum().map(lambda feature: (feature/df.shape[0])).to_frame().transpose()
        images.plotBarChart(x=list(nullDistribution.columns), y=list(nullDistribution.iloc[0, :]), title="Null values per column before drop")
        df = featureEngineering.dropRowsWithMissingValues(df=df, thresh=0.5, mode="columns").reset_index(drop=True)
        nullDistribution = df.isna().sum().map(lambda feature: (feature/df.shape[0]), na_action="ignore").to_frame().transpose()
        images.plotBarChart(x=list(nullDistribution.columns), y=list(nullDistribution.iloc[0, :]), title="Null values per column after drop")
        #print(df)
        print(f"\t...done in {sectionTimer.getHumanReadableElapsedTime()}")

        print(f"Scaling and centering integer values...")
        sectionTimer.restartTimer()
        # centers and scales number values
        colsToCenter = list(df.columns & {"runningTime", "budget", "income", "imdbReviews", "metascoreReviews", "imdbRating", "metascoreRating"})
        df = featureEngineering.scaleAndCenterValues(df=df, cols=colsToCenter)
        #print(df)
        print(f"\t...done in {sectionTimer.getHumanReadableElapsedTime()}")

        print(f"Undersampling the dataframe...")
        sectionTimer.restartTimer()
        # undersamples the dataframe
        trueValues = df.loc[:, "oscar"].sum()
        images.plotPieChart(x=[trueValues, df.shape[0] - trueValues], labels=["Movies that have won at least an oscar", "Movies that haven't won any oscar"], title="Categories distribution before undersampling")
        df = featureEngineering.undersample(df).reset_index(drop=True)
        trueValues = df.loc[:, "oscar"].sum()
        images.plotPieChart(x=[trueValues, df.shape[0] - trueValues], labels=["Movies that have won at least an oscar", "Movies that haven't won any oscar"], title="Categories distribution after undersampling")
        nullDistribution = df.isna().sum().map(lambda feature: (feature/df.shape[0]), na_action="ignore").to_frame().transpose()
        images.plotBarChart(x=list(nullDistribution.columns), y=list(nullDistribution.iloc[0, :]), title="Null values per column after undersampling")
        #print(df)
        print(f"\t...done in {sectionTimer.getHumanReadableElapsedTime()}")

        print(f"Binary encoding list features and dealing with null values...")
        sectionTimer.restartTimer()
        # splits some lists values into separate columns
        df = featureEngineering.splitListsToColumn(df = df, cols=["genre", "cast", "company", "country"], elementsPerList=elementsPerList)
        # removes columns with more than 90% of null values
        df = featureEngineering.dropRowsWithMissingValues(df=df, thresh=0.5, mode="columns").reset_index(drop=True)
        # transform every string with an integer ID
        df = featureEngineering.categoryEncode(df=df, mode="ordinal")
        # impute missing values using KNN imputer
        df = featureEngineering.imputeMissingValues(df)
        # checks what columns have at least one null value
        nullDistribution = df.isna().sum().to_frame()
        print(f"\t...there are still columns with null values? {(nullDistribution[0] != 0).all()}")
        # binary encoding the remaining features
        columns, colsToEncode = set(df.columns), ["classification", "director", "year"]
        for i in range(elementsPerList):
            for col in {"genre", "cast", "company", "country"}:
                column = col + str(i)
                if(column in columns):
                    colsToEncode.append(column)
        df = featureEngineering.categoryEncode(df=df, cols=colsToEncode, mode="binary")
        # fix some problems with last booleans (represented with 1 and 2s instead of 0 and 1s)
        df["otherPrizes"], df["oscar"] = df["otherPrizes"].replace([1, 2], [0, 1]), df["oscar"].replace([1, 2], [0, 1])
        print(df)
        parsing.writeDataframeToFile(df, preprocessedDatasetPath)
        print(f"\t...done in {sectionTimer.getHumanReadableElapsedTime()}")
        print()
    else:
        # reads the preprocessed file
        print(f"Reading the preprocessed file\n")
        df = parsing.parseFileToDataframe(filepath=preprocessedDatasetPath, printDataframe=True, rows=linesCap, clean=False)
        print()

    '''
    C L A S S I F I C A T I O N
    '''
    print(f"====\t####\t====\t####\t====\t####\t====\t####\t====\nC L A S S I F I C A T I O N\n====\t####\t====\t####\t====\t####\t====\t####\t====")
    # gets train and test sets
    columns = list(df.columns)
    X, y = classification.getXandy(df)
    X_train, X_test, y_train, y_test = classification.splitTrainAndTest(X, y, testSize=testSize)
    print(f"Predicting y_test ({y_test.shape[0]} movies) using ensemble methods...")
    sectionTimer.restartTimer()
    # predicts test set's labels
    y_pred_tuning = classification.predict(X_train, X_test, y_train, y_test, mode="ensemble", tuning=True)
    y_pred_noTuning = classification.predict(X_train, X_test, y_train, y_test, mode="ensemble", tuning=False)
    print(f"\t...done in {sectionTimer.getHumanReadableElapsedTime()}")
    print()

    '''
    E V A L U A T I O N
    '''
    print(f"====\t####\t====\t####\t====\t####\t====\t####\t====\nE V A L U A T I O N\n====\t####\t====\t####\t====\t####\t====\t####\t====")
    # evaluates features importance
    mostImportantFeatures, importances = evaluation.featureImportance(X, y, columns)
    print(f"Most important features:")
    for i in range(importantFeaturesNo):
        print(f"\t{i + 1}) {mostImportantFeatures[i]}\t({'%.3f' % (importances[i] * 100)}%)")
    print()

    # print a classification report
    print(f"Classification report with and without tuning:")
    evaluation.printClassificationReport(y_test, y_pred_tuning)
    evaluation.printClassificationReport(y_test, y_pred_noTuning)
    print()

    # print a confusion matrix
    # a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j
    print(f"Confusion matrix of tuned results:")
    evaluation.printConfusionMatrix(y_test, y_pred_tuning)
    print()

    print(f"\nEverything done in {totalTimer.getHumanReadableElapsedTime()}")