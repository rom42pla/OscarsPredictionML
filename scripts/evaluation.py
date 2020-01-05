import numpy as np 
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

import images 

def printClassificationReport(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=["Movies without oscars", "Movies with at least one oscar"], digits=4)
    print(report)
    return report

def printConfusionMatrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\tTN:\t{tn}\tFP:\t{fp}\n\tFN:\t{fn}\tTP:\t{tp}\t")
    return tn, fp, fn, tp

def featureImportance(X, y, featureNames):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    featuresOrder = list(np.argsort(importances)[::-1])
    importances = importances[featuresOrder]

    for i, featureIndex in enumerate(featuresOrder):
        featuresOrder[i] = featureNames[featureIndex]

    # plots feature importances
    images.plotBarChart(x=featuresOrder, y=importances, title="Feature importances", textSize=8)
    images.plotBarChart(x=featuresOrder[:20], y=importances[:20], title="Feature importances (top 20)", textSize=8)

    return featuresOrder, importances