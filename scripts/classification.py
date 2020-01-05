import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def getXandy(df):
    return df.loc[:, :"otherPrizes"].to_numpy(), df.loc[:, "oscar"].to_numpy()

def splitTrainAndTest(X, y, testSize=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    return X_train, X_test, y_train, y_test

def predict(X_train, X_test, y_train, y_test, mode="ensemble", testSize=0.2, tuning=False):
    if(mode == "ensemble"):
        # the label of positive movies (1)
        positiveLabel = list(filter(lambda value: "1" in str(value), set(y_train.tolist())))[0]
        
        # check if the program has to do the tuning of the hyper-parameters
        if(tuning):
            classifiers = []
            # random forest classifier
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            print(f"\t...tuning Random Forest classifier...")
            bestScore, n_estimators_best, criterion_best, max_features_best = 0, None, None, None
            for estimators in range(100, 501, 100):
                for criterion in ["gini", "entropy"]:
                    for maxFeatures in ["auto", None]:
                        classifier = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_features=maxFeatures, n_jobs=-1).fit(X_train, y_train)
                        score = f1_score(y_test, classifier.predict(X_test), pos_label=positiveLabel)
                        if(score > bestScore):
                            bestScore, n_estimators_best, criterion_best, max_features_best = score, estimators, criterion, maxFeatures
            print(f"\t\t...that reached an f1 score of {'%.3f' % score} with hyper-parameters:\tn_estimators={n_estimators_best}, criterion={criterion_best}, max_features={max_features_best}")
            classifiers.append(("Random forest", RandomForestClassifier(n_estimators=n_estimators_best, criterion=criterion_best, max_features=max_features_best, n_jobs=-1)))
            
            # naive Bayes
            # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
            print(f"\t...predicting using naive Bayes...")
            classifier = GaussianNB().fit(X_train, y_train)
            score = f1_score(y_test, classifier.predict(X_test), pos_label=positiveLabel)
            print(f"\t\t...that reached an f1 score of {'%.3f' % score} with default parameters")
            classifiers.append(("Naive Bayes", GaussianNB()))

            # K nearest neighbors classifier
            # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
            print(f"\t...tuning K Nearest Neighbor classifier...")
            bestScore, n_neighbors_best, weights_best, metric_best = 0, None, None, None
            for neighbors in range(2, 21, 1):
                for weights in ["uniform", "distance"]:
                    for distanceMetric in [1, 2]:
                        classifier = KNeighborsClassifier(n_neighbors=neighbors, weights=weights, p=distanceMetric, n_jobs=-1).fit(X_train, y_train)
                        score = f1_score(y_test, classifier.predict(X_test), pos_label=positiveLabel)
                        if(score > bestScore):
                            bestScore, n_neighbors_best, weights_best, metric_best = score, neighbors, weights, distanceMetric
            print(f"\t\t...that reached an f1 score of {'%.3f' % score} with hyper-parameters:\tn_neighbors={n_neighbors_best}, weights={weights_best}, p={metric_best}")
            classifiers.append(("KNN", KNeighborsClassifier(n_neighbors=n_neighbors_best, weights=weights_best, p=metric_best, n_jobs=-1)))
            
            # MLP classifier
            # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
            print(f"\t...tuning Multilayer Perceptron (MLP) classifier...")
            bestScore, activation_best, learning_rate_best, solver_best = 0, None, None, None
            for activation in ["identity", "logistic", "relu"]:
                for learningRate in ["constant", "invscaling", "adaptive"]:
                    for solver in ["adam", "lbfgs"]:
                        classifier = MLPClassifier(activation=activation, learning_rate=learningRate, solver=solver, max_iter=500).fit(X_train, y_train)
                        score = f1_score(y_test, classifier.predict(X_test), pos_label=positiveLabel)
                        if(score > bestScore):
                            bestScore, activation_best, learning_rate_best, solver_best = score, activation, learningRate, solver
            print(f"\t\t...that reached an f1 score of {'%.3f' % score} with hyper-parameters:\tactivation={activation_best}, learning_rate={learning_rate_best}, solver={solver_best}")
            classifiers.append(("MLP", MLPClassifier(activation=activation_best, learning_rate=learning_rate_best, solver=solver_best, max_iter=500)))

            # ADABoost classifier
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            print(f"\t...tuning ADABoost classifier...")
            bestScore, n_estimators_best, learning_rate_best= 0, None, None
            for estimators in range(50, 101, 50):
                classifier = AdaBoostClassifier(n_estimators=estimators).fit(X_train, y_train)
                score = f1_score(y_test, classifier.predict(X_test), pos_label=positiveLabel)
                if(score > bestScore):
                    bestScore, n_estimators_best = score, estimators
            print(f"\t\t...that reached an f1 score of {'%.3f' % score} with hyper-parameters:\tn_estimators={n_estimators_best}")
            classifiers.append(("ADABoost", AdaBoostClassifier(n_estimators=n_estimators_best)))

        # no tuning has to be made
        else:
            classifiers = [
                ("Random forest", RandomForestClassifier(n_jobs=-1)), 
                ("Naive Bayes", GaussianNB()),
                ("KNN", KNeighborsClassifier(n_jobs=-1)),
                ("MLP", MLPClassifier()),
                ("ADABoost", AdaBoostClassifier())
                ]

        # predicts the result
        ensembleClassifier = VotingClassifier(estimators=classifiers, voting='hard')
        ensembleClassifier = ensembleClassifier.fit(X_train, y_train)
        return ensembleClassifier.predict(X_test)

    else:
        raise Exception('Only supported mode, for now, is "ensemble"') 

