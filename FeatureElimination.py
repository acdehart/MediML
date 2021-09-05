from itertools import compress

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV


def RecursiveFeatureElimination(X_train, X_test, y_train, names):
    estimator = RandomForestClassifier()
    rfe = RFECV(estimator, cv=5, step=1)
    rfe.fit(X_train, y_train)

    X_train = rfe.transform(X_train)
    X_test = rfe.transform(X_test)

    rankings = np.array(rfe.ranking_)
    print(rankings)
    test = [rankings == 1][0]
    removed = [rankings > 1][0]
    selectedFeatures = list(compress(names, test))
    removedFeatures = list(compress(names, removed))

    print('Selected Features: ' + str(selectedFeatures))
    print('Removed Features: ' + str(removedFeatures))

    print('\n')
    return X_train, X_test, selectedFeatures