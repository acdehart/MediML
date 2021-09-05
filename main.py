import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from itertools import compress



def get_classifier():
    return XGBClassifier(
        n_estimators=200,  # 200
        # min_sample_split = 30,
        max_depth=4,  # 4, higher values might learn too-specific examples
        # max_leaf_nodes=4,  # replaces max depth
        min_child_weight=3,  # 3, usually 1 but higher values eliminate model-specific trees
        early_stopping_rounds=10,  # Stop if loss doesn't improve in N rounds
        # eval_metric='aucpr',
        eval_metric='aucpr',
        gamma=0.1,  # 0.1 determines minimum gain for split
        reg_lambda=1,  # L2 regularization
        reg_alpha=0,
        max_delta_step=0,  # helps with unbalanced classes
        silent=1,
        booster='gbtree',
        # eta=0.3,  # Old parameter for learning rate
        scale_pos_weight=1,  # 1
        subsample=0.85,  # 0.85 Reduces the number of rows to learn from
        colsample_bytree=0.85,  # 0.85 Reduces features the tree is allowed to train on
        learning_rate=0.1,  # 0.1
        # warm_start=True,
        verbose=1,
        max_features='sqrt'
    )


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


def PrepareData():
    # Load CSV to Pandas DF
    pass


def SmoteTest(X, y):
    sm = SMOTE(sampling_strategy='all', random_state=0)
    X_new, y_new = sm.fit_resample(X, y)
    return X_new, y_new


def main():

    # COLLECT DATA
    features, y, feature_names, df = PrepareData()

    # SPLIT SAMPLES
    features_train, features_test, y_train, y_test, df_train, df_test = train_test_split(features, y, df, test_size=0.3, stratify=y, random_state=42)

    # BALANCE CLASSES
    features_train, y_train = SmoteTest(features_train, y_train)

    # REMOVE USELESS DATA
    features_train, features_test, feature_names = RecursiveFeatureElimination(features_train, features_test, y_train, feature_names)
    features_train = pd.DataFrame(features_train, columns=feature_names)
    features_test = pd.DataFrame(features_test, columns=feature_names)

    # TRAIN ML MODEL
    clf = get_classifier()
    clf_name = 'XGBoost'
    clf.fit(features_train, y_train)

    # EVALUATE
    # plot_confusion_matrix()
    # plot_decission_surface(clf, features_train, y_train, feature_names)
    # plot_tree(clf, num_trees=tree_no, rankdir='LR')




if __name__ == '__main__':
    main()

