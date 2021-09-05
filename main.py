from random import random

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from FeatureElimination import RecursiveFeatureElimination
from SmoteUpsample import SmoteTest
from XGB import get_classifier
from textblob import TextBlob
import pyroot
# from root_pandas import read_root
from convert_ntuple_df import root_to_df


def ExtractData(column_names, y_name):
    X = root_to_df('TVAERS_ntuple.root', column_names)
    print(X)
    y = X[y_name]
    feature_names = column_names.remove(y_name)
    features = X[feature_names]

    # symptoms = X['symptoms']
    # symptom_sentiment = [TextBlob(text).sentiment for text in symptoms ]

    # cv = CountVectorizer(stop_words='english')
    # data_cv = cv.fit_transform(X['symptoms'])
    # data_dtm = pd.DataFrame(data_cv.toarray(), columns=feature_names)

    return features, y, feature_names, X


def PrepareData():
    column_names = ['vaers_id', 'age_yrs', 'died', 'datedied', 'l_threat', 'er_visit', 'hospital', 'hospdays', 'x_stay',
                    'disable', 'recovd', 'vax_date', 'numdays', 'symptoms']
    features, feature_names, y, df = ExtractData(column_names=column_names, y_name='died')
    return features, y, feature_names, df


def main():
    print("*** MediML ***")
    # COLLECT DATA
    # PrepareData()
    features, y, feature_names, df = PrepareData()

    # SPLIT SAMPLES
    features_train, features_test, y_train, y_test, df_train, df_test = train_test_split(features, y, df, test_size=0.3,
                                                                                         stratify=y, random_state=42)

    # BALANCE CLASSES
    # features_train, y_train = BootStrap(features_train, y_train)
    # features_train, y_train = SmoteTest(features_train, y_train)

    # REMOVE USELESS DATA
    # features_train, features_test, feature_names = RecursiveFeatureElimination(features_train, features_test, y_train, feature_names)
    # features_train = pd.DataFrame(features_train, columns=feature_names)
    # features_test = pd.DataFrame(features_test, columns=feature_names)

    # TRAIN ML MODEL
    # clf = get_classifier()
    # clf_name = 'XGBoost'
    # clf.fit(features_train, y_train)

    # EVALUATE
    # plot_confusion_matrix()
    # plot_decission_surface(clf, features_train, y_train, feature_names)
    # plot_tree(clf, num_trees=tree_no, rankdir='LR')


if __name__ == '__main__':
    main()

    my_list = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]

    print(random.choice(my_list))
