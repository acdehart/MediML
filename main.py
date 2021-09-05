import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split

from FeatureElimination import RecursiveFeatureElimination
from SmoteUpsample import SmoteTest
from XGB import get_classifier


def PrepareData():
    # Load CSV to Pandas DF

    column_names = ['died', 'recovered', 'hospitalized', 'ext_stay', 'disabled', 'l_threat', 'er_visit', 'hospital_days', 'days_to_death', 'days_to_onset']


    # text = "Your mother was a hampster, and your Father smelt of Elder Berries!!!"
    # print(clean_string(text))


def clean_string(text):
    # Part 1
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('!', '', text)
    text = re.sub(',', '', text)


    # Part 2
    text = re.sub('[`".]', '', text)
    text = re.sub('\n\t', '', text)

    return text


    # All text lower,
    # remove punctuation
    # remove numerical values
    # remove non speech (eg '\n')
    # tokenize text
    # remove stop words

    # stemming/ lemmazation
    # parts of speech tagging
    pass


def main():

    # COLLECT DATA
    # PrepareData()
    features, y, feature_names, df = PrepareData()

    # SPLIT SAMPLES
    # features_train, features_test, y_train, y_test, df_train, df_test = train_test_split(features, y, df, test_size=0.3, stratify=y, random_state=42)

    # BALANCE CLASSES
    # features_train, y_train = SmoteTest(features_train, y_train)
    # features_train, y_train = BootStrap(features_train, y_train)

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

