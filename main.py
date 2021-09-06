import itertools
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import plot_tree

from FeatureElimination import RecursiveFeatureElimination
from SmoteUpsample import SmoteTest
from XGB import get_classifier
from textblob import TextBlob
import pyroot
# from root_pandas import read_root
from bootstrap import bootstrap

from convert_ntuple_df import root_to_df


def ExtractData(column_names, y_name):
    if not column_names or not y_name:
        raise ValueError("Feed me daddy!!!")

    pd.set_option("display.max_rows", 10)
    pd.set_option("display.width", 4000)
    pd.set_option("display.max_columns", 999)
    X = root_to_df('TVAERS_ntuple.root', column_names)

    # REMOVE DATA
    X = X.replace({4294967295: 0})
    column_names.remove('datedied')
    column_names.remove('vax_date')
    column_names.remove('symptoms')
    column_names.remove('vaers_id')

    # X['time_to_death']
    # num days -> time to onset
    # Add columns to X and column_names
    # Add float version of symptoms
    # Add regression score

    bool_chart = ['died', 'l_threat', 'er_visit', 'hospital', 'x_stay', 'disable', 'recovd']
    severity_scores = []
    # input(X['died'])
    # for chart in bool_chart:

    for d, r, l, er, h, x, dis in zip(X['died'], X['recovd'], X['l_threat'], X['er_visit'], X['hospital'], X['x_stay'], X['disable']):
        severity_scores.append(round((d + r + l + er + h + dis+3.5)/7))
    print(f"0:{severity_scores.count(0)}")
    print(f"1:{severity_scores.count(1)}")
    X[y_name] = severity_scores
    column_names.remove('died')
    column_names.remove('recovd')
    print("Severity", max(X['severity']), min(X['severity']))

    # EXTRACT LISTS
    # X = X[:500000]
    y = X[[y_name]]
    # column_names.remove(y_name)
    features = X[column_names]

    # symptoms = X['symptoms']
    # symptom_sentiment = [TextBlob(text).sentiment for text in symptoms ]

    # cv = CountVectorizer(stop_words='english')
    # data_cv = cv.fit_transform(X['symptoms'])
    # data_dtm = pd.DataFrame(data_cv.toarray(), columns=feature_names)

    # X = features
    return features, column_names, y, X


def PrepareData():
    column_names = ['datedied', 'vax_date', 'vaers_id', 'age_yrs', 'died', 'l_threat', 'er_visit', 'hospital', 'hospdays', 'x_stay', 'disable', 'recovd', 'numdays', 'symptoms']
    features, feature_names, y, df = ExtractData(column_names=column_names, y_name='severity')
    return features, y, feature_names, df


def clean_bool_int(in_bool, reverse=False):
    if in_bool == 4294967295:
        in_bool = 0
    if reverse:
        return int(not bool(in_bool))
    return int(bool(in_bool))


def plot_confusion_matrix(clf, features_test, y_test):
    # https://queirozf.com/entries/visualizing-machine-learning-models-examples-with-scikit-learn-and-matplotlib
    y_pred = clf.predict(features_test)
    print(y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    plt.clf()
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    fmt = 'd'

    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    class_names = ['severe', 'not severe']
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label', size=14)
    plt.xlabel('Predicted label', size=14)
    plt.show()

def plot_roc(clf, X_train, y_train, features_test, y_test):
    # https://queirozf.com/entries/visualizing-machine-learning-models-examples-with-scikit-learn-and-matplotlib
    param_grid = [
        {
            'pca__n_components': [2, 4, 8],
            'clf__n_estimators': [5, 20, 50, 100, 200],
            'clf__max_depth': [1, 2, 3, 4]
        }
    ]

    pipeline = Pipeline([
        ('pca', PCA()),
        ('clf', get_classifier())
    ])

    num_cols = 3
    num_rows = math.ceil(len(ParameterGrid(param_grid)) / num_cols)
    plt.clf()
    fig, axes = plt.subplots(num_rows, num_cols, sharey=True)
    fig.set_size_inches(num_cols * 5, num_rows * 5)

    for i, g in enumerate(ParameterGrid(param_grid)):
        pipeline.set_params(**g)
        pipeline.fit(X_train, y_train)

        y_preds = pipeline.predict_proba(features_test)

        # take the second column because the classifier outputs scores for
        # the 0 class as well
        preds = y_preds[:, 1]

        # fpr means false-positive-rate
        # tpr means true-positive-rate
        fpr, tpr, _ = metrics.roc_curve(y_test, preds)

        auc_score = metrics.auc(fpr, tpr)

        ax = axes[i // num_cols, i % num_cols]

        # don't print the whole name or it won't fit
        ax.set_title(str([r"{}:{}".format(k.split('__')[1:], v) for k, v in g.items()]), fontsize=9)
        ax.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
        ax.legend(loc='lower right')

        # it's helpful to add a diagonal to indicate where chance
        # scores lie (i.e. just flipping a coin)
        ax.plot([0, 1], [0, 1], 'r--')

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')

    plt.gcf().tight_layout()
    plt.show()


def main():
    print("*** MediML ***")
    # COLLECT DATA
    # PrepareData()
    features, y, feature_names, df = PrepareData()

    # SPLIT SAMPLES
    print(f"selected_names: {len(feature_names)}")
    print(f"features: {features.shape}")
    print(f"y: {y.shape}")
    print(f"df: {df.shape}")
    features_train, features_test, y_train, y_test, df_train, df_test = train_test_split(features, y, df, test_size=0.3,stratify=y, random_state=42)

    # BALANCE CLASSES
    # features_train, y_train = bootstrap(features_train, y_train)
    # features_train, y_train = SmoteTest(features_train, y_train)
    # print(f"After smote: {features_train.shape}")

    # REMOVE USELESS DATA
    # features_train, features_test, feature_names = RecursiveFeatureElimination(features_train, features_test, y_train, feature_names)
    features_train = pd.DataFrame(features_train, columns=feature_names)
    features_test = pd.DataFrame(features_test, columns=feature_names)
    print(f"Surviving Features: {feature_names}")

    # TRAIN ML MODEL
    clf = get_classifier()
    # clf = LinearRegression()
    # clf_name = 'XGBoost'
    clf.fit(features_train, y_train)

    # EVALUATE
    # plot_confusion_matrix(clf, features_test, y_test)
    plot_roc(clf, features_test, y_test, features_test, y_test) # plot_decission_surface(clf, features_train, y_train, feature_names)
    # plot_tree(clf, num_trees=0, rankdir='LR')


if __name__ == '__main__':
    main()

    # my_list = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    #
    # print(random.choice(my_list))
