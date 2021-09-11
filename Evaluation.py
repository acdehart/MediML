import itertools
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from xgboost import plot_tree

from XGB import get_classifier


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
    class_names = get_class_names()
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


def get_class_names():
    return ['not severe', 'severe']


def check_index(X, y):
    X_index = set(X.index.tolist())
    y_index = set(y.index.tolist())
    if X_index != y_index:
        raise ValueError(f"X&y don't share the same set of indices\nX:{X_index}\ny:{y_index}")


def plot_decision_surface(clf, features_train, y_train, feature_names):
    check_index(features_train, y_train)
    plt.show()
    plt.close()
    plt.cla()
    # print(feature_names)
    # print(features_train)
    # print(y_train)
    # https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py
    pair_names = ['age_yrs', 'disable']
    # pair_names = ['r2048.0', 'width', 'r_average', 'r0.09']
    pair_names_reversed = pair_names[::-1]
    pairs = []
    for ft_row in pair_names:
        for ft_col in pair_names_reversed:
            if ft_col == ft_row:
                ft_col = 'l_threat'
            pairs.append([feature_names.index(ft_col), feature_names.index(ft_row)])

    # OCV, f0, r_average, slope -> Rotate through

    # print(pairs)
    # print(features_train)

    plot_colors = "rg"
    plot_step = 0.02

    class_names = get_class_names()
    n_classes = len(class_names)
    plot_numbers = range(n_classes)

    for pair_index, pair in enumerate(pairs):
        print(".", end="")
        # Take only 2 isolated features
        X = features_train.loc[:, [feature_names[pair[0]], feature_names[pair[1]]]]
        check_index(X, y_train)

        clf.fit(X, y_train)

        plt.subplot(len(pair_names), len(pair_names), pair_index + 1)

        x_min, x_max = X.loc[:, feature_names[pair[0]]].min() - 1, X.loc[:, feature_names[pair[0]]].max() + 1
        y_min, y_max = X.loc[:, feature_names[pair[1]]].min() - 1, X.loc[:, feature_names[pair[1]]].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict((np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)

        plt.xlabel(feature_names[pair[0]])
        plt.ylabel(feature_names[pair[1]])

        check_index(X, y_train)

        for i, color in zip(plot_numbers, plot_colors):
            # print(f"HEY! {i}, {color}, {class_names[i]}, {feature_names[pair[0]]}, {feature_names[pair[1]]}")
            # print(X)
            # print('.', end='')
            print(y_train)
            idx = np.where(y_train == i)
            # input(idx)
            for index in idx[0]:

                # input(index)
                if index not in X.index.tolist():
                    raise ValueError(f"Y index {index} does not pair with any row in X {X.index.tolist()}")
            # print(X.loc[idx, feature_names[pair[0]]])
            # print(idx)
            # X_temp = X.loc[idx, [feature_names[pair[0]]]]
            # input(X_temp)
            plt.scatter(X.loc[idx, feature_names[pair[0]]], X.loc[idx, feature_names[pair[1]]], c=color,
                        label=class_names[i],
                        cmap=plt.cm.RdYlGn, edgecolor='black', s=15)

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")

    plt.figure()
    clf = clf.fit(features_train, y_train)
    plot_tree(clf, num_trees=0, filled=True)
    plt.show()