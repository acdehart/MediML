import pandas as pd
from sklearn.model_selection import train_test_split

from Evaluation import plot_confusion_matrix, plot_decision_surface
from PrepareData import PrepareData
from XGB import get_classifier
# from root_pandas import read_root
from bootstrap import BootstrapUpsample


def main():
    print("*** MediML ***")
    # COLLECT DATA
    # PrepareData()
    features, y, feature_names, df = PrepareData()

    # SPLIT SAMPLES
    print(f"{features.head(1)}")
    features_train, features_test, y_train, y_test, df_train, df_test = train_test_split(features, y, df, test_size=0.3,stratify=y, random_state=42)


    # BALANCE CLASSES
    features_train, y_train = BootstrapUpsample(features_train, y_train)
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
    # plot_roc(clf, features_test, y_test, features_test, y_test) # plot_decission_surface(clf, features_train, y_train, feature_names)
    # plot_tree(clf, num_trees=0, rankdir='LR')
    plot_decision_surface(clf, features_train, y_train, feature_names)

if __name__ == '__main__':
    main()

    # my_list = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    #
    # print(random.choice(my_list))
