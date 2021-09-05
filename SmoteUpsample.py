from imblearn.over_sampling import SMOTE


def SmoteTest(X, y):
    sm = SMOTE(sampling_strategy='all', random_state=0)
    X_new, y_new = sm.fit_resample(X, y)
    return X_new, y_new