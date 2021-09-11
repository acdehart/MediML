import pandas as pd

from convert_ntuple_df import root_to_df


def ExtractData(column_names, y_name):
    # if not column_names:
    #     raise ValueError("Feed me daddy!!!")

    pd.set_option("display.max_rows", 10000)
    pd.set_option("display.width", 4000)
    pd.set_option("display.max_columns", 999)
    root_path = 'TVAERS_ntuple_v2.root'
    column_names = []
    X = root_to_df(root_path, column_names)

    # REMOVE DATA
    # X = X.replace({4294967295: 0})
    # column_names.remove('datedied')
    # column_names.remove('vax_date')
    # column_names.remove('symptoms')
    # column_names.remove('vaers_id')
    # input(X)

    # X['time_to_death']
    # num days -> time to onset
    # Add columns to X and column_names
    # Add float version of symptoms
    # Add regression score

    # bool_chart = ['died', 'l_threat', 'er_visit', 'hospital', 'x_stay', 'disable', 'recovd']
    severity_scores = []
    # input(X['died'])
    # for chart in bool_chart:

    for d, r, l, er, h, x, dis in zip(X['died'], X['recovd'], X['l_threat'], X['er_visit'], X['hospital'], X['x_stay'], X['disable']):
        severity_scores.append(round((d + r + l + er + h + dis+3.5)/7))
    print(f"0:{severity_scores.count(0)}")
    print(f"1:{severity_scores.count(1)}")
    X[y_name] = severity_scores
    # column_names.remove('died')
    # column_names.remove('recovd')
    print("Severity", max(X['severity']), sum(X['severity'])/len(X['severity']), min(X['severity']))

    # EXTRACT LISTS
    X = X[:1000]
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