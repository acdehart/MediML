import numpy as np
import pandas as pd


def BootstrapUpsample(features_train, y_train):
    classes = y_train[y_train.columns[0]].unique()
    features_train['y'] = y_train

    dfs = [features_train[features_train['y'] == c] for c in classes]

    n_rows = [len(df) for df in dfs]
    max_rows = max(n_rows)

    out_df = pd.DataFrame({ft: pd.Series(dtype='float') for ft in features_train})
    for df, row_count, idx in zip(dfs, n_rows, range(len(dfs))):
        diff = max_rows-row_count
        if diff != 0:
            df = bootstrap(df, diff)
        out_df = out_df.append(df)

    y_train = out_df['y']

    features_train = out_df.loc[:, out_df.columns != 'y']

    return features_train, y_train


def bootstrap(input_data, sample_diff, n_bootstrap=2):
    # input_data -> dataframe of rows for one class
    # sample_diff -> number of rows to be created
    # n_bootstrap -> number of points to average between
    cols = input_data.columns
    n_samples = input_data.shape[0]
    if n_samples == 0:
        raise ValueError("empty input data array")
    n_dimensions = input_data.shape[1]
    if n_dimensions == 0:
        raise ValueError("input data has no dimensions")
    input_data = input_data.to_numpy()
    output_data = np.zeros((n_samples + sample_diff, n_dimensions))
    subset_data = np.zeros((n_dimensions, n_bootstrap))
    for i in range(n_samples + sample_diff):
        if i < n_samples:
            output_data[i] = input_data[i]
            continue
        indeces = np.random.randint(0, n_samples,n_bootstrap)
        for j in range(n_bootstrap):
            for k in range(n_dimensions):
                subset_data[k][j] = input_data[ indeces[j] ][k]
        for j in range(n_dimensions-1):
            output_data[i][j] = np.mean(subset_data[j])*np.random.normal(1, n_bootstrap/np.sqrt(n_samples))

    output_data = pd.DataFrame(data=output_data, columns=cols)

    return output_data
