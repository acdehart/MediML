def bootstrap(input_data,sample_diff,n_bootstrap):
    n_samples = len(input_data)
    if n_samples == 0:
        raise ValueError("empty input data array")
    n_dimensions = len(input_data[0])
    if n_dimensions == 0:
        raise ValueError("input data has no dimensions")
    output_data = np.zeros((n_samples + sample_diff, n_dimensions))
    subset_data = np.zeros((n_dimensions, n_bootstrap))
    for i in range(n_samples + sample_diff):
        if i < n_samples:
            output_data[i] = input_data[i]
            continue
        indeces = np.random.randint(0,n_samples,n_bootstrap)
        for j in range(n_bootstrap):
            for k in range(n_dimensions):
                subset_data[k][j] = input_data[ indeces[j] ][k]
        for j in range(n_dimensions):
            output_data[i][j] = np.mean(subset_data[j])*np.random.normal(1,n_bootstrap/np.sqrt(n_samples))

    return output_data
