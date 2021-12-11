from numpy import array
from numpy import hstack

import datetime

import yfinance as yf


def normalize_data(dataset):
    cols = dataset.columns.tolist()
    col_name = [0] * len(cols)
    for i in range(len(cols)):
        col_name[i] = i
    dataset.columns = col_name
    dtypes = dataset.dtypes.tolist()
    #         orig_answers = dataset[attr_row_predict].values
    minmax = list()
    for column in dataset:
        dataset = dataset.astype({column: 'float32'})
    for i in range(len(cols)):
        col_values = dataset[col_name[i]]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for column in dataset:
        values = dataset[column].values
        for i in range(len(values)):
            values[i] = (values[i] - minmax[column][0]) / (minmax[column][1] - minmax[column][0])
        dataset[column] = values
    dataset[column] = values
    return dataset, minmax


def data_setup(symbol, data_len, seq_len):
    end = datetime.datetime.today().strftime('%Y-%m-%d')
    start = datetime.datetime.strptime(end, '%Y-%m-%d') - datetime.timedelta(days=(data_len / 0.463))
    orig_dataset = yf.download(symbol, start, end)
    close = orig_dataset['Close'].values
    open_ = orig_dataset['Open'].values
    high = orig_dataset['High'].values
    low = orig_dataset['Low'].values
    dataset, minmax = normalize_data(orig_dataset)
    cols = dataset.columns.tolist()

    data_seq = list()
    for i in range(len(cols)):
        if cols[i] < 4:
            data_seq.append(dataset[cols[i]].values)
            data_seq[i] = data_seq[i].reshape((len(data_seq[i]), 1))
    data = hstack(data_seq)
    n_steps = seq_len
    X, y = split_sequences(data, n_steps)
    n_features = X.shape[2]
    n_seq = len(X)
    n_steps = seq_len
    print(X.shape)
    X = X.reshape((n_seq, 1, n_steps, n_features))
    true_y = []
    for i in range(len(y)):
        true_y.append([y[i][0], y[i][1]])
    return X, array(true_y), n_features, minmax, n_steps, close, open_, high, low


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


if __name__ == "__main__":
    data_setup("AAPL", 1000, 1000)
