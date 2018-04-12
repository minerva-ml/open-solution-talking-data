import math


def train_valid_split_on_timestamp(meta, validation_size, timestamp_column, shuffle=True):
    n_rows = len(meta)
    train_size = n_rows - math.floor(n_rows * (1 - validation_size))

    meta = meta.sort_values(timestamp_column)

    meta_train_split = meta.iloc[:train_size]
    meta_valid_split = meta.iloc[train_size:]

    if shuffle:
        meta_train_split = meta_train_split.sample(frac=1, random_state=1234)
        meta_valid_split = meta_valid_split.sample(frac=1, random_state=1234)

    return meta_train_split, meta_valid_split
