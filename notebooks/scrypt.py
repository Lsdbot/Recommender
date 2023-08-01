import pandas as pd
import os

from tqdm.notebook import tqdm
from lightgbm import LGBMClassifier


def out_of_core_train(train_path: str, target_path: str, chunk: int = 250000, **kwargs) -> LGBMClassifier:
    train_dir = os.listdir(train_path)
    y_train = pd.read_csv(target_path, nrows=chunk)

    model = LGBMClassifier(**kwargs['params'], class_weight='balanced', n_jobs=4)

    X_0 = pd.read_parquet(train_path + 'train_data_0.pq.0')
    X_1 = pd.read_parquet(train_path + 'train_data_0.pq.1')

    X = pd.concat([X_0, X_1])
    Y = y_train[:chunk]

    model.fit(X, Y)

    for i in tqdm(range(1, len(train_dir) // 2)):
        X_0 = pd.read_parquet(train_path + 'train_data_0.pq.0')
        X_1 = pd.read_parquet(train_path + 'train_data_0.pq.1')

        X = pd.concat([X_0, X_1])
        Y = y_train[i * chunk:2 * i * chunk]

        model.fit(X, Y, init_model=model)

    return model
