import pandas as pd
import numpy as np
import os

from lightgbm import LGBMClassifier


def pipeline_preprocessing(df: pd.DataFrame, n: int = 4) -> np.ndarray:
    df = df.drop("pre_loans_total_overdue", axis=1)
    df = df.set_index('id')

    X = df.groupby(level=0).last()

    for i in range(2, n + 1):
        X = X.join(df.iloc[:, 1:].groupby(level=0).nth(-i),
                   rsuffix=f'_{i}', how='left')

    X = X.fillna(0)

    return X


def out_of_core_train(train_path: str, target_path: str, chunk=250000, **kwargs) -> LGBMClassifier:
    train_dir = os.listdir(train_path)
    Y = pd.read_csv(target_path, index_col='id')

    model = LGBMClassifier(**kwargs['params'])
    df = pd.read_parquet(train_path + 'train_data_0.pq')

    y_train = Y[0:250000]
    x_train = pipeline_preprocessing(df)

    model.fit(x_train, y_train)

    for i in range(1, len(train_dir)):
        file = train_path + f'train_data_{i}.pq'

        df = pd.read_parquet(file)

        y_train = Y[i * chunk:(i + 1) * chunk]
        x_train = pipeline_preprocessing(df)

        model.fit(x_train, y_train, init_model=model)

    return model
