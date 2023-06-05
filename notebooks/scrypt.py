import pandas as pd
import numpy as np
import os

from tqdm.notebook import tqdm
from lightgbm import LGBMClassifier
from eif import iForest


def out_of_core_preprocessing(raw_path: str, target_path: str, processed_path: str, n: int = 5, chunk=250000) -> None:
    raw_dir = os.listdir(raw_path)
    Y = pd.read_csv(target_path, index_col='id')

    for i in tqdm(range(len(raw_dir))):
        file = raw_path + f'train_data_{i}.pq'

        X = pd.read_parquet(file)

        X = X.drop("pre_loans_total_overdue", axis=1)
        X = X.set_index('id')

        df = X.groupby(level=0).last().join(Y[chunk*i:chunk*(i+1)])

        for j in range(2, n + 1):
            df = df.join(X.iloc[:, 1:].groupby(level=0).nth(-j),
                         rsuffix=f'_{j}', how='left')

        column_eif = iForest(df[df["flag"] == 0].values, ntrees=300,
                             sample_size=128, ExtensionLevel=1)

        S1 = column_eif.compute_paths(X_in=df[df["flag"] == 0].values)

        indexes_eif = np.argsort(S1)[-int(len(S1) * 0.05):]
        df = df.drop(df.index[indexes_eif])

        df = df.fillna(0)
        df.to_parquet(processed_path + f"train_data_{i}.pq")


def out_of_core_train(train_path: str, target_path: str, chunk=250000, **kwargs) -> LGBMClassifier:
    train_dir = os.listdir(train_path)

    model = LGBMClassifier(**kwargs['params'], class_weight='balanced', n_jobs=4)
    df = pd.read_parquet(train_path + 'train_data_0.pq')

    y_train = df['flag']
    x_train = df.drop('flag', axis=1)

    model.fit(x_train, y_train)

    for i in tqdm(range(1, len(train_dir)-1)):
        file = train_path + f'train_data_{i}.pq'

        df = pd.read_parquet(file)

        y_train = df['flag']
        x_train = df.drop('flag', axis=1)

        model.fit(x_train, y_train, init_model=model)

    return model
