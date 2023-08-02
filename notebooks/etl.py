import pandas as pd

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import os
import yaml
from typing import Literal

from math import ceil


TRAIN_PATH = "../data/raw/train_data/"
TEST_PATH = "../data/raw/test_data/"

PROCESSED_TRAIN_PATH = "../data/transformed/train_data/"
PROCESSED_TEST_PATH = "../data/transformed/test_data/"
COLUMNS_PATH = "../data/transformed/columns.yaml"


def get_columns(df, ohe_columns) -> set:
    encoded_columns = set()

    for col in ohe_columns:
        pivot_table = (
            df.groupBy('id')
            .pivot(col)
            .agg(F.count(col))
            .fillna(0)
        )

        pivot_columns = pivot_table.columns
        pivot_columns.remove("id")

        pivot_columns = list(map(lambda x: col + '_' + x, pivot_columns))

        encoded_columns = encoded_columns.union(set(pivot_columns))

    return encoded_columns


def get_columns_from_files(directory_path: str, target_path: str) -> None:
    spark = (
        SparkSession.builder.master("local[*]").getOrCreate()
    )

    all_columns = {'id', 'rn'}

    for file in os.listdir(directory_path):
        df = (
            spark
            .read
            .format("parquet")
            .option("header", "true")
            .load(TRAIN_PATH + file)
        )

        ohe_columns = df.columns
        ohe_columns.remove("id")
        ohe_columns.remove("rn")

        all_columns = all_columns.union(get_columns(df, ohe_columns))

    with open(target_path, 'w') as file:
        yaml.dump(list(all_columns), file)

    spark.stop()


def transform_data(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    data = data.astype("category")
    data["id"] = data.id.astype(int)
    data["rn"] = data.rn.astype(int)

    encoded_data = pd.get_dummies(data, drop_first=True)

    empty_columns = [x for x in columns if x not in encoded_data.columns]

    for col in empty_columns:
        encoded_data[col] = 0

    dummy_columns = columns.copy()

    dummy_columns.remove("id")
    dummy_columns.remove("rn")

    aggregated_data = encoded_data.groupby("id")[dummy_columns].sum()
    aggregated_data['rn'] = encoded_data.groupby("id")['rn'].last()

    return aggregated_data.reset_index(drop=False)


def pipeline_transform(directory_path: str, chunk_size: int,
                       columns_path: str, transform_type: Literal["train", "test"],
                       n_users: int = 250000) -> None:
    with open(columns_path, 'r') as file:
        columns = yaml.load(file, yaml.FullLoader)

    for j in range(len(os.listdir(directory_path))):
        if transform_type == "train":
            df = pd.read_parquet(directory_path + f"train_data_{j}.pq")
        elif transform_type == "test":
            df = pd.read_parquet(directory_path + f"test_data_{j}.pq")
            df.id = df.id - 3000000
        else:
            raise Exception("Wrong type of transforming the data")

        for i in range(ceil(n_users/chunk_size)):
            data = df[(df.id >= 2 * j * chunk_size + chunk_size * i)
                      & (df.id < 2 * j * chunk_size + chunk_size * (i + 1))]

            df_processed = transform_data(data, columns)

            if transform_type == "train":
                df_processed.to_parquet(PROCESSED_TRAIN_PATH + f"train_data_{j}.pq" + f'.{i}')
            elif transform_type == "test":
                df_processed.to_parquet(PROCESSED_TEST_PATH + f"test_data_{j}.pq" + f'.{i}')
            else:
                raise Exception("Wrong type of transforming the data")


# get_columns_from_files(TRAIN_PATH, COLUMNS_PATH)
# pipeline_transform(TRAIN_PATH, 125000, COLUMNS_PATH, "train")
# pipeline_transform(TEST_PATH, 125000, COLUMNS_PATH, "test")
