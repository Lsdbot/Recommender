import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def get_data(filepath) -> pd.DataFrame:
    return pd.read_parquet(filepath, engine='pyarrow')


def make_client_history(clients, num_rn, num_columns):
    row = np.zeros((num_rn, num_columns), dtype=np.int8)
    length = len(clients)

    for i in range(0, min(num_rn, length)):
        row[num_rn - 1 - i] = clients[length - 1 - i][1:]

    return row


def transform_dataframe(df):
    d = dict()
    values = df.values

    for value in values:
        try:
            d[value[0]].append(value[1:].astype(np.int8))
        except:
            d[value[0]] = list()
            d[value[0]].append(value[1:].astype(np.int8))

    return d


def transfrom_data(df, num_rn, num_columns):
    clients = transform_dataframe(df)
    values = list()

    for i in range(len(clients)):
        values.append(make_client_history(clients[i], num_rn, num_columns))

    return values


def load_data(df: pd.DataFrame, table_name, if_exists="replace", chunksize=100000) -> None:
    conn = create_engine("sqlite:///sqlite/clients.db")
    df.to_sql(name=table_name, con=conn, if_exists=if_exists, chunksize=chunksize)


def read_data(sql_path, table_name):
    conn = create_engine(sql_path)
    return pd.read_sql_table(table_name, conn.connect())
