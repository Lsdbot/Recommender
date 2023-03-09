import pandas as pd
from sqlalchemy import create_engine


def get_data(filepath) -> pd.DataFrame:
    return pd.read_parquet(filepath, engine='pyarrow')


def transform_data(df: pd.DataFrame, n_columns) -> pd.DataFrame:
    
    return


def load_data(df: pd.DataFrame, table_name, if_exists="replace", chunksize=100000) -> None:
    conn = create_engine("sqlite:///sqlite/clients.db")
    df.to_sql(name=table_name, con=conn, if_exists=if_exists, chunksize=chunksize)


def read_data(sql_path, table_name):
    conn = create_engine(sql_path)
    return pd.read_sql_table(table_name, conn.connect())
