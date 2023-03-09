import pandas as pd
from sqlalchemy import create_engine


def load_data(filepath, table_name, if_exists="replace", chunksize=100000):
    db = pd.read_parquet(filepath, engine='pyarrow')
    conn = create_engine("sqlite:///sqlite/clients.db")
    db.to_sql(name=table_name, con=conn, if_exists=if_exists, chunksize=chunksize)

def read_data(sql_path, table_name):
    conn = create_engine(sql_path)
    return pd.read_sql_table(table_name, conn.connect())