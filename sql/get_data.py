import pandas as pd
from sqlalchemy import create_engine


def load_data(filepath, table_name):
    db = pd.read_csv(filepath)
    db = db.T
    conn = create_engine("sqlite:///recommender.db")
    db.to_sql(name=table_name, con=conn, if_exists="replace")

def read_data(sql_path, table_name):
    conn = create_engine(sql_path)
    return pd.read_sql_table(table_name, conn.connect())