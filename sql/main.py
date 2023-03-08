import sqlite

if __name__ == "__main__":
    df = sqlite.read_sql_table("sqlite:///recommender.db", "train_data")
