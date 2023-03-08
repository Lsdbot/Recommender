from sqlite.get_data import load_data

if __name__ == "__main__":
    #    for i in range(12):
    load_data(
        "/home/sergey/projects/Credit_scoring/raw/train_data/train_data_1.pq",
        "train_data",
        if_exists="append"
    )
