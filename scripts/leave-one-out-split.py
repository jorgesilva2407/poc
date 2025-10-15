import pandas as pd


def main() -> None:
    df = pd.read_csv("./data/processed/merged_ratings.csv")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_list, val_list, test_list = [], [], []

    for _, group in df.groupby("user_id", sort=False):
        n = len(group)
        if n < 3:
            train_list.append(group)
        else:
            train_list.append(group.iloc[:-2])
            val_list.append(group.iloc[[-2]])
            test_list.append(group.iloc[[-1]])

    train = pd.concat(train_list, ignore_index=True)
    val = pd.concat(val_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    train.drop(columns=["timestamp"], inplace=True)
    val.drop(columns=["timestamp"], inplace=True)
    test.drop(columns=["timestamp"], inplace=True)

    train.to_csv("./data/processed/train.csv", index=False)
    val.to_csv("./data/processed/val.csv", index=False)
    test.to_csv("./data/processed/test.csv", index=False)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


if __name__ == "__main__":
    main()
