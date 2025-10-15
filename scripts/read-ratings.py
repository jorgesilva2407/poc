import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_ratings(ratings_file):
    df = pd.read_json(ratings_file, lines=True)
    df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df['label'] = (df['rating'] >= 4).astype(int)
    df.drop(columns=['rating'], inplace=True)
    return df

def merge_ratings(beauty_ratings, clothing_ratings):
    merged_ratings = pd.concat([beauty_ratings, clothing_ratings], ignore_index=True)
    return merged_ratings

def index_column(df, column_name):
    encoder = LabelEncoder()
    df[column_name] = encoder.fit_transform(df[column_name])
    return df, encoder

def save_ratings(ratings, output_file):
    ratings.to_csv(output_file, index=False)

def save_encoder(encoder, output_file):
    with open(output_file, 'wb') as f:
        pkl.dump(encoder, f)

def main() -> None:
    beauty_ratings = read_ratings('./data/raw/reviews_Beauty_5.jsonl')
    clothing_ratings = read_ratings('./data/raw/reviews_Clothing_Shoes_and_Jewelry_5.jsonl')
    merged_ratings = merge_ratings(beauty_ratings, clothing_ratings)
    merged_ratings, user_encoder = index_column(merged_ratings, 'user_id')
    merged_ratings, item_encoder = index_column(merged_ratings, 'item_id')
    save_ratings(merged_ratings, './data/processed/merged_ratings.csv')
    save_encoder(user_encoder, './data/processed/user_encoder.pkl')
    save_encoder(item_encoder, './data/processed/item_encoder.pkl')

if __name__ == '__main__':
    main()
