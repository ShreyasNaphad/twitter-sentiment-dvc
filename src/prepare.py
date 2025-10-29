import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

if __name__ == "__main__":
    df = pd.read_csv("data/raw/twitter_sentiment.csv")
    df['tweet'] = df['tweet'].apply(clean_text)
    df = df.dropna()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("✅ Data cleaned and split into train/test saved in data/processed/")
