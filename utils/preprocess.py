import re
import pandas as pd
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", "", text)
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=["post", "hashtag", "label"], inplace=True)
    df["text"] = df["post"] + " " + df["hashtag"]
    df["text"] = df["text"].apply(clean_text)
    return df["text"].tolist(), df["label"].astype(int).tolist()
