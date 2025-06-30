from collections import Counter
import numpy as np
import pandas as pd
import string
from pathlib import Path
from typing import List, Dict

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

LEMMATIZER = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))
PUNCT_TRANSFORM = str.maketrans("", "", string.punctuation)

from sklearn.feature_extraction.text import TfidfVectorizer
VECTORIZER = TfidfVectorizer()

RAW_FILES = list(Path("data/raw/").glob("*.csv"))

OUTPUT_DIR = Path("data/processed")


def load_data(files: list, chunk_size: int = 10_000):
    """Generator over raw CSVs, yielding DataFrame chunks."""
    for file in files:
        for subset in pd.read_csv(file, nrows=chunk_size):
            yield subset


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["completion_date"] = pd.to_datetime(df["completion_date"])
    df["label_"] = df["label_"].apply(lambda x: x.upper() == "TRUE")
    
    return df


def clean_sentence(text: str):
    sentence_cnt = len(sent_tokenize(text))

    no_punct = text.lower().translate(PUNCT_TRANSFORM)

    tokens = [
        w for w in word_tokenize(no_punct)
        if w.isalpha() and w not in STOP_WORDS
    ]

    lemmas = [LEMMATIZER.lemmatize(t, pos=wordnet.VERB) for t in tokens]

    return "".join(lemmas), sentence_cnt, len(lemmas)


def process_phase(text: str):
    if text.upper() == "NA":
        return 0
    phases = text.upper().split("PHASE").dtype(int)
    return np.mean(phases)


def count_top_feat_values(files: List, category: str, n: int) -> Dict:
    count = Counter
    for file in files:
        for chunk in load_data(file):
            # Dropping NA but might be common enough to be need inclusion
            count.update(chunk[category].dropna())
    # topN = [cnt for _,cnt in count.most_common(n)] + ["Other"]
    topN = {cat: idx for idx, (cat,_) in enumerate(count.most_common(n))}
    topN["Other"] = 20

    return topN

def encode_top_category(df: pd.DataFrame,
                        topN: Dict,
                        cat: str):
    df["condition_"] = df["condition_"].where(df["condition_"].isin(topN), other="Other")
    new_category_feats = pd.get_dummies(df["condition_"], prefix="condition_")
    # Ensure all columns are in each chunk
    all_possible_cols = [f"condition_{cat}" for cat in topN]
    new_category_feats = new_category_feats.reindex(all_possible_cols, fill_value=0)
    return pd.concat([df, new_category_feats], axis=1)


OUTCOME_KEYWORDS = ["efficacy", "placebo", "adverse", "safety", "tolerability"]

def count_outcome_keywords(chunk):
    # method 1: pandas .str.count
    for kw in OUTCOME_KEYWORDS:
        pat = rf"\b{kw}\b"
        chunk[f"count_{kw}"] = (
            chunk["description"]
              .fillna("")
              .str.lower()
              .str.count(pat)
        )
    chunk["total_outcome_kw"] = chunk[[f"count_{kw}" for kw in OUTCOME_KEYWORDS]].sum(axis=1)
    return chunk


def extract_features(df: pd.DataFrame,
                     top20conditions: Dict,
                     top20interventions: Dict) -> pd.DataFrame:
    # NLP Features
    df[["clean_lemmas", "sentence_count", "word_count"]] = {
        df["description"].apply(lambda x: pd.Series(clean_sentence(x)))
    }
    tfidf_matrix = VECTORIZER.fit_transform(df["clean_lemmas"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.to_array(),
        columns=VECTORIZER.get_feature_names_out(),
        index=df.index
    )
    df = df.concat([df, tfidf_df], axis=1)

    # Metadata
    df["duration"] = df["completion_date"] - df["start_date"]
    df["phase"] = df["phase"].apply(lambda p: process_phase(p))
    df = encode_top_category(df, top20conditions, "conditions_")
    df = encode_top_category(df, top20interventions, "intervention_type")

    df = count_outcome_keywords(df)

    return df


def process_and_save(input_files: list, output_dir: Path):
    top20conditions = count_top_feat_values(files=input_files,
                                            category="condition_",
                                            n=20)
    top20interventions = count_top_feat_values(files=input_files,
                                            category="intervention_type",
                                            n=20)
    for idx, raw_chunk in enumerate(load_data(input_files)):
        df = preprocess_data(raw_chunk)
        df = extract_features(df, top20conditions, top20interventions)

        df.to_csv(f"{output_dir}/featureset_{idx}.csv.gz",
                  header=True,
                  index=False,
                  compression='gzip')
    

if __name__ == "__main__":
    process_and_save(RAW_FILES, OUTPUT_DIR)