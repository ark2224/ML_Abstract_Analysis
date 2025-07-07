from collections import Counter
import numpy as np
import pandas as pd
import string
from pathlib import Path
from typing import List, Dict
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

LEMMATIZER = WordNetLemmatizer()
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))
PUNCT_TRANSFORM = str.maketrans("", "", string.punctuation)

from sklearn.feature_extraction.text import TfidfVectorizer
VECTORIZER = TfidfVectorizer()

RAW_FILES = list(Path("data/raw").glob("*.csv"))

OUTPUT_DIR = Path("data/processed")


def load_data(files, chunk_size=10_000):
    """Generator over raw CSVs, yielding DataFrame chunks."""
    for file in files:
        for subset in pd.read_csv(file, chunksize=chunk_size):
            yield subset


def preprocess_data(df) -> pd.DataFrame:
    df.dropna(inplace=True)
    df = df[df["description"].notna()].reset_index(drop=True)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["completion_date"] = pd.to_datetime(df["completion_date"])
    # df["label_"] = df["label_"].apply(lambda x: x.upper() == "TRUE")
    return df


def clean_sentence(text):
    sentence_cnt = len(sent_tokenize(text))

    no_punct = text.lower().translate(PUNCT_TRANSFORM)

    tokens = [
        w for w in word_tokenize(no_punct)
        if w.isalpha() and w not in STOP_WORDS
    ]

    lemmas = [LEMMATIZER.lemmatize(t, pos=wordnet.VERB) for t in tokens]

    return "".join(lemmas), sentence_cnt, len(lemmas)


def process_phase(text):
    if not isinstance(text, str):
        return None
    if text.strip().upper() == "NA":
        return 0
    phases = [int(num) for num in re.findall(r'\d+', text)]
    return np.mean(phases) if phases else None


def count_top_feat_values(files, category, chunksize=10_000, n=20) -> Dict:
    ctr = Counter()
    for fn in files:
        for chunk in pd.read_csv(fn, usecols=[category], chunksize=chunksize):
            # Dropping NA but might be common enough to be need inclusion
            ctr.update(chunk[category].dropna())
    topN = {cat: idx for idx, (cat,_) in enumerate(ctr.most_common(n))}
    topN["Other"] = 20

    return topN

def encode_top_category(df,
                        topN,
                        category_name):
    df[category_name] = df[category_name].where(df[category_name].isin(topN), other="Other")
    new_category_feats = pd.get_dummies(df[category_name], prefix=f"{category_name}_")
    # Ensure all columns are in each chunk
    all_possible_cols = [f"{category_name}_{cat}" for cat in topN]
    new_category_feats = new_category_feats.reindex(all_possible_cols, fill_value=0)
    return pd.concat([df, new_category_feats], axis=1)


OUTCOME_KEYWORDS = ["efficacy", "placebo", "adverse", "safety", "tolerability"]

def count_outcome_keywords(df):
    pattern = r"\b(" + "|".join(map(re.escape, OUTCOME_KEYWORDS)) + r")\b"
    df["total_outcome_kw"] = df["description"].apply(
        lambda text: len(re.findall(pattern, text)) if isinstance(text, str) else 0
    ).values  # Bypass reindexing mismatch
    return df
    # df = df.copy().reset_index(drop=True)

    # # Normalize descriptions
    # desc_series = df["description"].apply(lambda desc: desc.lower() if isinstance(desc, str) else "")

    # # Regex pattern to count all keywords
    # pattern = r"\b(" + "|".join(map(re.escape, OUTCOME_KEYWORDS)) + r")\b"

    # # Apply counta nd assign by value (avoids reindexing errors)
    # df["total_outcome_kw"] = pd.Series(
    #     desc_series.apply(lambda text: len(re.findall(pattern, text))).values,
    #     index=df.index
    # )
    # return df

    # # method 1: pandas .str.count
    # # out_kw_count = {}
    # # for kw in OUTCOME_KEYWORDS:
    # #     pat = rf"\b{kw}\b"
    # #     desc_col = chunk["description"].apply(lambda x: x.lower() if isinstance(x, str) else "")
    # #     out_kw_count[f"count_{kw}"] = desc_col.str.count(pat)
    # #         #   (chunk["description"].fillna("")
    # #         #   .astype(str)
    # #         #   .str.lower()
    # #         #   .str.count(pat))
    # # kw_df = pd.DataFrame(out_kw_count)
    # # print(kw_df.size)
    # # print(chunk.size)
    # # # kw_df["total_outcome_kw"] = kw_df.select_dtypes(include="number").sum(axis=1)
    # # chunk["total_outcome_kw"] = kw_df.select_dtypes(include="number").sum(axis=1) # kw_df.sum(axis=1)#sum([out_kw_count[f"count_{kw}"] for kw in OUTCOME_KEYWORDS])
    
    # df = df.reset_index(drop=True)  # Fix for duplicate index

    # print("description col type:", type(df["description"]))
    # print("description col head:", df["description"].head())
    # print("description col length:", len(df["description"]))
    # print("description index duplicated?", df["description"].index.duplicated().any())
    # print("value types in description:", df["description"].apply(type).value_counts())

    # df["description"] = df["description"].apply(lambda desc: desc.lower()
    #                                             if isinstance(desc, str) else "")

    # pattern = r"\b(" + "|".join(map(re.escape, OUTCOME_KEYWORDS))  + r")\b"

    # # df["total_outcome_kw"] = df["description"].apply(lambda text:
    # #                                                  len(re.findall(pattern, text))
    # #                                                  if isinstance(text,str) else 0)
    # results = df["description"].apply(
    #     lambda text: len(re.findall(pattern, text)) if isinstance(text, str) else 0
    # )

    # print("Result type:", type(results))
    # print("Index match:", results.index.equals(df.index))
    # print("Index duplicated?", results.index.duplicated().any())
    # print("Length match:", len(results), len(df))

    # print("Sample result values:")
    # print(results.head(10).tolist())
    # print(results.apply(type).value_counts())
    # df["total_outcome_kw"] = results
    # return df


def extract_features(df,
                     top20conditions,
                     top20interventions) -> pd.DataFrame:
    # NLP Features
    # Line below might've been causing downstream error due to df indexing. Changing to drop empty clean_lemmas
    # df[["clean_lemmas", "sentence_count", "word_count"]] = df["description"].apply(lambda x: pd.Series(clean_sentence(x)))

    sentence_parts = df["description"].apply(lambda x: pd.Series(clean_sentence(x)))
    sentence_parts.columns = ["clean_lemmas", "sentence_count", "word_count"]
    df = pd.concat([df.reset_index(drop=True), sentence_parts.reset_index(drop=True)], axis=1)

    # Remove rows with empty lemmas before TF-IDF
    df = df[df["clean_lemmas"].str.strip().astype(bool)].reset_index(drop=True)

    # Apply TF-IDF
    tfidf_matrix = VECTORIZER.fit_transform(df["clean_lemmas"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=VECTORIZER.get_feature_names_out()
    )

    # Add TF-IDF to main DataFrame
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # Date-based feature
    df["duration"] = df["completion_date"] - df["start_date"]

    # Process phase
    df["phase"] = df["phase"].apply(lambda p: process_phase(p))

    # Encode categories
    df = encode_top_category(df, top20conditions, "condition_")
    df = encode_top_category(df, top20interventions, "intervention_type")

    # Keyword count
    # Ensure description column is clean before keyword search
    if isinstance(df["description"], pd.DataFrame):
        df["description"] = df["description"].iloc[:, 0]    
    df["description"] = df[["description"]].squeeze().astype(str).str.lower()
    # df["description"] = df["description"].astype(str).str.lower()

    df = count_outcome_keywords(df)

    return df

    # # OLD EXTRACT_FEATURES(DF)
    # df[["clean_lemmas", "sentence_count", "word_count"]] = df["description"] \
    #     .apply(lambda x: pd.Series(clean_sentence(x))) \
    #     .reset_index(drop=True)

    # df = df[df["clean_lemmas"].str.strip().astype(bool)]  # Drop empty lemmas
    # df = df.reset_index(drop=True)

    # tfidf_matrix = VECTORIZER.fit_transform(df["clean_lemmas"])
    # tfidf_df = pd.DataFrame(
    #     tfidf_matrix.toarray(),
    #     columns=VECTORIZER.get_feature_names_out(),
    # )
    # # Debug check before concatenation
    # print("TF-IDF rows:", tfidf_matrix.shape[0])
    # print("DF rows:", df.shape[0])
    
    # df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # # Metadata
    # df["duration"] = df["completion_date"] - df["start_date"]
    # df["phase"] = df["phase"].apply(lambda p: process_phase(p) if isinstance(p, str) else None)
    # df = df.reset_index(drop=True)

    # df = encode_top_category(df, top20conditions, "condition_")
    # df = encode_top_category(df, top20interventions, "intervention_type")

    # df = count_outcome_keywords(df)

    # return df


def process_and_save(input_files, output_dir):
    top20conditions = count_top_feat_values(files=input_files,
                                            category="condition_",
                                            n=20)
    # top20interventions = count_top_feat_values(files=input_files,
    #                                         category="intervention_type",
    #                                         n=20)
    top20interventions = {}

    for idx, raw_chunk in enumerate(load_data(input_files)):
        raw_chunk = raw_chunk.reset_index(drop=True)  # Ensure unique index at the chunk level

        df = preprocess_data(raw_chunk)

        df = extract_features(df, top20conditions, top20interventions)

        df.to_csv(f"{output_dir}/featureset_{idx}.csv.gz",
                  header=True,
                  index=False,
                  compression='gzip')
    

if __name__ == "__main__":
    process_and_save(RAW_FILES, OUTPUT_DIR)