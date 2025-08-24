import io
import zipfile
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# ---------------------- Streamlit Page ----------------------
st.set_page_config(page_title="Sentilytics — Amazon Review Sentiment", layout="wide")
st.title("🛒 Sentilytics — Amazon Review Sentiment Explorer")
st.caption("Compare VADER (lexicon) vs RoBERTa (transformer) on Amazon reviews")

# ---------------------- Helper Functions ----------------------
@st.cache_data(show_spinner=False)
def read_csv_or_zip(file_or_path: Union[str, io.BytesIO, "UploadedFile"]) -> pd.DataFrame:
    if isinstance(file_or_path, str):
        if file_or_path.lower().endswith(".zip"):
            with zipfile.ZipFile(file_or_path) as zf:
                first = zf.namelist()[0]
                with zf.open(first) as f:
                    return pd.read_csv(f)
        return pd.read_csv(file_or_path)
    name = getattr(file_or_path, "name", "uploaded")
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(file_or_path) as zf:
            first = zf.namelist()[0]
            with zf.open(first) as f:
                return pd.read_csv(f)
    return pd.read_csv(file_or_path)

@st.cache_resource(show_spinner=False)
def load_vader() -> SentimentIntensityAnalyzer:
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_roberta(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def ensure_score_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "Score" in df.columns:
        df = df.copy()
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        df = df.dropna(subset=["Score"])
        df["Score"] = df["Score"].astype(int)
    return df

def add_vader_scores(df: pd.DataFrame, text_col: str = "Text") -> pd.DataFrame:
    sia = load_vader()
    scores = df[text_col].astype(str).apply(sia.polarity_scores)
    vader_df = pd.DataFrame(scores.tolist()).add_prefix("vader_")
    return pd.concat([df.reset_index(drop=True), vader_df.reset_index(drop=True)], axis=1)

def roberta_scores_batch(texts: List[str], tokenizer, model, max_length=256) -> pd.DataFrame:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        out = model(**enc)
    probs = softmax(out.logits.detach().cpu().numpy(), axis=1)
    return pd.DataFrame(probs, columns=["roberta_neg", "roberta_neu", "roberta_pos"])

def add_roberta_scores(df: pd.DataFrame, text_col: str = "Text", limit: Optional[int] = None, batch_size: int = 32) -> pd.DataFrame:
    tokenizer, model = load_roberta()
    texts = df[text_col].astype(str).tolist()
    if limit is not None:
        texts = texts[:limit]

    out_frames = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        out_frames.append(roberta_scores_batch(batch, tokenizer, model))
    scores_df = pd.concat(out_frames, ignore_index=True)

    base = df.iloc[:len(scores_df)].reset_index(drop=True)
    return pd.concat([base, scores_df], axis=1)

def clean_text(text):
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word.lower() not in stop_words]
    return " ".join(words)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# ---------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.header("⚙️ Options")
    data_source = st.selectbox("Select Data Source", ["Local CSV/ZIP", "Real Amazon Reviews (Hugging Face)"])
    uploaded = None
    default_path = None

    if data_source == "Local CSV/ZIP":
        uploaded = st.file_uploader("Upload CSV or ZIP containing a CSV", type=["csv", "zip"])
        if uploaded is None:
            st.write("Or place a CSV/ZIP in this folder and enter the name below.")
            default_path = st.text_input("Local file path (optional)", value="NewReviews.csv")
    else:
        st.info("Using Hugging Face Amazon Polarity dataset (binary stars).")

    sample_n = st.slider("Rows to analyze", min_value=100, max_value=5000, value=500, step=100)
    run_roberta = st.checkbox("Run RoBERTa on dataset (slower)", value=False)
    roberta_limit = st.number_input("RoBERTa rows (cap for speed)", min_value=50, max_value=2000, value=500, step=50)
    text_col = st.text_input("Text column name", value="Text")
    id_col = st.text_input("ID column name (optional)", value="Id")

# ---------------------- Load Data ----------------------
try:
    if data_source == "Local CSV/ZIP":
        if uploaded is not None:
            df = read_csv_or_zip(uploaded)
        elif default_path:
            df = read_csv_or_zip(default_path)
        else:
            st.stop()
        if text_col not in df.columns:
            st.error(f"Column `{text_col}` not found. Available columns: {list(df.columns)}")
            st.stop()
        df = df.head(sample_n).copy()
        df = ensure_score_numeric(df)

    else:
        dataset = load_dataset("amazon_polarity", split=f"train[:{sample_n}]")
        df = pd.DataFrame({
            "Id": range(len(dataset)),
            "Score": dataset["label"],
            "Text": dataset["content"]
        })
        df["Score"] = df["Score"].map({0:1, 1:5})  # Binary stars
        df["CleanText"] = df["Text"].apply(clean_text)
        df["Sentiment"] = df["CleanText"].apply(get_sentiment)

except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ---------------------- Compute VADER/RoBERTa ----------------------
if data_source == "Local CSV/ZIP":
    with st.spinner("Computing VADER sentiment..."):
        df_v = add_vader_scores(df, text_col=text_col)

    if run_roberta:
        with st.spinner("Running RoBERTa (transformer) on dataset..."):
            df_r = add_roberta_scores(df, text_col=text_col, limit=int(roberta_limit))
    else:
        df_r = None

# ---------------------- Overview ----------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Review Score Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Score", hue="Score", legend=False, palette="viridis")
st.pyplot(fig)

if data_source == "Real Amazon Reviews (Hugging Face)":
    st.subheader("Sentiment Polarity Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Sentiment"], bins=20, kde=True, ax=ax, color="blue")
    st.pyplot(fig)

    st.subheader("Word Cloud of Reviews")
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(df["CleanText"]))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# ---------------------- Try Your Own Text ----------------------
st.subheader("Try your own text")
user_text = st.text_area("Enter a review text", value="This arrived next day and the quality is fantastic. Totally worth it!")
col1, col2 = st.columns(2)

with col1:
    sia = load_vader()
    v = sia.polarity_scores(user_text)
    st.metric("VADER compound", f"{v['compound']:.3f}")
    st.write({k: round(v[k], 3) for k in ("pos", "neu", "neg")})

with col2:
    tokenizer, model = load_roberta()
    df_one = roberta_scores_batch([user_text], tokenizer, model)
    row = df_one.iloc[0]
    st.metric("RoBERTa positive", f"{float(row['roberta_pos']):.3f}")
    st.write({k: f"{float(row[k]):.3f}" for k in ("roberta_pos", "roberta_neu", "roberta_neg")})

st.caption("Notes: VADER is lexicon-based; RoBERTa is a transformer fine-tuned for sentiment. "
           "Running RoBERTa on the whole dataset can be slower on CPU; use the limit control.")