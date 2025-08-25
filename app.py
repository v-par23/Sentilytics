import io
import zipfile
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from textblob import TextBlob
from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

st.set_page_config(page_title="🛒 Sentilytics — Amazon Review Sentiment", layout="wide")
st.title("🛒 Sentilytics — Amazon Review Sentiment Explorer")
st.caption("Compare VADER (lexicon) vs RoBERTa (transformer) on Amazon reviews")

# ---------- Helpers & Caching ----------

@st.cache_data(show_spinner=False)
def read_csv_or_zip(file_or_path: Union[str, io.BytesIO, "UploadedFile"]) -> pd.DataFrame:
    """Read a CSV or a ZIP containing a CSV."""
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

def ensure_binary_stars(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only reviews with 1 or 5 stars"""
    if "Score" in df.columns:
        df = df.copy()
        df = df[df["Score"].isin([1,5])]
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
    """Clean text by removing stopwords"""
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word.lower() not in stop_words]
    return " ".join(words)

def get_sentiment(text):
    """Get TextBlob sentiment polarity"""
    return TextBlob(text).sentiment.polarity

# ---------- Sidebar Controls ----------

with st.sidebar:
    st.header("⚙️ Options")
    data_source = st.selectbox("Select Data Source", ["Local CSV/ZIP", "Real Amazon Reviews (Hugging Face)"])
    uploaded = None

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

# ---------- Load Data ----------

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
        df = ensure_binary_stars(df)

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
        # Initialize df_v for Hugging Face data (no VADER scores needed)
        df_v = None

except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ---------- Compute Scores ----------

if data_source == "Local CSV/ZIP":
    with st.spinner("Computing VADER sentiment..."):
        df_v = add_vader_scores(df, text_col=text_col)

    if run_roberta:
        with st.spinner("Running RoBERTa (transformer) on dataset..."):
            df_r = add_roberta_scores(df, text_col=text_col, limit=int(roberta_limit))
    else:
        df_r = None

# For Hugging Face Amazon reviews, optionally run RoBERTa
if data_source == "Real Amazon Reviews (Hugging Face)":
    if run_roberta:
        with st.spinner("Running RoBERTa on Hugging Face Amazon reviews..."):
            df_r = add_roberta_scores(df, text_col="Text", limit=int(roberta_limit))
    else:
        df_r = None

# ---------- Overview ----------

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

# ---------- VADER vs Stars ----------

if data_source == "Local CSV/ZIP" and df_v is not None:
    group = df_v.groupby("Score")[["vader_pos", "vader_neu", "vader_neg", "vader_compound"]].mean()
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    group["vader_compound"].plot(kind="bar", ax=ax2)
    ax2.set_title("Average VADER compound by stars")
    ax2.set_xlabel("Stars")
    ax2.set_ylabel("Mean compound")
    st.pyplot(fig2)

    fig3, axs = plt.subplots(1, 3, figsize=(12, 3))
    sns.barplot(x=group.index, y=group["vader_pos"].values, ax=axs[0])
    sns.barplot(x=group.index, y=group["vader_neu"].values, ax=axs[1])
    sns.barplot(x=group.index, y=group["vader_neg"].values, ax=axs[2])
    axs[0].set_title("Positive")
    axs[1].set_title("Neutral")
    axs[2].set_title("Negative")
    for ax in axs:
        ax.set_xlabel("Stars")
        ax.set_ylabel("Mean")
    st.pyplot(fig3)

# ---------- RoBERTa vs Stars ----------

if df_r is not None and "Score" in df_r.columns:
    st.subheader("RoBERTa sentiment vs. star rating")
    group_r = df_r.groupby("Score")[["roberta_pos", "roberta_neu", "roberta_neg"]].mean()
    fig4, ax4 = plt.subplots(figsize=(7, 3))
    group_r["roberta_pos"].plot(kind="bar", ax=ax4)
    ax4.set_title("Average RoBERTa positive by stars")
    ax4.set_xlabel("Stars")
    ax4.set_ylabel("Mean positive probability")
    st.pyplot(fig4)

# ---------- Mismatch Explorer ----------

if df_r is not None and "Score" in df_r.columns:
    st.subheader("Where model sentiment disagrees with stars (interesting cases)")
    working = df_r.copy()
    working["disagree"] = np.where(
        working["Score"] == 1, working["roberta_pos"], working["roberta_neg"]
    )
    top_k = working.sort_values("disagree", ascending=False).head(10)
    show_cols = [c for c in [id_col if id_col in top_k.columns else None, "Score", text_col,
                             "roberta_pos", "roberta_neu", "roberta_neg"] if c]
    st.dataframe(top_k[show_cols], use_container_width=True)

# ---------- RoBERTa Sample Display ----------

if df_r is not None:
    st.subheader("RoBERTa Sentiment Probabilities")
    sample_df = df_r.head(10)[["Text", "roberta_neg", "roberta_neu", "roberta_pos"]]
    st.dataframe(sample_df)

# ---------- Try your own text ----------

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