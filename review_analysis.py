import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize


plt.style.use("ggplot")
import nltk

df = pd.read_csv('NewReviews.csv.zip')
df = df.head(500)
ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10,5))
ax.set_xlabel('Review Stars')
# plt.show()

# Basic NLTK
example = df['Text'][50]
nltk.download('punkt')
tokens = nltk.word_tokenize(example) #splits in list each word, this is a way the computer can process it
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()

# run polarity score on every data set
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T #makes the data look nice
vaders = vaders.reset_index().rename(columns={'index':'Id'})
vaders = vaders.merge(df, how='left')

ax = sns.barplot(data=vaders, x="Score", y='compound')
ax.set_title('Compound Score by Amazon Store Review')

figs, axs = plt.subplots(1,3,figsize=(12,3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
plt.show() #scores each word and review individually

#Roberta pretrained model connects words for meanings
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Run so roberta understands
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
        break
    except RuntimeError:
        print(f'broke for ID {myid}')


results_df = pd.DataFrame(res).T #makes the data look nice
results_df = results_df.reset_index().rename(columns={'index':'Id'})
results_df = results_df.merge(df, how='left')

results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]
# 'I felt energized within five minutes, but it lasted for about 45 minutes. I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money.'
results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]
# 'So we cancelled the order.  It was cancelled without any problem.  That is a positive note...'

#this determines the sentiment for any sentence you input
from transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis")
