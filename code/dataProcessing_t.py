import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


# Load the dataset
df = pd.read_csv('../us_airline_data/Tweets.csv')

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df['cleaned_text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df['cleaned_text'] = df['cleaned_text'].str.lower()
df['cleaned_text'] = df['cleaned_text'].str.replace('[^\w\s]','')
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.split())  # Convert each sentence to a list of words after all the string operations

le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['airline_sentiment'])

train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)


np.save('../p_data_3/train_texts.npy', train_texts)
np.save('../p_data_3/train_labels.npy', train_labels)
np.save('../p_data_3/test_texts.npy', test_texts)
np.save('../p_data_3/test_labels.npy', test_labels)
np.save('../p_data_3/val_texts.npy', val_texts)
np.save('../p_data_3/val_labels.npy', val_labels)


