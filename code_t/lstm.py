import numpy as np
import pandas as pd
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SimpleRNN, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint

data = pd.read_csv("../us_airline_data/Tweets.csv")
data.shape

data = data[data['airline_sentiment_confidence'] > 0.5]
data.shape

data = data[['text', 'airline_sentiment']]


def clean_train_data(x):
    text = x
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('\n', '', text)
    return text

data['text'] = data.text.apply(lambda x : clean_train_data(x))


# all_cat_data = data.copy()
#
# model1_data = data.copy()

max_features = 2000
token = Tokenizer(num_words=max_features, split = ' ')
token.fit_on_texts(data['text'].values)

X = token.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

Y = pd.get_dummies(data['airline_sentiment']).values

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

batch_size = 32
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Model checkpoint callback
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint])

# score = model.predict(X_test)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
print('score', score)
print('accuracy', acc)