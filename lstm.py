from dataProcessing import *
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
import numpy as np

# Data processing
train_texts, train_labels = load_data('../aclImdb_data/train')
test_texts, test_labels = load_data('../aclImdb_data/test')

train_texts = preprocess_text(train_texts)
test_texts = preprocess_text(test_texts)

# Set the maximum number of words we want to keep based on frequency
max_words = 10000

# Initialize a tokenizer
tokenizer = Tokenizer(num_words=max_words)

# Fit it on the texts
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

maxlen = 500
train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

vocabulary_size = max_words
embedding_dim = 128

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=maxlen))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Use early stopping to prevent overfitting
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

train_labels_array = np.array(train_labels)
model.fit(train_data, train_labels_array, batch_size=128, epochs=50, validation_split=0.2, callbacks=[early_stopping])
