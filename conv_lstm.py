import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from dataProcessing import *  # Assuming dataProcessing.py contains your data loading and preprocessing functions

# Data processing
train_texts, train_labels = load_data('../aclImdb_data/train')
test_texts, test_labels = load_data('../aclImdb_data/test')

train_texts = preprocess_text(train_texts)
test_texts = preprocess_text(test_texts)

w2v_model = w2v_train(train_texts)

# Set the maximum number of words we want to keep based on frequency
max_words = 10000

# Initialize a tokenizer
tokenizer = Tokenizer(num_words=max_words)

# Fit it on the texts
tokenizer.fit_on_texts(train_texts)

# Create a weight matrix for words in training docs
embedding_matrix = np.zeros((max_words, w2v_model.vector_size))
for word, i in tokenizer.word_index.items():
    if i < max_words:  # words indexed max_words and above are discarded
        if word in w2v_model.wv:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

maxlen = 500
train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

vocabulary_size = max_words
embedding_dim = w2v_model.vector_size

# Define your model architecture
def create_model():
    model = Sequential()
    model.add(Embedding(vocabulary_size,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=maxlen,
                        trainable=False))  # Set the Embedding layer to not trainable
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# Perform K-Fold cross-validation
k_folds = 5  # Number of folds
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Lists to store the evaluation results
accuracy_scores = []
loss_scores = []

for train_index, val_index in skf.split(train_data, train_labels):
    # Split the data into training and validation sets for this fold
    X_train_fold, X_val_fold = train_data[train_index], train_data[val_index]
    y_train_fold, y_val_fold = train_labels[train_index], train_labels[val_index]

    # Create a new instance of the model for each fold
    model = create_model()

    # Train the model on the current fold
    model.fit(X_train_fold, y_train_fold, batch_size=128, epochs=50, verbose=1)

    # Evaluate the model on the validation data
    loss, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)

    # Store the evaluation scores for this fold
    accuracy_scores.append(accuracy)
    loss_scores.append(loss)

# Calculate and print the average performance across all folds
print("Average accuracy:", np.mean(accuracy_scores))
print("Average loss:", np.mean(loss_scores))
