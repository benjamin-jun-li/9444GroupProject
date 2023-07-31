import numpy as np
import pandas as pd
import re
from sklearn.utils import resample
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import Bidirectional
from gensim.models import Word2Vec
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns





data = pd.read_csv("../data/Tweets.csv")
data.shape



data = data[data['airline_sentiment_confidence'] > 0.5]
data.shape


class_count = data['airline_sentiment'].value_counts()
class0_count = class_count[0]
class1_count = class_count[1]
class2_count = class_count[2]



max_count = max(class0_count, class1_count, class2_count)
diff_class1 = max_count - class1_count
diff_class2 = max_count - class2_count

class1_samples = resample(data[data['airline_sentiment'] == 'neutral'],
                n_samples=diff_class1, replace=True, random_state=42)

class2_samples = resample(data[data['airline_sentiment'] == 'positive'],
                n_samples=diff_class2, replace=True, random_state=42)

data = pd.concat([data,class1_samples, class2_samples])

X = data['text']
y = data['airline_sentiment']

data = data[['text', 'airline_sentiment']]


def clean_train_data(x):
    text = x
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('\n', '', text)
    return text

data['text'] = data.text.apply(lambda x : clean_train_data(x))

max_features = 2000
token = Tokenizer(num_words=max_features, split = ' ')
token.fit_on_texts(data['text'].values)

X = token.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['airline_sentiment']).values

# Train Word2Vec embeddings
word2vec_embedding_dim = 128 # Set the desired embedding dimension
word2vec_model = Word2Vec(sentences=[text.split() for text in data['text'].values],
                          vector_size=word2vec_embedding_dim, window=3, min_count=1, workers=4)
# Create embedding matrix
embedding_matrix = np.zeros((max_features, word2vec_embedding_dim))
for word, i in token.word_index.items():
    if word in word2vec_model.wv and i < max_features:
        embedding_matrix[i] = word2vec_model.wv[word]

# Model architecture
embed_dim = word2vec_embedding_dim # Use Word2Vec embedding dimension
lstm_out = 196
batch_size = 32

# Store the scores
scores = []
accuracies = []
conf_matrices = []  # Store confusion matrices for each fold

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# Re-compile the model with L2 regularization
model = Sequential()
model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix], input_length=X.shape[1], trainable=True))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))  # Add L2 regularization to the LSTM layer
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Model checkpoint callback
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)


# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=batch_size, verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)

# Store the score and accuracy
scores.append(score)
accuracies.append(acc)

# Predict the test data and get the confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
conf_matrices.append(conf_matrix)

# Print the average score and accuracy
print('Average score:', np.mean(scores))
print('Average accuracy:', np.mean(accuracies))

# Print confusion matrices and classification reports for each fold
for i, conf_matrix in enumerate(conf_matrices):
    print(f"Confusion Matrix (Fold {i+1}):\n{conf_matrix}\n")
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=['negative', 'neutral', 'positive'])
    print(f"Classification Report (Fold {i+1}):\n{class_report}\n")

# Calculate and print overall confusion matrix and classification report
overall_conf_matrix = np.sum(conf_matrices, axis=0)
print("Overall Confusion Matrix:\n", overall_conf_matrix)
overall_class_report = classification_report(y_true_classes, y_pred_classes, target_names=['negative', 'neutral', 'positive'])
print("Overall Classification Report:\n", overall_class_report)

# Plot the overall confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Overall Confusion Matrix')
plt.show()


def predict_sentiment(text):
    # Clean the text
    text = clean_train_data(text)

    # Tokenize the text
    sequences = token.texts_to_sequences([text])
    sequences = pad_sequences(sequences, maxlen=X.shape[1])

    # Predict the sentiment
    prediction = model.predict(sequences)

    # Convert the prediction into a readable form
    sentiment = np.argmax(prediction)
    sentiment_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

    print('The sentiment of the text is', sentiment_dict[sentiment])


# Now use this function to predict the sentiment of a new text
new_text = "I am very happy with the service!"  # Replace with your text
predict_sentiment(new_text)

# 684/684 - 37s - loss: 0.7659 - accuracy: 0.6597 - val_loss: 0.5798 - val_accuracy: 0.7588 - 37s/epoch - 54ms/step
# Epoch 2/25
# 684/684 - 35s - loss: 0.5300 - accuracy: 0.7858 - val_loss: 0.4678 - val_accuracy: 0.8150 - 35s/epoch - 52ms/step
# Epoch 3/25
# 684/684 - 36s - loss: 0.4155 - accuracy: 0.8362 - val_loss: 0.3927 - val_accuracy: 0.8523 - 36s/epoch - 52ms/step
# Epoch 4/25
# 684/684 - 36s - loss: 0.3484 - accuracy: 0.8655 - val_loss: 0.3607 - val_accuracy: 0.8658 - 36s/epoch - 53ms/step
# Epoch 5/25
# 684/684 - 37s - loss: 0.3009 - accuracy: 0.8854 - val_loss: 0.3453 - val_accuracy: 0.8751 - 37s/epoch - 55ms/step
# Epoch 6/25
# 684/684 - 36s - loss: 0.2587 - accuracy: 0.9040 - val_loss: 0.3194 - val_accuracy: 0.8885 - 36s/epoch - 53ms/step
# Epoch 7/25
# 684/684 - 38s - loss: 0.2292 - accuracy: 0.9164 - val_loss: 0.3176 - val_accuracy: 0.9002 - 38s/epoch - 55ms/step
# Epoch 8/25
# 684/684 - 38s - loss: 0.2046 - accuracy: 0.9273 - val_loss: 0.3126 - val_accuracy: 0.9009 - 38s/epoch - 55ms/step
# Epoch 9/25
# 684/684 - 38s - loss: 0.1769 - accuracy: 0.9374 - val_loss: 0.2992 - val_accuracy: 0.9100 - 38s/epoch - 55ms/step
# Epoch 10/25
# 684/684 - 37s - loss: 0.1574 - accuracy: 0.9440 - val_loss: 0.2728 - val_accuracy: 0.9206 - 37s/epoch - 54ms/step
# Epoch 11/25
# 684/684 - 37s - loss: 0.1404 - accuracy: 0.9518 - val_loss: 0.2943 - val_accuracy: 0.9210 - 37s/epoch - 54ms/step
# Epoch 12/25
# 684/684 - 38s - loss: 0.1222 - accuracy: 0.9564 - val_loss: 0.2930 - val_accuracy: 0.9210 - 38s/epoch - 55ms/step
# Epoch 13/25
# 684/684 - 37s - loss: 0.1120 - accuracy: 0.9605 - val_loss: 0.2929 - val_accuracy: 0.9278 - 37s/epoch - 54ms/step
# 171/171 - 3s - loss: 0.2929 - accuracy: 0.9278 - 3s/epoch - 20ms/step
# 171/171 [==============================] - 2s 13ms/step
# Average score: 0.29285070300102234
# Average accuracy: 0.9277747273445129
# Confusion Matrix (Fold 1):
# [[1526  181   84]
#  [  46 1745   35]
#  [  23   26 1803]]
#
# Classification Report (Fold 1):
#               precision    recall  f1-score   support
#
#     negative       0.96      0.85      0.90      1791
#      neutral       0.89      0.96      0.92      1826
#     positive       0.94      0.97      0.96      1852
#
#     accuracy                           0.93      5469
#    macro avg       0.93      0.93      0.93      5469
# weighted avg       0.93      0.93      0.93      5469
#
#
# Overall Confusion Matrix:
#  [[1526  181   84]
#  [  46 1745   35]
#  [  23   26 1803]]
# Overall Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.96      0.85      0.90      1791
#      neutral       0.89      0.96      0.92      1826
#     positive       0.94      0.97      0.96      1852
#
#     accuracy                           0.93      5469
#    macro avg       0.93      0.93      0.93      5469
# weighted avg       0.93      0.93      0.93      5469
#
# 1/1 [==============================] - 0s 11ms/step
# The sentiment of the text is positive
