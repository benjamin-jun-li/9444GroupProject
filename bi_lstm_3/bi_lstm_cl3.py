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
from keras.layers import Bidirectional
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

# Define number of splits
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)

# Store the scores
scores = []
accuracies = []
conf_matrices = []  # Store confusion matrices for each fold

# k-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


# Re-compile the model with L2 regularization
model = Sequential()
model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix], input_length=X.shape[1], trainable=True))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01))))  # Add L2 regularization to the LSTM layer
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Model checkpoint callback
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=2,
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


# 684/684 - 36s - loss: 1.0975 - accuracy: 0.6310 - val_loss: 0.7541 - val_accuracy: 0.7212 - 36s/epoch - 53ms/step
# Epoch 2/50
# 684/684 - 35s - loss: 0.6901 - accuracy: 0.7433 - val_loss: 0.5903 - val_accuracy: 0.7906 - 35s/epoch - 51ms/step
# Epoch 3/50
# 684/684 - 35s - loss: 0.5752 - accuracy: 0.8000 - val_loss: 0.5471 - val_accuracy: 0.8069 - 35s/epoch - 51ms/step
# Epoch 4/50
# 684/684 - 35s - loss: 0.5155 - accuracy: 0.8235 - val_loss: 0.4971 - val_accuracy: 0.8287 - 35s/epoch - 51ms/step
# Epoch 5/50
# 684/684 - 35s - loss: 0.4735 - accuracy: 0.8404 - val_loss: 0.4787 - val_accuracy: 0.8382 - 35s/epoch - 51ms/step
# Epoch 6/50
# 684/684 - 35s - loss: 0.4415 - accuracy: 0.8528 - val_loss: 0.4528 - val_accuracy: 0.8501 - 35s/epoch - 51ms/step
# Epoch 7/50
# 684/684 - 36s - loss: 0.4200 - accuracy: 0.8596 - val_loss: 0.4498 - val_accuracy: 0.8541 - 36s/epoch - 53ms/step
# Epoch 8/50
# 684/684 - 36s - loss: 0.4086 - accuracy: 0.8675 - val_loss: 0.4334 - val_accuracy: 0.8546 - 36s/epoch - 53ms/step
# Epoch 9/50
# 684/684 - 36s - loss: 0.3857 - accuracy: 0.8722 - val_loss: 0.4216 - val_accuracy: 0.8576 - 36s/epoch - 53ms/step
# Epoch 10/50
# 684/684 - 36s - loss: 0.3755 - accuracy: 0.8801 - val_loss: 0.4175 - val_accuracy: 0.8603 - 36s/epoch - 53ms/step
# Epoch 11/50
# 684/684 - 36s - loss: 0.3611 - accuracy: 0.8839 - val_loss: 0.4136 - val_accuracy: 0.8669 - 36s/epoch - 53ms/step
# Epoch 12/50
# 684/684 - 36s - loss: 0.3490 - accuracy: 0.8896 - val_loss: 0.4183 - val_accuracy: 0.8649 - 36s/epoch - 52ms/step
# Epoch 13/50
# 684/684 - 36s - loss: 0.3423 - accuracy: 0.8924 - val_loss: 0.4010 - val_accuracy: 0.8685 - 36s/epoch - 53ms/step
# Epoch 14/50
# 684/684 - 36s - loss: 0.3341 - accuracy: 0.8953 - val_loss: 0.4051 - val_accuracy: 0.8713 - 36s/epoch - 53ms/step
# Epoch 15/50
# 684/684 - 36s - loss: 0.3232 - accuracy: 0.8993 - val_loss: 0.3993 - val_accuracy: 0.8769 - 36s/epoch - 53ms/step
# Epoch 16/50
# 684/684 - 37s - loss: 0.3215 - accuracy: 0.9043 - val_loss: 0.4121 - val_accuracy: 0.8720 - 37s/epoch - 54ms/step
# Epoch 17/50
# 684/684 - 36s - loss: 0.3067 - accuracy: 0.9073 - val_loss: 0.3922 - val_accuracy: 0.8777 - 36s/epoch - 53ms/step
# Epoch 18/50
# 684/684 - 36s - loss: 0.3004 - accuracy: 0.9077 - val_loss: 0.3911 - val_accuracy: 0.8795 - 36s/epoch - 53ms/step
# Epoch 19/50
# 684/684 - 37s - loss: 0.2923 - accuracy: 0.9139 - val_loss: 0.3881 - val_accuracy: 0.8828 - 37s/epoch - 54ms/step
# Epoch 20/50
# 684/684 - 37s - loss: 0.2886 - accuracy: 0.9160 - val_loss: 0.3986 - val_accuracy: 0.8843 - 37s/epoch - 54ms/step
# Epoch 21/50
# 684/684 - 37s - loss: 0.2802 - accuracy: 0.9195 - val_loss: 0.3973 - val_accuracy: 0.8813 - 37s/epoch - 54ms/step
# Epoch 22/50
# 684/684 - 37s - loss: 0.2763 - accuracy: 0.9201 - val_loss: 0.3860 - val_accuracy: 0.8916 - 37s/epoch - 54ms/step
# Epoch 23/50
# 684/684 - 37s - loss: 0.2729 - accuracy: 0.9215 - val_loss: 0.3886 - val_accuracy: 0.8907 - 37s/epoch - 54ms/step
# Epoch 24/50
# 684/684 - 37s - loss: 0.2643 - accuracy: 0.9262 - val_loss: 0.3725 - val_accuracy: 0.8923 - 37s/epoch - 54ms/step
# Epoch 25/50
# 684/684 - 37s - loss: 0.2608 - accuracy: 0.9272 - val_loss: 0.3741 - val_accuracy: 0.8947 - 37s/epoch - 54ms/step
# Epoch 26/50
# 684/684 - 37s - loss: 0.2531 - accuracy: 0.9305 - val_loss: 0.3712 - val_accuracy: 0.8987 - 37s/epoch - 54ms/step
# Epoch 27/50
# 684/684 - 37s - loss: 0.2499 - accuracy: 0.9301 - val_loss: 0.3770 - val_accuracy: 0.8938 - 37s/epoch - 53ms/step
# Epoch 28/50
# 684/684 - 37s - loss: 0.2471 - accuracy: 0.9342 - val_loss: 0.3802 - val_accuracy: 0.8961 - 37s/epoch - 53ms/step
# Epoch 29/50
# 684/684 - 36s - loss: 0.2420 - accuracy: 0.9334 - val_loss: 0.3744 - val_accuracy: 0.9002 - 36s/epoch - 53ms/step
# 171/171 - 3s - loss: 0.3744 - accuracy: 0.9002 - 3s/epoch - 19ms/step
# 171/171 [==============================] - 2s 13ms/step
# Average score: 0.37443798780441284
# Average accuracy: 0.9001645445823669
# Confusion Matrix (Fold 1):
# [[1495  244   81]
#  [  92 1749   67]
#  [  24   38 1679]]
#
# Classification Report (Fold 1):
#               precision    recall  f1-score   support
#
#     negative       0.93      0.82      0.87      1820
#      neutral       0.86      0.92      0.89      1908
#     positive       0.92      0.96      0.94      1741
#
#     accuracy                           0.90      5469
#    macro avg       0.90      0.90      0.90      5469
# weighted avg       0.90      0.90      0.90      5469
#
#
# Overall Confusion Matrix:
#  [[1495  244   81]
#  [  92 1749   67]
#  [  24   38 1679]]
# Overall Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.93      0.82      0.87      1820
#      neutral       0.86      0.92      0.89      1908
#     positive       0.92      0.96      0.94      1741
#
#     accuracy                           0.90      5469
#    macro avg       0.90      0.90      0.90      5469
# weighted avg       0.90      0.90      0.90      5469
