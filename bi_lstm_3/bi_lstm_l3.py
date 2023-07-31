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


# Store the scores
scores = []
accuracies = []
conf_matrices = []  # Store confusion matrices for each fold

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

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

# 684/684 - 38s - loss: 1.0973 - accuracy: 0.6288 - val_loss: 0.7020 - val_accuracy: 0.7407 - 38s/epoch - 55ms/step
# Epoch 2/50
# 684/684 - 36s - loss: 0.6880 - accuracy: 0.7463 - val_loss: 0.6039 - val_accuracy: 0.7822 - 36s/epoch - 53ms/step
# Epoch 3/50
# 684/684 - 37s - loss: 0.5793 - accuracy: 0.7966 - val_loss: 0.5351 - val_accuracy: 0.8133 - 37s/epoch - 55ms/step
# Epoch 4/50
# 684/684 - 38s - loss: 0.5123 - accuracy: 0.8270 - val_loss: 0.4918 - val_accuracy: 0.8320 - 38s/epoch - 55ms/step
# Epoch 5/50
# 684/684 - 38s - loss: 0.4701 - accuracy: 0.8438 - val_loss: 0.4784 - val_accuracy: 0.8431 - 38s/epoch - 56ms/step
# Epoch 6/50
# 684/684 - 38s - loss: 0.4420 - accuracy: 0.8542 - val_loss: 0.4722 - val_accuracy: 0.8406 - 38s/epoch - 55ms/step
# Epoch 7/50
# 684/684 - 37s - loss: 0.4234 - accuracy: 0.8590 - val_loss: 0.4439 - val_accuracy: 0.8513 - 37s/epoch - 54ms/step
# Epoch 8/50
# 684/684 - 37s - loss: 0.4010 - accuracy: 0.8697 - val_loss: 0.4442 - val_accuracy: 0.8587 - 37s/epoch - 54ms/step
# Epoch 9/50
# 684/684 - 37s - loss: 0.3881 - accuracy: 0.8737 - val_loss: 0.4305 - val_accuracy: 0.8673 - 37s/epoch - 54ms/step
# Epoch 10/50
# 684/684 - 37s - loss: 0.3740 - accuracy: 0.8785 - val_loss: 0.4291 - val_accuracy: 0.8663 - 37s/epoch - 54ms/step
# Epoch 11/50
# 684/684 - 37s - loss: 0.3589 - accuracy: 0.8868 - val_loss: 0.4230 - val_accuracy: 0.8691 - 37s/epoch - 54ms/step
# Epoch 12/50
# 684/684 - 37s - loss: 0.3495 - accuracy: 0.8894 - val_loss: 0.4136 - val_accuracy: 0.8724 - 37s/epoch - 54ms/step
# Epoch 13/50
# 684/684 - 36s - loss: 0.3410 - accuracy: 0.8918 - val_loss: 0.4144 - val_accuracy: 0.8738 - 36s/epoch - 53ms/step
# Epoch 14/50
# 684/684 - 36s - loss: 0.3292 - accuracy: 0.8961 - val_loss: 0.4088 - val_accuracy: 0.8758 - 36s/epoch - 53ms/step
# Epoch 15/50
# 684/684 - 36s - loss: 0.3234 - accuracy: 0.8987 - val_loss: 0.4132 - val_accuracy: 0.8762 - 36s/epoch - 53ms/step
# Epoch 16/50
# 684/684 - 36s - loss: 0.3148 - accuracy: 0.9041 - val_loss: 0.4190 - val_accuracy: 0.8726 - 36s/epoch - 53ms/step
# Epoch 17/50
# 684/684 - 37s - loss: 0.3040 - accuracy: 0.9086 - val_loss: 0.3950 - val_accuracy: 0.8854 - 37s/epoch - 54ms/step
# Epoch 18/50
# 684/684 - 37s - loss: 0.2990 - accuracy: 0.9115 - val_loss: 0.3971 - val_accuracy: 0.8857 - 37s/epoch - 54ms/step
# Epoch 19/50
# 684/684 - 37s - loss: 0.2933 - accuracy: 0.9127 - val_loss: 0.3916 - val_accuracy: 0.8850 - 37s/epoch - 54ms/step
# Epoch 20/50
# 684/684 - 37s - loss: 0.2897 - accuracy: 0.9132 - val_loss: 0.3922 - val_accuracy: 0.8874 - 37s/epoch - 54ms/step
# Epoch 21/50
# 684/684 - 37s - loss: 0.2826 - accuracy: 0.9148 - val_loss: 0.3805 - val_accuracy: 0.8897 - 37s/epoch - 55ms/step
# Epoch 22/50
# 684/684 - 37s - loss: 0.2812 - accuracy: 0.9160 - val_loss: 0.3985 - val_accuracy: 0.8854 - 37s/epoch - 55ms/step
# Epoch 23/50
# 684/684 - 37s - loss: 0.2711 - accuracy: 0.9222 - val_loss: 0.4051 - val_accuracy: 0.8879 - 37s/epoch - 55ms/step
# Epoch 24/50
# 684/684 - 38s - loss: 0.2687 - accuracy: 0.9216 - val_loss: 0.3831 - val_accuracy: 0.8929 - 38s/epoch - 55ms/step
# 171/171 - 4s - loss: 0.3831 - accuracy: 0.8929 - 4s/epoch - 23ms/step
# 171/171 [==============================] - 3s 18ms/step
# Average score: 0.38307949900627136
# Average accuracy: 0.8928506374359131
# Confusion Matrix (Fold 1):
# [[1484  217   90]
#  [ 112 1643   71]
#  [  38   58 1756]]
#
# Classification Report (Fold 1):
#               precision    recall  f1-score   support
#
#     negative       0.91      0.83      0.87      1791
#      neutral       0.86      0.90      0.88      1826
#     positive       0.92      0.95      0.93      1852
#
#     accuracy                           0.89      5469
#    macro avg       0.89      0.89      0.89      5469
# weighted avg       0.89      0.89      0.89      5469
#
#
# Overall Confusion Matrix:
#  [[1484  217   90]
#  [ 112 1643   71]
#  [  38   58 1756]]
# Overall Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.91      0.83      0.87      1791
#      neutral       0.86      0.90      0.88      1826
#     positive       0.92      0.95      0.93      1852
#
#     accuracy                           0.89      5469
#    macro avg       0.89      0.89      0.89      5469
# weighted avg       0.89      0.89      0.89      5469
