import numpy as np
import pandas as pd
import re

from keras.src.layers import Bidirectional
from sklearn.utils import resample
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import ConvLSTM2D, Flatten, Reshape, Conv1D, MaxPooling1D


data = pd.read_csv("../data/threads1.csv")
data.shape

def clean_train_data(x):
    text = str(x)  # ensure x is str
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('\n', '', text)
    return text

data['review'] = data.review.apply(lambda x : clean_train_data(x))
data = data[data['sentiment'] != 'neutral']

print(data['review'])
max_features = 2000
token = Tokenizer(num_words=max_features, split = ' ')
token.fit_on_texts(data['review'].values)

X = token.texts_to_sequences(data['review'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['sentiment']).values

# Model architecture
embed_dim = 128 # Use Word2Vec embedding dimension
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

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01))))
model.add(Dense(2, activation='softmax'))
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
    class_report = classification_report(y_true_classes, y_pred_classes, target_names=['negative', 'positive'])
    print(f"Classification Report (Fold {i+1}):\n{class_report}\n")

# Calculate and print overall confusion matrix and classification report
overall_conf_matrix = np.sum(conf_matrices, axis=0)
print("Overall Confusion Matrix:\n", overall_conf_matrix)
overall_class_report = classification_report(y_true_classes, y_pred_classes, target_names=['negative', 'positive'])
print("Overall Classification Report:\n", overall_class_report)

# Plot the overall confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Overall Confusion Matrix')
plt.show()

# Epoch 1/50
# 150/150 - 23s - loss: 0.7573 - accuracy: 0.7252 - val_loss: 0.3769 - val_accuracy: 0.8608 - 23s/epoch - 154ms/step
# Epoch 2/50
# 150/150 - 23s - loss: 0.3222 - accuracy: 0.8875 - val_loss: 0.3681 - val_accuracy: 0.8617 - 23s/epoch - 152ms/step
# Epoch 3/50
# 150/150 - 23s - loss: 0.2567 - accuracy: 0.9133 - val_loss: 0.3591 - val_accuracy: 0.8808 - 23s/epoch - 156ms/step
# Epoch 4/50
# 150/150 - 23s - loss: 0.2091 - accuracy: 0.9300 - val_loss: 0.4021 - val_accuracy: 0.8483 - 23s/epoch - 154ms/step
# Epoch 5/50
# 150/150 - 23s - loss: 0.1699 - accuracy: 0.9458 - val_loss: 0.4226 - val_accuracy: 0.8675 - 23s/epoch - 154ms/step
# Epoch 6/50
# 150/150 - 24s - loss: 0.1340 - accuracy: 0.9590 - val_loss: 0.4389 - val_accuracy: 0.8717 - 24s/epoch - 158ms/step
# 38/38 - 1s - loss: 0.4389 - accuracy: 0.8717 - 1s/epoch - 35ms/step
# 38/38 [==============================] - 1s 33ms/step
# Average score: 0.4389059543609619
# Average accuracy: 0.871666669845581
# Confusion Matrix (Fold 1):
# [[585  73]
#  [ 81 461]]
# 
# Classification Report (Fold 1):
#               precision    recall  f1-score   support
# 
#     negative       0.88      0.89      0.88       658
#     positive       0.86      0.85      0.86       542
# 
#     accuracy                           0.87      1200
#    macro avg       0.87      0.87      0.87      1200
# weighted avg       0.87      0.87      0.87      1200
# 
# 
# Overall Confusion Matrix:
#  [[585  73]
#  [ 81 461]]
# Overall Classification Report:
#                precision    recall  f1-score   support
# 
#     negative       0.88      0.89      0.88       658
#     positive       0.86      0.85      0.86       542
# 
#     accuracy                           0.87      1200
#    macro avg       0.87      0.87      0.87      1200
# weighted avg       0.87      0.87      0.87      1200
