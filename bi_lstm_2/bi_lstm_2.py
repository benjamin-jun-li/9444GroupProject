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

embed_dim = 128
lstm_out = 196
batch_size = 32


# Store the scores
scores = []
accuracies = []
conf_matrices = []  # Store confusion matrices for each fold


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# Re-compile the model with L2 regularization
model = Sequential()
# model.add(Embedding(max_features, embed_dim, weights=[embedding_matrix], input_length=X.shape[1], trainable=True))
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))  # Add L2 regularization to the LSTM layer
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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

# 150/150 - 50s - loss: 0.5610 - accuracy: 0.7029 - val_loss: 0.3980 - val_accuracy: 0.8325 - 50s/epoch - 332ms/step
# Epoch 2/50
# 150/150 - 48s - loss: 0.3309 - accuracy: 0.8683 - val_loss: 0.3454 - val_accuracy: 0.8667 - 48s/epoch - 318ms/step
# Epoch 3/50
# 150/150 - 59s - loss: 0.2714 - accuracy: 0.8942 - val_loss: 0.3329 - val_accuracy: 0.8783 - 59s/epoch - 391ms/step
# Epoch 4/50
# 150/150 - 60s - loss: 0.2354 - accuracy: 0.9106 - val_loss: 0.3654 - val_accuracy: 0.8533 - 60s/epoch - 402ms/step
# Epoch 5/50
# 150/150 - 58s - loss: 0.2214 - accuracy: 0.9175 - val_loss: 0.3675 - val_accuracy: 0.8625 - 58s/epoch - 388ms/step
# Epoch 6/50
# 150/150 - 60s - loss: 0.1754 - accuracy: 0.9398 - val_loss: 0.3952 - val_accuracy: 0.8575 - 60s/epoch - 400ms/step
# 38/38 - 2s - loss: 0.3952 - accuracy: 0.8575 - 2s/epoch - 53ms/step
# 38/38 [==============================] - 2s 46ms/step
# Average score: 0.39521169662475586
# Average accuracy: 0.8575000166893005
# Confusion Matrix (Fold 1):
# [[536  97]
#  [ 74 493]]
#
# Classification Report (Fold 1):
#               precision    recall  f1-score   support
#
#     negative       0.88      0.85      0.86       633
#     positive       0.84      0.87      0.85       567
#
#     accuracy                           0.86      1200
#    macro avg       0.86      0.86      0.86      1200
# weighted avg       0.86      0.86      0.86      1200
#
#
# Overall Confusion Matrix:
#  [[536  97]
#  [ 74 493]]
# Overall Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.88      0.85      0.86       633
#     positive       0.84      0.87      0.85       567
#
#     accuracy                           0.86      1200
#    macro avg       0.86      0.86      0.86      1200
# weighted avg       0.86      0.86      0.86      1200
