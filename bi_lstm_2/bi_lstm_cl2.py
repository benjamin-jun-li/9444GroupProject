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


# Re-compile the model with L2 regularization
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.01))))  # Add L2 regularization to the LSTM layer
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


# 150/150 - 53s - loss: 1.3200 - accuracy: 0.6660 - val_loss: 0.4821 - val_accuracy: 0.8133 - 53s/epoch - 356ms/step
# Epoch 2/50
# 150/150 - 63s - loss: 0.4780 - accuracy: 0.8200 - val_loss: 0.4208 - val_accuracy: 0.8492 - 63s/epoch - 422ms/step
# Epoch 3/50
# 150/150 - 63s - loss: 0.3440 - accuracy: 0.8823 - val_loss: 0.3597 - val_accuracy: 0.8683 - 63s/epoch - 423ms/step
# Epoch 4/50
# 150/150 - 63s - loss: 0.2993 - accuracy: 0.8929 - val_loss: 0.3529 - val_accuracy: 0.8683 - 63s/epoch - 417ms/step
# Epoch 5/50
# 150/150 - 64s - loss: 0.2750 - accuracy: 0.9073 - val_loss: 0.3640 - val_accuracy: 0.8608 - 64s/epoch - 427ms/step
# Epoch 6/50
# 150/150 - 64s - loss: 0.2540 - accuracy: 0.9173 - val_loss: 0.3466 - val_accuracy: 0.8700 - 64s/epoch - 424ms/step
# Epoch 7/50
# 150/150 - 63s - loss: 0.2312 - accuracy: 0.9219 - val_loss: 0.3677 - val_accuracy: 0.8650 - 63s/epoch - 421ms/step
# Epoch 8/50
# 150/150 - 63s - loss: 0.2146 - accuracy: 0.9312 - val_loss: 0.3816 - val_accuracy: 0.8467 - 63s/epoch - 423ms/step
# Epoch 9/50
# 150/150 - 62s - loss: 0.2350 - accuracy: 0.9231 - val_loss: 0.3627 - val_accuracy: 0.8775 - 62s/epoch - 417ms/step
# 38/38 - 2s - loss: 0.3627 - accuracy: 0.8775 - 2s/epoch - 56ms/step
# 38/38 [==============================] - 2s 48ms/step
# Average score: 0.3627375662326813
# Average accuracy: 0.8774999976158142
# Confusion Matrix (Fold 1):
# [[567  65]
#  [ 82 486]]
#
# Classification Report (Fold 1):
#               precision    recall  f1-score   support
#
#     negative       0.87      0.90      0.89       632
#     positive       0.88      0.86      0.87       568
#
#     accuracy                           0.88      1200
#    macro avg       0.88      0.88      0.88      1200
# weighted avg       0.88      0.88      0.88      1200
#
#
# Overall Confusion Matrix:
#  [[567  65]
#  [ 82 486]]
# Overall Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.87      0.90      0.89       632
#     positive       0.88      0.86      0.87       568
#
#     accuracy                           0.88      1200
#    macro avg       0.88      0.88      0.88      1200
# weighted avg       0.88      0.88      0.88      1200
#
#
# Process finished with exit code 0
