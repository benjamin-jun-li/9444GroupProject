from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import re

def clean_train_data(x):
    text = x
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    text = re.sub('\n', '', text)
    return text

# data processing
data = pd.read_csv("../data/threads1.csv")
data['review'] = data.review.apply(clean_train_data)

# TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english', max_features=10000)
X = tfidf.fit_transform(data['review'])
y = data['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm = LinearSVC(random_state=42)


svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


print('Accuracy:', accuracy_score(y_test, y_pred))


print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))


print('Classification Report:\n', classification_report(y_test, y_pred))

# Accuracy: 0.8491666666666666
# Confusion Matrix:
#  [[539  94]
#  [ 87 480]]
# Classification Report:
#                precision    recall  f1-score   support
#
#     negative       0.86      0.85      0.86       633
#     positive       0.84      0.85      0.84       567
#
#     accuracy                           0.85      1200
#    macro avg       0.85      0.85      0.85      1200
# weighted avg       0.85      0.85      0.85      1200
#
#
# Process finished with exit code 0
