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

# 读取数据并预处理
data = pd.read_csv("../data/Tweets.csv")
data['text'] = data.text.apply(clean_train_data)

# 用TF-IDF向量化文本数据
tfidf = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore', stop_words='english', max_features=10000)
X = tfidf.fit_transform(data['text'])
y = data['airline_sentiment']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化一个SVM模型
svm = LinearSVC(random_state=42)

# 训练模型
svm.fit(X_train, y_train)

# 用模型进行预测
y_pred = svm.predict(X_test)

# 输出模型的准确率
print('Accuracy:', accuracy_score(y_test, y_pred))

# 输出混淆矩阵
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# 输出分类报告
print('Classification Report:\n', classification_report(y_test, y_pred))