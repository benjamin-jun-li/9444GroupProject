from dataProcessing import *
# Load data
train_texts, train_labels = load_data('../aclImdb_data/train')
test_texts, test_labels = load_data('../aclImdb_data/test')

# Preprocess texts
train_texts = preprocess_text(train_texts)
test_texts = preprocess_text(test_texts)

# Train a Word2Vec model
w2v_model = w2v_train(train_texts)

# Convert texts to vectors

def get_max_length(texts):
    max_length = 0
    for text in texts:
        words = text.split()
        if len(words) > max_length:
            max_length = len(words)
    return max_length

max_length_train = get_max_length(train_texts)
max_length_test = get_max_length(test_texts)

print(max(max_length_train,max_length_test))