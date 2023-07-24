import os
import glob
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Get the data, the source is sited.
# @InProceedings{maas-EtAl:2011:ACL-HLT2011,
#   author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
#   title     = {Learning Word Vectors for Sentiment Analysis},
#   booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
#   month     = {June},
#   year      = {2011},
#   address   = {Portland, Oregon, USA},pi
#   publisher = {Association for Computational Linguistics},
#   pages     = {142--150},
#   url       = {http://www.aclweb.org/anthology/P11-1015}
# }

class VectorDataset(Dataset):
    def __init__(self, data_list, target_list):
        self.data_list = data_list
        self.target_list = target_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.target_list[idx]

def load_data(directory):
    texts = []
    labels = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for fname in glob.glob(os.path.join(dir_name, '*.txt')):
            with open(fname, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
    return texts, labels


# Download the NLTK data package

# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialising word reducers and deactivators

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(texts):
    preprocessed_texts = []
    for text in texts:
        # Text cleaning: removes non-alphabetic characters
        text = re.sub(r'\W', ' ', text)

        # Tokenization
        words = nltk.word_tokenize(text)

        # Word Restoration and Deactivation Removal
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

        preprocessed_texts.append(' '.join(words))
    return preprocessed_texts


# Train a Word2Vec model
def w2v_train(texts):
    train_tokens = [text.split() for text in texts]
    w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)
    return w2v_model


def text_to_vec(text, model):
    words = text.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv.key_to_index]
    return np.mean(word_vecs, axis=0)


# For the ConvLstm
def text_to_vec_con(text, model, max_seq_length):
    words = text.split()
    word_vecs = []
    for word in words:
        if word in model.wv.key_to_index:
            word_vecs.append(model.wv[word])
    for i in range(max_seq_length - len(words)):
        word_vecs.append(np.zeros(model.vector_size))
    return np.array(word_vecs[:max_seq_length])

# Load data
train_texts, train_labels = load_data('../aclImdb_data/train')
test_texts, test_labels = load_data('../aclImdb_data/test')

# Preprocess texts
train_texts = preprocess_text(train_texts)
test_texts = preprocess_text(test_texts)


np.save('../p_data/train_texts.npy', train_texts)
np.save('../p_data/train_labels.npy', train_labels)
np.save('../p_data/test_texts.npy', test_texts)
np.save('../p_data/test_labels.npy', test_labels)



#
# # Convert texts to vectors
# train_data = [text_to_vec(text, w2v_model) for text in train_texts]
# test_data = [text_to_vec(text, w2v_model) for text in test_texts]
#
# # Divide the training set and validation set
# train_data, val_data, train_labels, val_labels = train_test_split(
#     train_data, train_labels, test_size=0.2, random_state=42)
#
# # Convert lists to tensors
# train_data = [torch.tensor(vec) for vec in train_data]
# val_data = [torch.tensor(vec) for vec in val_data]
# test_data = [torch.tensor(vec) for vec in test_data]
#
# np.save('train_data.npy', train_data)
# np.save('train_labels.npy', train_labels)
# np.save('val_data.npy', val_data)
# np.save('val_labels.npy', val_labels)
# np.save('test_data.npy', test_data)
# np.save('test_labels.npy', test_labels)

# # For conv_lstm
# max_length = 1519
# train_data_conv = [text_to_vec_con(text, w2v_model, max_length) for text in train_texts]
# test_data_conv = [text_to_vec_con(text, w2v_model, max_length) for text in test_texts]
#
# # Divide the training set and validation set
# train_data_conv, val_data_conv, train_labels_conv, val_labels_conv = train_test_split(
#     train_data_conv, train_labels, test_size=0.2, random_state=42)
#
# # Convert lists to tensors
# train_data_conv = [torch.tensor(vec).unsqueeze(0) for vec in train_data_conv]
# val_data_conv = [torch.tensor(vec).unsqueeze(0) for vec in val_data_conv]
# test_data_conv = [torch.tensor(vec).unsqueeze(0) for vec in test_data_conv]
#
# # Create DataLoader
# train_loader_conv = DataLoader(VectorDataset(train_data_conv, train_labels), batch_size)
# val_loader_conv = DataLoader(VectorDataset(val_data_conv, val_labels), batch_size)
# test_loader_conv = DataLoader(VectorDataset(test_data_conv, test_labels), batch_size)
#
# np.save('../p_data/train_loader_conv.npy', np.array(train_loader_conv))
# np.save('../p_data/val_loader_conv.npy', np.array(val_loader_conv))
# np.save('../p_data/test_loader_conv.npy', np.array(test_loader_conv))