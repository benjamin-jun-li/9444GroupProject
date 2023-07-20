import os
import glob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np

# Get the data, the source is sited.
# @InProceedings{maas-EtAl:2011:ACL-HLT2011,
#   author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
#   title     = {Learning Word Vectors for Sentiment Analysis},
#   booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
#   month     = {June},
#   year      = {2011},
#   address   = {Portland, Oregon, USA},
#   publisher = {Association for Computational Linguistics},
#   pages     = {142--150},
#   url       = {http://www.aclweb.org/anthology/P11-1015}
# }


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

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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
