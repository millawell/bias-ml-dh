import numpy as np
import torch as tr
from tqdm import tqdm
from bisect import bisect_left
import spacy
from lxml import html
nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def __index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def create_embedding_matrix(filepath, vocab, embedding_dim):
    'Create embedding matrix to given vocab'
    vocab_size = len(vocab)
    embedding_matrix = np.zeros([vocab_size, embedding_dim])

    with open(filepath) as f:
        for line in tqdm(f):
            word, *vector = line.split()
            try:
                embedding_matrix[__index(vocab, word)] = np.array(vector, dtype=np.float32)
            except ValueError:
                pass

    return embedding_matrix


def load_vocab(filepath):
    'Load vocab from trained glove embeddings'
    vocab = []

    with open(filepath) as f:
        for line in tqdm(f):
            word = line.split()[0]
            vocab.append(word)

    return sorted(vocab)


def lookup_embeddings(text, vocab, embedding_matrix):
    'For a given list of words, create embedding matrix'
    embeddings = np.zeros([len(text), embedding_matrix.shape[1]])

    for iword, word in enumerate(text):
        for token in tokenizer(str(word)):
            try:
                embeddings[iword] = embedding_matrix[__index(vocab, word)]
            except ValueError:
                pass

    return embeddings


def data_padding(word_seq, max_len=1000):
    input_data = tr.zeros([word_seq.shape[0], max_len], dtype=tr.int64)
    for i in range(word_seq.shape[0]):
        input_data[i, :len(word_seq[i])] = tr.Tensor(word_seq[i][:max_len])

    return input_data


def tokenize_data(comments, vocab, max_sentences=1000, REVIEWS=True):
    max_sentences = np.min([max_sentences, len(comments)])
    if not REVIEWS:
        y = np.array(labels[:max_sentences])
    word_seq = np.empty(max_sentences, dtype=object)

    for idx, sen in tqdm(enumerate(comments[:max_sentences])):
        #     doc = nlp(str(sen))
        word_seq[idx] = []
        for token in tokenizer(str(sen)):
            if (token.lemma_.lower() in vocab):
                word_seq[idx].append(__index(vocab, token.lemma_.lower()))

    return word_seq

def get_link_from_identifier(id_):
    dict_ = {
        "amazon_sentiment_english": "https://drive.google.com/u/0/uc?id=1YWlDDxlK2IXQCQ8oywLg5Mc4JS7XxFrK&export=download",
        "imdb_sentiment_english": "https://drive.google.com/u/0/uc?id=1ZuPfdOgdeuSDWwhlgZveJfdKhg-9TYK7&export=download",
        "yelp_sentiment_english": "https://drive.google.com/u/0/uc?id=1S2TMhBJQ9snj2ovbc2tA-C9aCVPmoXbW&export=download",
        "twitter_sentiment_german": "https://drive.google.com/u/0/uc?id=19MSkWtA1AxH1BEVN12E4Lqwe4oNUm0jt&export=download",
        "twitter_thueringen_small": "https://drive.google.com/u/0/uc?id=1sXaz50VC3qXIqnBsLsMT5vThOjT0fslC&export=download",
        "twitter_thueringen": "https://drive.google.com/u/0/uc?id=18cgj5D81mheRv-9fpI7uh7rRyZYGbgWr&export=download",
        "glove.6B.50d": "https://drive.google.com/uc?export=download&id=1S0ae8Q65ggjRqnK_NdzoMeuTGmMBIZXT",
        
    
    }
    return dict_[id_]

def get_suffix_from_identifier(id_):
    dict_ = {
        "amazon_sentiment_english": ".txt",
        "imdb_sentiment_english": ".txt",
        "yelp_sentiment_english": ".txt",
        "twitter_sentiment_german": ".pkl",
        "twitter_thueringen_small": ".jsonl",
        "twitter_thueringen": ".jsonl",
        "glove.6B.50d": ".txt",
    }
    return dict_[id_]

def download_dataset(id_, password):

    import requests
    import os
    from zipfile import ZipFile

    url = get_link_from_identifier(id_)

    download_dir = os.path.join("data", id_)

    if not os.path.exists("data"):
        os.makedirs("data")


    out_file = os.path.join(download_dir, id_+".zip")
    if not os.path.exists(out_file):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        with open(out_file, 'wb') as f:
            answer = requests.get(url)
            content = answer.content
            cookies = answer.cookies

            if len(content) < 1e4:
                
                tree = html.document_fromstring(content)
                href = tree.get_element_by_id("uc-download-link").attrib["href"]
                new_url = "https://drive.google.com/u/0"+href

                content = requests.get(new_url, cookies=cookies.get_dict()).content
                
            f.write(content)

    result_file = os.path.join(download_dir, id_+get_suffix_from_identifier(id_))

    if not os.path.exists(result_file):
        with ZipFile(out_file) as fin:
            fin.extractall(pwd=password, path=download_dir)

    return result_file

def get_max_row(word_seq):
    # returns the longest row to decide on padding

    max_row = 0
    for irow, row in enumerate(word_seq):
        if len(row) > max_row:
            max_row = len(row)

    return max_row
