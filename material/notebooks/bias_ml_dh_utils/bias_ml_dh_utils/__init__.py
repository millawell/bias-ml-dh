import numpy as np
import torch as tr
from tqdm import tqdm
from bisect import bisect_left
import spacy
from lxml import html
from getpass import getpass
nlp = spacy.load("en_core_web_sm")
# tokenizer = nlp.Defaults.create_tokenizer(nlp)
tokenizer = nlp.tokenizer

def index_sorted_list(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def create_embedding_matrix(filepath):
    'Create embedding matrix to given vocab'
    vocab = []
    embedding_matrix = []

    with open(filepath) as f:
        for iline, line in tqdm(enumerate(f)):
            word, *vector = line.split()
            if "\xa0" in line:
                continue

            embedding_matrix.append(np.array(vector, dtype=np.float32))
            vocab.append(word)


    vocab.append("[PAD]")
    embedding_matrix = np.vstack(embedding_matrix)
    pad_vec = np.zeros((1, embedding_matrix.shape[1]))
    embedding_matrix = np.vstack([embedding_matrix, pad_vec])
    vocab = np.array(vocab)
    
    sorter = np.argsort(vocab)

    vocab = vocab[sorter]
    embedding_matrix = embedding_matrix[sorter]

    return embedding_matrix, vocab.tolist()


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
                embeddings[iword] = embedding_matrix[index_sorted_list(vocab, word)]
            except ValueError:
                pass

    return embeddings


data_identifier_dict = {
        "amazon_sentiment_english": "https://drive.google.com/u/0/uc?id=1FdsOuba3skNR1cL6m_jfD6Mfw4QW6fg2&export=download",
        "imdb_sentiment_english": "https://drive.google.com/u/0/uc?id=1rsy4Vj1Rlj3V5uvhfLryIPVYHmbf8dgL&export=download",
        "yelp_sentiment_english": "https://drive.google.com/u/0/uc?id=15tioKwjp0azhpFBzOeX_PpbzKnYvp6RT&export=download",
        "twitter_sentiment_german": "https://drive.google.com/u/0/uc?id=1avB1Ot50782TfOmVDtEZOUms1LNdD-Vu&export=download",
        "twitter_thueringen_small": "https://drive.google.com/u/0/uc?id=1NZt6qxkDlCA9VzM-icr5EjA3i4Mk_8Rs&export=download",
        # "twitter_thueringen": "https://drive.google.com/u/0/uc?id=18cgj5D81mheRv-9fpI7uh7rRyZYGbgWr&export=download",
        "glove.6B.50d": "https://drive.google.com/uc?export=download&id=18CJHOYJqDe3RjNa7dS9pWZM1Vp3R3lo9",
        "agression_comments_wikipedia": "https://drive.google.com/uc?export=download&id=1biF6BoNJxnuAzwybadqEeHlEKeSBGGn8",
    }
def get_link_from_identifier(id_):
    return data_identifier_dict[id_]

def get_suffix_from_identifier(id_):
    dict_ = {
        "amazon_sentiment_english": ".txt",
        "imdb_sentiment_english": ".txt",
        "yelp_sentiment_english": ".txt",
        "twitter_sentiment_german": ".pkl",
        "twitter_thueringen_small": ".jsonl",
        "twitter_thueringen": ".jsonl",
        "glove.6B.50d": ".txt",
        "agression_comments_wikipedia": ".pkl",
    }
    return dict_[id_]

def download_dataset(id_, password=None):

    import requests
    import os
    from zipfile import ZipFile

    url = get_link_from_identifier(id_)

    download_dir = os.path.join("data", id_)

    if not os.path.exists("data"):
        os.makedirs("data")


    out_file = os.path.join(download_dir, id_+get_suffix_from_identifier(id_)+".zip")
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
        if password is None:
            password = getpass("password for extracting zip: ")
            password = bytes(password, "utf-8")
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
