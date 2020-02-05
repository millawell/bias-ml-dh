import numpy as np
from tqdm import tqdm
from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def create_embedding_matrix(filepath, vocab, embedding_dim):
    vocab_size = len(vocab)
    # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros([vocab_size, embedding_dim])

    with open(filepath) as f:
        for line in tqdm(f):
            word, *vector = line.split()
            embedding_matrix[index(vocab, word)] = np.array(vector, dtype=np.float32)
#             if word in vocab:
#                 idx = word_index[word]
#                 embedding_matrix[index(vocab, word)] = tr.from_numpy(np.array(
#                                         vector, dtype=np.float32))

    return embedding_matrix