from tqdm import tqdm
from utils import *
import numpy as np
import sys

class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("<unk>")]


def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    word_indexer.add_and_get_index("<pad>")
    word_indexer.add_and_get_index("<start>")
    word_indexer.add_and_get_index("<end>")
    vectors.append(np.random.randn(200))
    vectors.append(np.random.randn(200))
    vectors.append(np.random.randn(200))
    for line in tqdm(f):
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print('Vector size:',len(vectors))
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


def relativize(file, outfile, words):
    """
    Relativize the word vectors to the given dataset represented by word counts
    :param file: word vectors file
    :param outfile: output file
    :param word_counter: Counter of words occurring in train/dev/test data
    :return:
    """
    f = open(file)
    o = open(outfile, 'w')
    voc = []
    for line in tqdm(f):
        word = line[:line.find(' ')]
        if word in words:
            voc.append(word)
            o.write(line)
    for word in words:
        if word not in voc:
            print(f"Missing word {word}")
    f.close()
    o.close()


def relativize_data():

    with open("data/vocab.txt", 'r') as file:
        words = []
        for line in file:
            words.append(line.strip())
    
    relativize("data/glove.6B.200d.txt", "data/glove.6B.200d-relativized.txt", words)


def text_to_id(text, word_vectors: WordEmbeddings):
    ids = []
    for word in text:
        word_idx = word_vectors.word_indexer.index_of(word)
        if word_idx != -1:
            ids.append(word_vectors.word_indexer.index_of(word))
        else:
            ids.append(word_vectors.word_indexer.index_of("<unk>"))
    return torch.tensor(ids)
    
