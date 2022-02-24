import string
import json
import pickle
from collections import Counter
import nltk
from constants import FOLDER

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx or 'xxxx' in word:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


if __name__ == '__main__':
    
    with open( FOLDER + 'reports.json') as f:
        reports = json.load(f)

    counter = Counter()

    for caseid, report in reports.items():
        text = ''
        if report['findings'] is not None:
            text += report['findings']
        text += ' '
        if report['impression'] is not None:
            text += report['impression']
        text = text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
        text = text.replace('.', ' .')
        tokens = text.strip().split()

        counter.update(tokens)
    words = [word for word, cnt in counter.items() if cnt >= 3]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for word in words:
        vocab.add_word(word)

    with open( FOLDER + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print('Total vocabulary size {}. Saved to {}'.format(len(vocab), 'vocab.pkl'))
