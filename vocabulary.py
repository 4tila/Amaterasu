from pickle import load
import numpy as np
from os import listdir
path = './path/'
books = list(listdir(f'{path}'))
books = [book for book in books if 'tokens_' in book]
encoding = np.load(f'{path}encoding.npy')
vocabulary =  {encoding.shape[0]-1} # include last element as it corresponds to [EOS] which is necessary or prediction
for book in books:
    print(book[7:]) # removes tokens_ from the print
    with open(f'{path}{book}', 'rb') as f: sentences = load(f)
    sentences = np.concatenate(sentences)
    sentences = np.unique(sentences)
    vocabulary = vocabulary.union(set(sentences))
vocabulary = np.array(list(vocabulary))
vocabulary_encoding = encoding
vocabulary_encoding = vocabulary_encoding[vocabulary]
vocabulary_encoding = np.unique(vocabulary_encoding)
np.save(f'{path}vocabulary_encoding.npy', vocabulary_encoding)
