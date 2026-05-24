from amaterasu import *
from pickle import load
from random import shuffle
import numpy as np
path = './path/'
noise = False
prefix = "stochastic" if noise else "deterministic"
books = ['tokens_Charles Dickens___The Uncommercial Traveller', "tokens_Charles Darwin___A Naturalist's Voyage Round the World", 'tokens_Charles Dickens___David Copperfield', 'tokens_Charles Dickens___The Magic Fishbone', 'tokens_Grant Allen___Charles Darwin']

X = np.load(f'{path}word_vector.npy')
rots = np.load(f'{path}rotation_matrix.npy')

A, B = np.load(f"{path}dynamics_A.npy"), np.load(f"{path}dynamics_B.npy")
alpha = np.load(f"{path}alpha.npy")

with open(f'{path}encoding', 'rb') as f: encoding = load(f)
tf_input = list()
tf_output = list()
context_window = 10+1
for book in books:
    with open(f'{path}{book}', 'rb') as f: sentences = load(f)
    CTXTs = [np.zeros(121, dtype=np.float32) for j in range(context_window)]
    for sentence in sentences:
        if len(sentence)>2:
            CTXT = np.identity(11)
            PCA_append(tf_input, CTXTs, CTXT)
            for i in range(len(sentence)):
                R = rots[sentence[i]]
                CTXT = iteration(CTXT, R, A, B, alpha, noise=noise, epsilon=2e-2)
                PCA_append(tf_input, CTXTs, CTXT)
            CTXTs.pop(0)
            CTXTs.append(list())
shuffle(tf_input)
tf_input = np.array(tf_input[0:50000])
mean = np.mean(tf_input, axis=0)
tf_input = tf_input - mean
del X, rots, encoding, A, B, sentences, books
print('Started PCA')
U, S, VT = np.linalg.svd(tf_input, full_matrices=False)
print("saving files")
np.save(f'{path}{prefix}_PCA_S', S) 
np.save(f'{path}{prefix}_PCA_VT', VT)
np.save(f'{path}{prefix}_PCA_mean', mean)
