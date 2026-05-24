from amaterasu import *
import spacy
import numpy as np
from pickle import dump
import os

np.random.seed(42)

nlp = spacy.load("en_core_web_lg")
encoding_size = 48
path = './path/'
os.makedirs(path, exist_ok=True)

# --- Load vectors ---
keys = list(nlp.vocab.vectors.keys())
tokens = [nlp.vocab.strings[k] for k in keys]
tokens.append('[EOS]') # EOS means end of sentence
X = np.array([nlp.vocab.vectors[k] for k in keys])

# --- Center ---
X = X - np.mean(X, axis=0)

# --- PCA ---
U, S, VT = np.linalg.svd(X, full_matrices=False)
X = X @ VT[3:58].T

# --- Median ---
median = np.median(X, axis=0)

encoding = np.zeros(X.shape[0], dtype=np.int64)
for j in range(encoding_size): encoding |= ((X[:, j] > median[j]).astype(np.int64) << j)
encoding = np.concatenate([encoding, [np.int64(0)]]) # EOS or last token from tokens list is zero
# --- Stats ---
unique_encoding = np.unique(encoding).shape[0]
print(f"Ratio of unique encodings over number of tokens: {100*unique_encoding / len(set([t.lower() for t in tokens])):02.2f}%")

# --- Scaling ---
norms = np.linalg.norm(X, axis=1)
SCALE = np.max(norms)
X *= 0.05 / SCALE
median *= 0.05 / SCALE

# --- Rotations ---
rots = np.array([vector_to_rotation(X[i]) for i in range(X.shape[0])])

# --- Save ---
np.save(f'{path}word_vector.npy', X)
np.save(f'{path}median.npy', median)
np.save(f'{path}rotation_matrix.npy', rots)
np.save(f'{path}encoding.npy', encoding)
np.save(f'{path}vector_embedding_singular_value.npy', S)
np.savetxt(f'{path}vector_embedding_singular_value.txt', S, delimiter=' ', fmt='%+10.4E', header='Singular values of spacy large model vector embeddings')
with open(f'{path}tokens.pkl', 'wb+') as f: dump(tokens, f)

