import spacy
import numpy as np
from pickle import load, dump
np.set_printoptions(linewidth=1000)
dim = 11
position_matrix = [[0 for j in range(dim)] for i in range(dim)]
counter =0
for i in range(dim):
    for j in range(i):
        position_matrix[i][j]=counter
        position_matrix[j][i]=counter
        counter+=1

def vector2algebra(vec):
    global dim, position_matrix
    return np.array([[vec[position_matrix[i][j]] if j<i else (0 if i==j else -vec[position_matrix[i][j]]) for j in range(dim)] for i in range(dim)])

def book_sentences(nlp, text):
    #nlp = spacy.load("en_core_web_sm") # we use small because we just wanna parse the text
    doc = nlp(text)
    return [sent.text.replace('\n', ' ').strip() for sent in doc.sents]

def iteration(CTXT_, R,A, B, alpha, noise=False, epsilon=2e-2):
    X = CTXT_ @ R
    Y = X + alpha*X @ A @ X.T @ X @B 
    if noise: Y += np.random.uniform(-epsilon, epsilon, (11, 11))
    F = np.linalg.norm(Y, ord='fro')
    return Y*3.3166247903554/F # the constant multiplying is the square root of 11

def vector_to_rotation(u): # maps vector to rotation matrix
    M_a = vector2algebra(u)
    val, vec = np.linalg.eig(M_a)
    val = val.tolist()
    D = np.diag([np.cos(v.imag)+np.sin(v.imag)*1j for v in val])
    vec_inv = np.linalg.inv(vec)
    return vec.dot(D.dot(vec_inv)).real

def compute_entropy(CTXT):
    E = np.linalg.norm(CTXT, axis=0)**2
    p = E / E.sum()  
    return -np.sum(p * np.log(p))

def bits_to_token(bits, encoding_to_token):
    enc = bits_to_encoding(bits)
    return encoding_to_token[enc]
def PCA_append(tf_input, CTXTs, CTXT):
    CTXTs[-1]=CTXT.flatten()
    X = np.concatenate(CTXTs)
    tf_input.append(X)
    return None
def tf_input_append(tf_input, CTXTs, vecs, CTXT, vec, projection_matrix, projection_mean):
    vecs.pop(0)
    vecs.append(vec)
    CTXTs[-1]=CTXT.flatten()
    X = (np.concatenate(CTXTs)-projection_mean) @ projection_matrix
    Y = np.concatenate(vecs)*20 # times 20 because the maximum was 0.05=1/20
    tf_input.append(np.concatenate([X, Y]).astype(np.float32))
    return None
def encoding_to_bits(x, nbits=48): return ((x >> np.arange(nbits)) & 1).astype(np.float32)
def bits_to_encoding(bits):
    x = (np.asarray(bits) > 0.5).astype(np.int64)
    return np.dot(x, 1 << np.arange(x.size, dtype=np.int64))

def compute_dynamics(token_ids, CTXT, rot, M, C, alpha):
    CTXTs = [CTXT.copy()]
    for idx in token_ids:
        R = rot[idx]
        CTXTs.append(iteration(CTXTs[-1], R, M, C, alpha=alpha))
    return np.array(CTXTs)
