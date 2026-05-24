from amaterasu import *
import spacy
nlp = spacy.load("en_core_web_sm")
s = "The electrical problems of the present day lie largely in the economical transmission of power and in the radical improvement of the means and methods of illumination."
sentence = nlp(s)

with open(f'{path}tokens', 'rb') as f: tokens = load(f)
tokens = {tokens[i]:i for i in range(len(tokens))}
token_ids = [tokens[token.text] for token in sentence]
del tokens, sentence

rot = np.load(f'{path}rotation_matrix.npy')
sqrt11 = 3.3166247903554
eps = 1e-12
base_CTXT = np.identity(11)

# Generation of parameters

alpha = (np.random.random() - 0.5) * 1e-1
A = np.random.uniform(-0.5, 0.5, (11,11))
B = np.diag(np.random.uniform(-0.5, 0.5, 11)) + np.random.uniform(-5e-3, 5e-3, (11,11))

A = A * sqrt11 / np.linalg.norm(A, ord='fro')
B = B * sqrt11 / np.linalg.norm(B, ord='fro')

dynamics = compute_dynamics(token_ids, base_CTXT, rot, A, B, alpha=alpha)

E = np.linalg.norm(dynamics, axis=1)**2

# Proper entropy
p = E / (np.sum(E, axis=1, keepdims=True) + eps)
S = -np.sum(p * np.log(p + eps), axis=1)

np.save(f"{path}dynamics_A", A)
np.save(f"{path}dynamics_B", B)
np.save(f'{path}alpha', np.float64(alpha))
np.savetxt('energy.txt', E, delimiter=' ', fmt='%16.6E', header='Energies')
np.savetxt('entropy.txt', S, delimiter=' ', fmt='%16.6E', header='Entropy')
