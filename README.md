# Amaterasu: Next Token Prediction Through Codewords For Parameter Reduction

This is a next-token prediction task over a 56k vocabulary using a 2.1M-parameter language model that replaces standard vocabulary prediction with 48-bit token encoding.

## 1. Overview

The goal of this project, named **Amaterasu**, is to reduce the number of parameters required for language modeling so that useful models can run locally on low-cost hardware without the need for GPUs.

The proposed model has **2.1M parameters** (approximately 50× fewer than GPT-2 small) and achieves **22.08% top-1** and **29.92% top-10 accuracy** on in-domain next-token prediction.

Instead of predicting a probability distribution over the entire vocabulary, the model represents each token using a **48-bit encoding** in which individual bits carry semantic information. The network predicts each bit independently, reducing the output layer from hundreds of thousands of neurons to only 48.

To incorporate context, the model uses a **dynamical system** inspired by reservoir computing. This system encodes past tokens into **context matrices**, whose evolution captures information about the sequence. These matrices are then projected into a lower-dimensional space and used as input for prediction.

## 2. Results

It was evaluated the model under two configurations: deterministic and stochastic variants, using in-domain prediction accuracy with and without context ablations.

### Deterministic model

| Configuration     | Top-1 Accuracy | Top-10 Accuracy |
| ----------------- | -------------: | --------------: |
| No Ablation       |         22.08% |          29.92% |
| Mean Ablation     |         20.06% |          27.38% |
| Zero Substitution |         19.34% |          26.72% |
| Shuffle Ablation  |         18.54% |          25.88% |

### Stochastic model

| Configuration     | Top-1 Accuracy | Top-10 Accuracy |
| ----------------- | -------------: | --------------: |
| No Ablation       |         17.14% |          24.78% |
| Mean Ablation     |         16.60% |          24.00% |
| Zero Substitution |         16.60% |          23.90% |
| Shuffle Ablation  |         15.96% |          23.32% |

### Observations

A consistent degradation in performance is observed when the context matrices are perturbed or removed. In particular, shuffle ablation leads to a near-complete collapse of predictive performance, suggesting that the ordering and structure of the context representation is critical for correct decoding.

### Computational efficiency

Inference time was measured on a standard Kaggle CPU environment using a batch of 100 samples to benchmark the forward pass and decoding pipeline.

```
=== CPU Inference Benchmark ===

Total time (100 samples): 396.584 ms
Time per sample:          3.966 ms

--- CPU Info ---
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Model name:                              Intel(R) Xeon(R) CPU @ 2.20GHz
Thread(s) per core:                      2
Core(s) per socket:                      2
NUMA node0 CPU(s):                       0-3
Vectorization: avx, avx2
```

### Model size

```text
Total parameters: 2,189,736 (~8.35 MB)
```

## 3. Example Predictions

The following examples illustrate qualitative behaviors of the model, including semantic continuation, syntactic pattern learning, local linguistic regularities, and failure cases. A green check mark (✅) indicates the correct next-token prediction.

---

### Semantic Continuation and Phrasal Structure

The model learns common phrasal continuations and verb-particle constructions, assigning high probability to multiple plausible continuations while correctly identifying the target token.

```
Input: It was called , as I remember , " The Briton Conquers but to Save , " and he rolled it

Predictions:
up  -  close  -  opens  -  back  -  over  -  off  -  out ✅  -  tucked  -  behind  -  down
```

In this example, the model also assigns high probability to semantically and syntactically consistent alternatives (e.g., *up*, *back*, *over*), reflecting learned structure in verb–particle usage.

---

### Contextual Continuation in Natural Narrative Text

The model captures longer-range narrative flow and discourse continuity, even under noisy or highly descriptive input.

```
Input: The doors of the churches stand wide open ; and in this hot weather great red curtains flutter and wave in their palaces ; and if you go and sit in one of these to get out of the sun , you see the queerest figures kneeling against pillars , and the strangest people passing in and

Predictions:
up  -  slipt  -  out ✅  -  close  -  behind  -  away  -  over  -  down  -  off  -  floor
```

---

### Local Linguistic Regularities

The model successfully captures frequent function-word continuations and structural patterns in natural text.

```
Input: And so , if I were to go out of the record and read for your people , I should bring such a house about my ears as would shake " Little " out

Predictions:
of ✅  -  26  -  11:15  -  27  -  11:10  -  44  -  24th  -  .  -  28  -  12:45
```

```
Input: We can give you everything but a bed ( all ours are occupied in consequence of the boys being at home ) , and shall all be delighted to see

Predictions:
you ✅  -  yours  -  hesitate  -  link  -  please  -  to  -  recipient  -  appreciate  -  giving  -  always
```

---

### Failure Cases and Limitations

The architecture receives only the four most recent tokens directly as input. In this example, none of the last 4 tokens are related to body parts, yet the model assigns high probability to semantically related continuations such as *hands, arms, forehead, butt, mouth, brow and finger*.

```
Input: Wilson stood in the position from which he had derived his nickname , his left hand and left foot well to the front , his

Predictions:
hands  -  arms  -  sits  -  forehead  -  butt  -  mouth  -  locks  -  brow  -  finger  -  kneels
```

The model also shows sensitivity to vocabulary artifacts such as numeric or temporal tokens, which may receive non-negligible probability despite being irrelevant to the semantic context.

```
Input: Range of view and air , most free and delightful ; hill - side garden , delicious ; field , stupendous ; speculations in already effected by the undersigned , with the view to the keeping up

Predictions:
26  -  its  -  11:15  -  44  -  26th  -  the  -  5th  -  192  -  .  -  24th
```

## 4. Technical Details

### 4.1 Encodings and Rotation Matrices

Although the dimension of spacy large model is 300, the intrinsic dimension of the manifold is actually much lower. To reduce the dimension of the input for the neural network, it was applied PCA to the list of vector embeddings of all tokens in spacy large model in the following way

```python
# generate_files.py
# --- Load vectors ---
keys = list(nlp.vocab.vectors.keys())
tokens = [nlp.vocab.strings[k] for k in keys]
X = np.array([nlp.vocab.vectors[k] for k in keys])

# --- Center ---
X = X - np.mean(X, axis=0)

# --- PCA ---
U, S, VT = np.linalg.svd(X, full_matrices=False)
```

The first principal components of the embeddings are generally related to frequency instead of the semantics, so it was discarded the first 3 components and extracted the next 55 principal components (explanation for the number 55 later)

```python
# generate_files.py
X = X @ VT[3:58].T
```

The matrix X contains the vector embeddings used by the neural network.

The encoding of each token cannot be arbitrary because for the neural network to be able to predict each bit of the encoding, they must be each one related to the semantics of the token. The way it was approached to obtain a useful encoding is by extracting the median of each principal component axis for all embeddings

```python
# generate_files.py
median = np.median(X, axis=0)
```

and for each principal component, if the embedding is above the median then it is assigned 1 for the bit corresponding to that axis and 0 otherwise. The following shows the implementation of that

```python
# generate_files.py
encoding = np.zeros(X.shape[0], dtype=np.int64)
for j in range(encoding_size): encoding |= ((X[:, j] > median[j]).astype(np.int64) << j)
```

Although the dimension of the vector embedding is 55, it was chosen the greatest multiple of 8 which is less than 55 to be the dimension of the encoding or 48 since more bits does not translate into better prediction and being a multiple of 8 can be useful for writing to a binary file a sequence of codewords more easily. It was also tested the probability of a collision between token encodings to not occur the following way

```python
# generate_files.py
unique_encoding = np.unique(encoding).shape[0]
print(f"Ratio of unique encodings over number of tokens: {100*unique_encoding / len(set([t.lower() for t in tokens])):02.2f}%")
```

where the probability obtained was 99.62% therefore a collision probability of 0.38% chance which shows that the choice of 48 bits has already low collision rate.

The architecture encodes contextual and semantic information through the evolution of a matrix referred to as the *context matrix*. The update rule of the dynamical system depends on successive multiplications by token-dependent rotation matrices.

As the rotation angle increases, the trajectory followed by the context matrix becomes progressively more irregular and unstable, making it harder for the system to preserve coherent long-range structure. To maintain smooth dynamics, each token rotation matrix was therefore constrained to have a maximum rotation angle of 0.05 radians (approximately 2.86 degrees).

The way it was mapped each token to their corresponding rotation matrix is by mapping each vector embedding to an element of the algebra of SO(11) and then exponentiating using the identity

```tex
e^{ix}=\cos(x)+i \sin (x)
```

The implementation being the following

```python
# amaterasu.py
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

...

def vector_to_rotation(u): # maps vector to rotation matrix
    M_a = vector2algebra(u)
    val, vec = np.linalg.eig(M_a)
    val = val.tolist()
    D = np.diag([np.cos(v.imag)+np.sin(v.imag)*1j for v in val])
    vec_inv = np.linalg.inv(vec)
    return vec.dot(D.dot(vec_inv)).real
```

The dimension of SO(11) is

```tex
\frac{10\times 11}{2}=55
```

so that's why it was chosen the vector embedding to be of dimension exactly 55 so that it can be mapped to a rotation matrix of dimension 11x11. The implementation of what I described being

```python
# generate_files.py
# --- Scaling ---
norms = np.linalg.norm(X, axis=1)
SCALE = np.max(norms)
X *= 0.05 / SCALE
median *= 0.05 / SCALE

# --- Rotations ---
rots = np.array([vector_to_rotation(X[i]) for i in range(X.shape[0])])
```

Everything is later saved at the end

```python
# generate_files.py
# --- Save ---
np.save(f'{path}word_vector.npy', X)
np.save(f'{path}median.npy', median)
np.save(f'{path}rotation_matrix.npy', rots)
np.save(f'{path}encoding.npy', encoding)
np.save(f'{path}vector_embedding_singular_value.npy', S)
np.savetxt(f'{path}vector_embedding_singular_value.txt', S, delimiter=' ', fmt='%+10.4E', header='Singular values of spacy large model vector embeddings')
with open(f'{path}tokens.pkl', 'wb+') as f: dump(tokens, f)
```

### 4.2 Processing Of The Text Files

A preprocessing stage was introduced to simplify subsequent stages of the pipeline and improve computational efficiency. The preprocessing operates on the cleaned text files of project Gutenberg.

During preprocessing, each text is segmented into sequences of sentences and every token is converted into its corresponding vocabulary index. The resulting sequences of token indices are then stored as binary pickle files. For each original text file, a corresponding tokenized file is generated with the naming convention `"tokens_" + original_filename`.

```python
# text2token.py
with open(f'{path}tokens', 'rb') as f: tokens = load(f)
tokens = {tokens[i]:i for i in range(len(tokens))}

nlp = spacy.load("en_core_web_sm")
books = ['Charles Darwin___A Monograph on the Sub-class Cirripedia (Volume 1 of 2).txt',\
            "Charles Darwin___A Naturalist's Voyage Round the World.txt",\
            'Charles Darwin___Coral Reefs.txt',\
...
]
for book in books:
    with open(f"{path}Gutenberg/txt/{book}", 'r') as f: paragraphs=f.read().split('\n\n')
    phrases = list()
    for p in paragraphs:
        sentences = book_sentences(nlp, clean_text(p))
        sentences = [nlp(s) for s in sentences]
        for sentence in sentences:
            phrase = list()
            for token in sentence:
                if tokens.get(token.text, None)==None:
                    ll = token.text.lower()
                    if tokens.get(ll, None)!=None:
                        phrase.append(tokens[ll])
                else: phrase.append(tokens[token.text])
            phrase=np.array(phrase, dtype=np.int32)
            phrases.append(phrase)
    with open(f"{path}tokens_{book.replace('.txt', '')}", 'wb+') as f: dump(phrases, f)

```

During the preprocessing it was used a function to remove certain sequences of characters that may hinder the performance of the model which was the following

```python
# text2token.py
def clean_text(text):

    # normalize unicode (important for old typography)
    text = unicodedata.normalize("NFKC", text)

    # normalize quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
            
    # normalize dashes
    text = text.replace("—", " ").replace("–", " ")

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # remove isolated punctuation tokens
    text = re.sub(r"\s+[^\w\s]\s+", " ", text)

    return text.strip()

```

The preprocessing also relies on the function that removes line breaks, which is the following

```python
#amaterasu.py
def book_sentences(nlp, text):
    #nlp = spacy.load("en_core_web_sm") # we use small because we just wanna parse the text
    doc = nlp(text)
    return [sent.text.replace('\n', ' ').strip() for sent in doc.sents]
```

The following was done inside python shell to test if the preprocessing worked as intended

```python
>>> from pickle import load
>>> with open ("tokens_Sir Arthur Conan Doyle___The Refugees", 'rb') as f: sentences = load(f)
... 
>>> with open ("tokens.pkl", 'rb') as f: tokens= load(f)
... 
>>> sentences[100]
array([  2072,   1973,    101,  12226,      6,   1318,      1,      6,
          115,   1097,     32,    504,    806,      6,    206,   1201,
        23585,   5189,      5, 202999,   4994,      4,    115,  31332,
         6918,    803,      6,    115,  22458,   1049,      0],
      dtype=int32)
>>> ' '.join([tokens[x] for x in sentences[100]])
'Her features were delicate and sweet , and her blue - black hair and long dark eyelashes formed a piquant contrast to her dreamy gray eyes and her ivory skin .'
>>> ' '.join([tokens[x] for x in sentences[101]])
'In her whole expression there was something quiet and subdued , which was accentuated by her simple dress of black taffeta , and by the little jet brooch and bracelet which were her sole ornaments .'
>>> ' '.join([tokens[x] for x in sentences[102]])
'Such was Adele , the only daughter of the famous Huguenot cloth - merchant .'
```

### 4.3 Context Matrices and Context Vectors

It was encoded the semantics or context of a sentence by updating with each token the context matrix by the following rule 

```python
# amaterasu.py
def iteration(CTXT_, R,A, B, alpha, noise=False, epsilon=2e-2):
    X = CTXT_ @ R
    Y = X + alpha*X @ A @ X.T @ X @B # chatgpt suggested tweking the non-linearity to tanh(X A X^T) X B
    if noise: Y += np.random.uniform(-epsilon, epsilon, (11, 11))
    F = np.linalg.norm(Y, ord='fro')
    return Y*3.3166247903554/F # the constant multiplying is the square root of 11
```

where \(R\) denotes the rotation-matrix embedding associated with a given token, \(A\), \(B\), and \(\alpha\) are fixed parameters, and \(\mathrm{CTXT}_n\) represents the context matrix at step \(n\).

When the `noise` parameter is set to `True`, the dynamical system follows a stochastic update rule, while setting it to `False` yields the deterministic version of the model. The introduced nonlinearity is intended to reduce the effective entropy of the dynamical system, making the resulting representations more interpretable by the neural network. Further details regarding the update rule and its properties are provided in the preprint.

The final context matrix obtained after processing the last token of a sentence is defined as the context matrix associated with that sentence.

It was generated the parameters of the dynamical system of the context matrices in the following way

```python
# energy_entropy.py
alpha = (np.random.random() - 0.5) * 1e-1
A = np.random.uniform(-0.5, 0.5, (11,11))
B = np.diag(np.random.uniform(-0.5, 0.5, 11)) + np.random.uniform(-5e-3, 5e-3, (11,11))

A = A * sqrt11 / np.linalg.norm(A, ord='fro')
B = B * sqrt11 / np.linalg.norm(B, ord='fro')
```

On file `energy_entropy.py` it is also computed energy and entropy for the phrase *"The electrical problems of the present day lie largely in the economical transmission of power and in the radical improvement of the means and methods of illumination."* to illustrate the behavior of the nonlinear system.

To provide long-range contextual information for prediction, the neural network receives as input the context matrices of the 10 previous sentences together with the context matrix currently being updated during next-token prediction.

Although this representation formally lies in a space of dimension 1331, similarly to vector embeddings, the trajectories followed by sequences of context matrices appear to concentrate near a substantially lower-dimensional subspace. To obtain a compact representation, a PCA analysis was performed on 50,000 sequences of context matrices in order to construct a projection matrix.

By examining the singular value spectrum and identifying the final significant flattening region, it was decided to retain the first 130 principal components for the deterministic model and 170 for the stochastic model.

To avoid data leakage, the books used to compute the PCA projection matrix were excluded from the books used to generate the training and test datasets.

The PCA being implemented in the following way

```python
# PCA_training.py
from amaterasu import *
from pickle import load
from random import shuffle
import numpy as np
path = '/media/atila/C117-51DB/books/'
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
```

Where it was used the function

```python
# amaterasu
def PCA_append(tf_input, CTXTs, CTXT):
    CTXTs[-1]=CTXT.flatten()
    X = np.concatenate(CTXTs)
    tf_input.append(X)
    return None
```

With the projection matrix extracted from the result in the following way

```python
# projection_matrix.py
import numpy as np
path = '/media/atila/C117-51DB/books/'
noise = False
prefix = "stochastic" if noise else "deterministic"
dimension = 170 if noise else 130
S = np.load(f'{path}{prefix}_PCA_S.npy')
VT = np.load(f'{path}{prefix}_PCA_VT.npy')
projection_matrix = VT.T[:, :dimension]/S[:dimension]
np.save(f'{path}{prefix}_projection_matrix.npy', projection_matrix)
```

The of the sequence of context matrices on the lower dimensional subspace is given the name of context vector.

### 4.4 Generation Of Training/Test Dataset

To generate training as well as test dataset, the algorithm starts from a list of books that were not used during the stage where it was generated the projection matrices and shuffles it using function `shuffle` from library `random`

```python
# generate_training_data.py
books = ['tokens_Sir Arthur Conan Doyle___Memoirs of Sherlock Holmes',\
            'tokens_Lewis Carroll___Sylvie and Bruno',\
            'tokens_Charles Dickens___A Message from the Sea',\
            'tokens_Sir Arthur Conan Doyle___The Poison Belt',\
            'tokens_Sir Arthur Conan Doyle___The Adventures of Sherlock Holmes',\
...
]
shuffle(books)
```

It then loads matrices that it will need for the dataset generation

```python
# generate_training_data.py
X = np.load(f'{path}word_vector.npy')
rots = np.load(f'{path}rotation_matrix.npy')
A = np.load(f'{path}dynamics_A.npy')
B = np.load(f'{path}dynamics_B.npy')
alpha = np.load(f'{path}alpha.npy')
            
projection_matrix = np.load(f'{path}{prefix}_projection_matrix.npy')
projection_mean = np.load(f'{path}{prefix}_PCA_mean.npy')
context_window = 10+1
with open(f'{path}encoding', 'rb') as f: encoding = load(f)
```

The pipeline then enters a loop in which 5 books are loaded at each iteration. This procedure ensures that each generated dataset contains shuffled sentences originating from multiple books, increasing the diversity of the training data and reducing the likelihood of overfitting.

Empirically, it was observed that processing 13 groups of books produced approximately 7–8 GB of training data on average, which was considered sufficient for a model containing roughly 2 million parameters.

```python
# generate_training_data.py
for book_i in range(0, 65, 5): # only 13 datasets of at most 500,000 elements (13*5=65)
    tf_input = list()
    tf_output = list()

    for book_j in range(5):
        book = books[book_i+book_j]
        print(book_i//5, book)
        with open(f'{path}{book}', 'rb') as f: sentences = load(f)
```

The variables `tf_input` and `tf_output` store the input features and target outputs required for training the TensorFlow model.

At initialization, the sequence of past context matrices is set to zero matrices, since no prior sentences are available at the beginning of a book. For each sentence, the model also initializes four previous token embeddings as zero vectors, representing the absence of preceding context. The context matrix is initialized as the identity matrix.

Before processing each sentence, an input sample is constructed that contains only the current context matrix, while the embeddings corresponding to previous tokens are set to zero. The corresponding target output is the first token of the sentence. This allows the network to learn to predict the initial token of a sentence conditioned solely on the context matrix.

For each token in the sequence, the model updates the context matrix and constructs the corresponding context vector. The list of previous token embeddings is then updated by removing the oldest element and appending the embedding of the current token.

The variable `tf_output` is populated at each step with the encoding of the next token to be predicted. At the end of a sentence, where no next token exists, a special value of 0 is used to represent the end-of-sentence token.

The procedure described above is implemented as follows:

```python
# generate_training_data.py
		CTXTs = [np.zeros(121, dtype=np.float32) for j in range(context_window)]
        for sentence in sentences:
            if len(sentence)>2:
                CTXT = np.identity(11)
                vec = np.zeros(55, dtype=np.float32)
                vecs = [np.zeros(55, dtype=np.float32) for j in range(4)]
                tf_input_append(tf_input, CTXTs, vecs, CTXT, vec, projection_matrix, projection_mean)
                tf_output.append(encoding[sentence[0]])
                for i in range(len(sentence)):
                    vec = X[sentence[i]]
                    R = rots[sentence[i]]
                    CTXT = iteration(CTXT, R, A, B, alpha=alpha, noise=noise, epsilon=2e-2)
                    tf_input_append(tf_input, CTXTs, vecs, CTXT, vec, projection_matrix, projection_mean)
                    if i+1<len(sentence): tf_output.append(encoding[sentence[i+1]])
                    else: tf_output.append(np.int64(0))
                CTXTs.pop(0)
                CTXTs.append(np.zeros(121, dtype=np.float32))
```

Where the functions that were used in this part were the following

```python
# amaterasu.py
def iteration(CTXT_, R,A, B, alpha, noise=False, epsilon=2e-2):
    X = CTXT_ @ R
    Y = X + alpha*X @ A @ X.T @ X @B
    if noise: Y += np.random.uniform(-epsilon, epsilon, (11, 11))
    F = np.linalg.norm(Y, ord='fro')
    return Y*3.3166247903554/F # the constant multiplying is the square root of 11
...
def tf_input_append(tf_input, CTXTs, vecs, CTXT, vec, projection_matrix, projection_mean):
    vecs.pop(0)
    vecs.append(vec)
    CTXTs[-1]=CTXT.flatten()
    X = (np.concatenate(CTXTs)-projection_mean) @ projection_matrix
    Y = np.concatenate(vecs)*20 # times 20 because the maximum was 0.05=1/20
    tf_input.append(np.concatenate([X, Y]).astype(np.float32))
    return None
```

Then the tf_input and tf_output lists are converted to numpy arrays as no new elements are going to be appended to them at this stage. It will be also be generated an order that input and output will be shuffled

```python
# generate_training_data.py
    tf_input = np.array(tf_input, dtype=np.float32)
    tf_output = np.array(tf_output, dtype=np.int64)
    order = np.arange(len(tf_input))
    np.random.shuffle(order)
```

The first 500,000 elements of the input/output lists are used to construct each training dataset.

During the first iteration, a separate dataset for test/validation is also created such that it does not overlap with the training data, ensuring that evaluation metrics are not artificially inflated due to data leakage. In the standard configuration, the first 500,000 elements are used for training, while the subsequent 10,000 elements are reserved for test/validation. If the total number of available elements is smaller than 500,000, the last 10,000 elements are instead used as the test/validation set, and the remaining elements are used for training. In all cases, training and test/validation datasets remain strictly disjoint.

```python
# generate_training_data.py
    LIST_SIZE = 500000
    if book_i==0:
        TEST_VAL_SIZE = 10000
        LIST_SIZE = min(tf_input.shape[0]-TEST_VAL_SIZE, LIST_SIZE)
        test_val_order = order[LIST_SIZE:LIST_SIZE+TEST_VAL_SIZE]
        test_val_input = tf_input[test_val_order]
        test_val_output = tf_output[test_val_order]
        np.save(f'{path}{prefix}_input', test_val_input) # used for test and validation dataset
        np.save(f'{path}{prefix}_output', test_val_output)

    order = order[:LIST_SIZE]
    tf_input = tf_input[order]
    tf_output = tf_output[order]
    np.save(f'{path}{prefix}_tf_input_{book_i//5:04d}', tf_input)
    np.save(f'{path}{prefix}_tf_output_{book_i//5:04d}', tf_output)
```

 As it was said at the beginning as well as on the preprint, it was evaluated merely in-domain prediction so it was not used a separate set of books to generate test dataset as analyzing capabilities of generalization by the model seem too ambitious for a first work on this line of thought.

### 4.5 Generation Of The Vocabulary

To limit the predictions to only what the model was trained, it was generated a vocabulary containing encodings of tokens that were used at least once by one of the books that the model was trained on in the following way

```python
# vocabulary.py
books = list(listdir(f'{path}'))
books = [book for book in books if 'tokens_' in book]
vocabulary = set()
for book in books:
    print(book)
    with open(f'{path}{book}', 'rb') as f: sentences = load(f)
    sentences = np.concatenate(sentences)
    sentences = np.unique(sentences)
    vocabulary = vocabulary.union(set(sentences))
vocabulary = np.array(list(vocabulary))
vocabulary_encoding = np.load(f'{path}encoding.npy')
vocabulary_encoding = vocabulary_encoding[vocabulary]
vocabulary_encoding = np.unique(vocabulary_encoding)
np.save(f'{path}vocabulary_encoding.npy', vocabulary_encoding)
```

The vocabulary size obtained was of the order of 56k encodings.

## 4.6 Architecture

To make it simpler, in this section I will alternate between giving a short justification of each choice of component in the architecture followed by the actual tensorflow code. I will be describing the best performing model which was the deterministic model, but the architecture for the stochastic model is similar in all aspects with the only difference being the input dimension which would be 390 instead of 350. Have in mind that some of the layers have number of neurons equal to the input dimension and that as well is modified for the stochastic model.

```py
inputs = tf.keras.Input(shape=(350,))
```

To increase training speed, the first layer multiplies each entry of the input by a learned scale so that the neural network learns to balance the contribution of each entry more quickly.

```py
x = DimensionScaling(350)(inputs)

```
Then to model interactions between input features, the neural network starts with 2 layers, each with number of neurons equal to input dimension.

```py
x = tf.keras.layers.Dense(350, activation='relu')(x)
x = tf.keras.layers.Dense(350, activation='relu')(x)
```

Then it passes the output by one dense layer that is then normalized and activated by GeLU. Layer normalization was utilized so that the output is within the range where the action of GeLU is meaningful.

```py
x = tf.keras.layers.Dense(350, activation=None)(x)
x = tf.keras.layers.LayerNormalization()(x)
x = tf.keras.layers.Activation('gelu')(x)
```

Then the output is mapped into a sequence of 256 vectors of features of 20 dimensions each. 

```py
x = tf.keras.layers.Dense(20<<8, activation='relu')(x) # 256 of dimension 20
```

The features are combined hierarchically in a binary tree structure across successive hidden layers. At each stage, pairs of feature vectors are concatenated and passed through a dense layer with 40 units and ReLU activation. The resulting representation is then mapped to a set of candidate features using a dense layer with 20 units. A gating mechanism is applied to selectively retain or suppress components of these candidate features. The gated output is subsequently passed through an additional ReLU-activated layer before proceeding to the next level of the hierarchy.

```py
for N in range(7, 1, -1):
    x = tf.keras.layers.Reshape((1<<N, 2, 20))(x)
    x1 = x[:, :, 0, :]   # (batch, groups, 20)
    x2 = x[:, :, 1, :]   # (batch, groups, 20)
    
    concat = tf.keras.layers.Concatenate()([x1, x2])

    h = tf.keras.layers.Dense(40, activation='relu')(concat)

    gate = tf.keras.layers.Dense(20, activation='sigmoid')(h)
    candidate = tf.keras.layers.Dense(20)(h)

    merged = gate * candidate

    x = tf.keras.layers.Activation('relu')(merged)
```
To not map to a set of vectors with dimension lower than the output dimension, which may compromise the model, it was added a dense layer of 24 units so that the two feature vectors when concatenated have a total of $2times 24=48$ dimensions.

```py
x = tf.keras.layers.Reshape((2,40))(x)
x = tf.keras.layers.Dense(24, activation='relu')(x) # 2 of dimension 24 instead of 20
```

Then it is concatenated those vectors and passed through the final layer with sigmoid activation to obtain the probability of each of the 48 bits of the token codeword

```py
x = tf.keras.layers.Reshape((48,))(x)
outputs = tf.keras.layers.Dense(48, activation='sigmoid')(x)# output layer
model = tf.keras.Model(inputs, outputs)
```

## 4.7 Generating Vocabulary

To restrict predictions to tokens observed during training, the file `vocabulary_encoding.npy` stores the codeword encodings of all tokens appearing in at least one training document.

This vocabulary is constructed by first concatenating all token index sequences extracted from the training sentences. The resulting array is then passed through `np.unique` to obtain the set of distinct token indices, which defines the model’s vocabulary. Finally, the set of unique indices is converted into a NumPy array, which is used to index and extract the corresponding token codeword encodings.

```python
# vocabulary.py
books = list(listdir(f'{path}'))
books = [book for book in books if 'tokens_' in book]
vocabulary = set()
for book in books:
    print(book)
    with open(f'{path}{book}', 'rb') as f: sentences = load(f)
    sentences = np.concatenate(sentences)
    sentences = np.unique(sentences)
    vocabulary = vocabulary.union(set(sentences))
vocabulary = np.array(list(vocabulary))
vocabulary_encoding = np.load(f'{path}encoding.npy')
vocabulary_encoding = vocabulary_encoding[vocabulary]
vocabulary_encoding = np.unique(vocabulary_encoding)
np.save(f'{path}vocabulary_encoding.npy', vocabulary_encoding)
```

### 4.8 Heuristic For Estimating Most Likely Tokens

By leveraging the predicted probabilities of individual bits, a scoring function can be constructed over the token codewords in the vocabulary that reflects their compatibility with the model’s output distribution. Sorting token encodings according to this score allows retrieval of both top-1 and top-k predictions.

More specifically, for a given codeword \( b \), and bitwise probabilities \( P(b_i \mid x, i) \) produced by the network for input \( x \), the score is defined as:

```tex
S(b) = \sum_i \alpha_i \log P(b_i \mid x, i)
```

where \(\alpha_i\) is a weighting factor defined as the empirical accuracy of the \( i\)-th bit minus \(0.5\). This choice ensures that bits performing at random level (accuracy 0.5) contribute zero weight to the score.

The bitwise accuracies are computed over the full training dataset to avoid any possibility of data leakage.

A more detailed derivation of this scoring function is provided in the accompanying preprint. In practice, the formulation enables an efficient vectorized implementation using NumPy, allowing batch computation of scores for all vocabulary tokens. In the Kaggle notebook, this computation is implemented as follows:

```python
        preds = model.predict(X, verbose=0)
        preds = np.clip(preds, eps, 1 - eps)
        
        logit = np.log(preds) - np.log(1 - preds)   # (N_preds, bits)
        bias  = np.log(1 - preds)                  # (N_preds, bits)
     
        # apply bit weights
        logit *= alpha
        bias  *= alpha
        
        # main computation
        scores = logit @ encoding_bits.T + bias.sum(axis=1, keepdims=True)
        result = np.argsort(-scores, axis=1)
        result = result[:,:10]
```

which was how it calculated the score (saved on `scores` variable) as well as the expected encodings (saved on `result`).

## 5. Kaggle Links

The training of the model due to hardware limitations had to be done remotely. The iron python notebooks were downloaded and uploaded to github as well as their html versions. The links to the scripts that ran on kaggle are the following

- *Kaggle Link 1*
- *Kaggle Link 2*
- *Kaggle Link 3*

**I WILL CHANGE THIS SECTION ONCE I UPLOAD TO KAGGLE**



## 6. Preprint link

It was also made a preprint available here *(Link for The Preprint Once Published)* where you can find a more in depth explanation of how the model was developed. 

## 7. Support / Donation

If you find this work interesting or helpful, consider supporting its development:

👉 [Buy Me a Coffee](https://buymeacoffee.com/loboatilax)

Any support is greatly appreciated and helps keep the project active.



**Also, I appreciate anyone who volunteers to endorse my publication on Arxiv as I wish to publish there but am not able to do so currently.**
