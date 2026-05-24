from amaterasu import *
from pickle import load, dump
import numpy as np
from random import shuffle
path = './path/'
noise = False
prefix = "stochastic" if noise else "deterministic"
books = ['tokens_Sir Arthur Conan Doyle___Memoirs of Sherlock Holmes',\
            'tokens_Lewis Carroll___Sylvie and Bruno',\
            'tokens_Charles Dickens___A Message from the Sea',\
            'tokens_Sir Arthur Conan Doyle___The Poison Belt',\
            'tokens_Sir Arthur Conan Doyle___The Adventures of Sherlock Holmes',\
            'tokens_Sir Arthur Conan Doyle___The Man from Archangel',\
            'tokens_Charles Dickens___Sketches of Young Gentlemen',\
            'tokens_Sir Arthur Conan Doyle___The Lost World',\
            'tokens_Charles Dickens___Barnaby Rudge',\
            'tokens_Sir Arthur Conan Doyle___The Adventure of Wisteria Lodge',\
            'tokens_Sir Arthur Conan Doyle___The Last Galley',\
            'tokens_Lewis Carroll___Three Sunsets and Other Poems',\
            'tokens_Sir Arthur Conan Doyle___The Adventure of the Bruce-Partington Plans',\
            'tokens_Sir Arthur Conan Doyle___My Friend The Murderer',\
            'tokens_Sir Arthur Conan Doyle___The Refugees',\
            'tokens_Sir Arthur Conan Doyle___The Great Keinplatz Experiment and Other Tales of Twilight and the Unseen',\
            'tokens_Charles Dickens___A Tale of Two Cities',\
            'tokens_Charles Dickens___Life And Adventures Of Martin Chuzzlewit',\
            'tokens_Charles Darwin___Geological Observations On South America',\
            'tokens_Charles Darwin___The Autobiography of Charles Darwin',\
            'tokens_Charles Darwin___The Formation of Vegetable Mould through the action of worms with observations of their habits',\
            'tokens_Sir Arthur Conan Doyle___The Disappearance of Lady Frances Carfax',\
            "tokens_Charles Dickens___Somebody's Luggage",\
            'tokens_Bram Stoker___The Man',\
            'tokens_Lewis Carroll___Feeding the Mind',\
            'tokens_Charles Dickens___Holiday Romance',\
            'tokens_Charles Darwin___The Variation of Animals and Plants under Domestication Volume II',\
            'tokens_Charles Dickens___Oliver Twist',\
            "tokens_Charles Dickens___Mrs. Lirriper's Lodgings",\
            'tokens_Charles Dickens___To be Read at Dusk',\
            "tokens_Charles Dickens___The Haunted Man and the Ghost's Bargin",\
            'tokens_Sir Arthur Conan Doyle___The War in South Africa',\
            'tokens_Charles Dickens___The Lazy Tour of Two Idle Apprentices',\
            'tokens_Sir Arthur Conan Doyle___The Dealings of Captain Sharkey and Other Tales of Pirates',\
            'tokens_Charles Dickens___American Notes for General Circulation',\
            'tokens_Charles Dickens___Great Expectations',\
            'tokens_Charles Dickens___Sketches by Boz',\
            'tokens_Charles Dickens___Captain Boldheart & the Latin-Grammar Master',\
            'tokens_Charles Dickens___Miscellaneous Papers',\
            'tokens_Charles Dickens___Dombey and Son',\
            'tokens_Sir Arthur Conan Doyle___The Parasite',\
            'tokens_Sir Arthur Conan Doyle___Uncle Bernac',\
            'tokens_Sir Arthur Conan Doyle___The New Revelation',\
            "tokens_Sir Arthur Conan Doyle___The Cabman's Story",\
            "tokens_Charles Dickens___George Silverman's Explanation",\
            'tokens_Charles Dickens___Hard Times',\
            'tokens_Charles Dickens___The Poems and Verses of Charles Dickens',\
            'tokens_Lewis Carroll___Phantasmagoria and Other Poems',\
            'tokens_Sir Arthur Conan Doyle___The Hound of the Baskervilles',\
            'tokens_Charles Dickens___The Old Curiosity Shop',\
            'tokens_Charles Dickens___Pictures from Italy',\
            'tokens_Charles Dickens___The Letters of Charles Dickens Volume 3',\
            'tokens_Bram Stoker___The Lair of the White Worm',\
            'tokens_Sir Arthur Conan Doyle___A Study In Scarlet',\
            'tokens_Charles Darwin___The Variation of Animals and Plants under Domestication Volume I',\
            'tokens_Sir Arthur Conan Doyle___A Visit to Three Fronts',\
            'tokens_Sir Arthur Conan Doyle___The Doings Of Raffles Haw',\
            'tokens_Sir Arthur Conan Doyle___His Last Bow',\
            'tokens_Charles Dickens___Mudfog and Other Sketches',\
            "tokens_Lewis Carroll___Alice's Adventures in Wonderland",\
            'tokens_Charles Dickens___Little Dorrit',\
            'tokens_Charles Darwin___Insectivorous Plants',\
            "tokens_Charles Dickens___Tom Tiddler's Ground",\
            'tokens_Charles Dickens___The Trial of William Tinkling',\
            'tokens_Lewis Carroll___Symbolic Logic',\
            'tokens_Sir Arthur Conan Doyle___Round the Red Lamp',\
            'tokens_Sir Arthur Conan Doyle___The Tragedy of The Korosko',\
            'tokens_Sir Arthur Conan Doyle___The Captain of the Pole-Star and Other Tales',\
            'tokens_Sir Arthur Conan Doyle___The Mystery of Cloomber',\
            'tokens_Sir Arthur Conan Doyle___The Return of Sherlock Holmes',\
            'tokens_Sir Arthur Conan Doyle___The Exploits Of Brigadier Gerard',\
            'tokens_Charles Dickens___A Christmas Carol',\
            'tokens_Charles Dickens___Bleak House',\
            'tokens_Charles Darwin___The Effects of Cross & Self-Fertilisation in the Vegetable Kingdom',\
            'tokens_Sir Arthur Conan Doyle___The Valley of Fear',\
            'tokens_Charles Dickens___Three Ghost Stories',\
            'tokens_Bram Stoker___The Lady of the Shroud',\
            'tokens_Sir Arthur Conan Doyle___The Guards Came Through and Other Poems',\
            'tokens_Sir Arthur Conan Doyle___The Adventures of Gerard',\
            'tokens_Charles Darwin___The Different Forms of Flowers on Plants of the Same Species',\
            'tokens_Charles Dickens___The Letters of Charles Dickens Volume 1',\
            'tokens_Sir Arthur Conan Doyle___The Last of the Legions and Other Tales of Long Ago',\
            'tokens_Sir Arthur Conan Doyle___Beyond the City',\
            "tokens_Charles Dickens___Master Humphrey's Clock",\
            'tokens_Lewis Carroll___A Tangled Tale',\
            "tokens_Sir Arthur Conan Doyle___The Adventure of the Devil's Foot",\
            'tokens_Lewis Carroll___Eight or Nine Wise Words about Letter-Writing',\
            'tokens_Charles Darwin___The Movements and Habits of Climbing Plants',\
            'tokens_Charles Dickens___Some Christmas Stories',\
            'tokens_Sir Arthur Conan Doyle___A Duet',\
            'tokens_Sir Arthur Conan Doyle___Songs Of The Road',\
            "tokens_Lewis Carroll___Alice's Adventures Under Ground",\
            'tokens_Charles Dickens___Our Mutual Friend',\
            'tokens_Charles Dickens___The Cricket on the Hearth',\
            "tokens_Charles Dickens___Charles Dickens' Children Stories",\
            'tokens_Charles Dickens___The Wreck of the Golden Mary',\
            'tokens_Lewis Carroll___The Game of Logic',\
            'tokens_Charles Dickens___Reprinted Pieces',\
            'tokens_Bram Stoker___Dracula',\
            'tokens_Charles Dickens___The Pickwick Papers',\
            'tokens_Charles Dickens___The Mystery of Edwin Drood',\
            'tokens_Charles Dickens___Sketches of Young Couples',\
            'tokens_Sir Arthur Conan Doyle___The Adventure of the Red Circle',\
            'tokens_Charles Dickens___The Holly-Tree',\
            'tokens_Charles Dickens___Mugby Junction',\
            'tokens_Sir Arthur Conan Doyle___The Crime of the Congo',\
            'tokens_Charles Dickens___The Chimes',\
            'tokens_Charles Dickens___Hunted Down',\
            'tokens_Lewis Carroll___Songs From Alice in Wonderland and Through the Looking-Glass',\
            'tokens_Charles Dickens___The Seven Poor Travellers',\
            'tokens_Charles Dickens___The Letters of Charles Dickens Volume 2',\
            'tokens_Sir Arthur Conan Doyle___Rodney Stone',\
            'tokens_Sir Arthur Conan Doyle___The Adventure of the Cardboard Box',\
            'tokens_Sir Arthur Conan Doyle___The White Company',\
            'tokens_Sir Arthur Conan Doyle___The Great Boer War',\
            "tokens_Charles Dickens___Mrs. Lirriper's Legacy",\
            'tokens_Lewis Carroll___Through the Looking-Glass',\
            'tokens_Sir Arthur Conan Doyle___Tales of Terror and Mystery',\
            'tokens_Sir Arthur Conan Doyle___The Firm of Girdlestone',\
            'tokens_Sir Arthur Conan Doyle___Songs of Action',\
            'tokens_Charles Dickens___Nicholas Nickleby',\
            'tokens_Charles Dickens___The Perils of Certain English Prisoners',\
            'tokens_Charles Darwin___A Monograph on the Sub-class Cirripedia (Volume 1 of 2)',\
            'tokens_Sir Arthur Conan Doyle___The Adventure of the Dying Detective',\
            "tokens_Charles Dickens___A Child's History of England",\
            'tokens_Sir Arthur Conan Doyle___The Vital Message',\
            'tokens_Sir Arthur Conan Doyle___The Green Flag',\
            'tokens_Charles Dickens___Going into Society',\
            'tokens_Sir Arthur Conan Doyle___Danger! and Other Stories',\
            'tokens_Sir Arthur Conan Doyle___Micah Clarke',\
            'tokens_Charles Darwin___Volcanic Islands',\
            'tokens_Bram Stoker___The Jewel of Seven Stars',\
            'tokens_Sir Arthur Conan Doyle___A Desert Drama',\
            'tokens_Charles Dickens___The Battle of Life',\
            'tokens_Sir Arthur Conan Doyle___Sir Nigel',\
            'tokens_Charles Dickens___Doctor Marigold',\
            'tokens_Sir Arthur Conan Doyle___The Great Shadow and Other Napoleonic Tales',\
            'tokens_Charles Dickens___The Lamplighter',\
            'tokens_Lewis Carroll___The Hunting of the Snark',\
            'tokens_Sir Arthur Conan Doyle___The Sign of the Four',\
            "tokens_Bram Stoker___Dracula's Guest",\
            'tokens_Sir Arthur Conan Doyle___Through the Magic Door',\
            'tokens_Sir Arthur Conan Doyle___The Croxley Master',\
            'tokens_Charles Dickens___Sunday Under Three Heads',\
            'tokens_Charles Darwin___The Expression of Emotion in Man and Animals',\
            'tokens_Charles Darwin___Coral Reefs']
books = list(books)
shuffle(books)
X = np.load(f'{path}word_vector.npy')
rots = np.load(f'{path}rotation_matrix.npy')
A = np.load(f'{path}dynamics_A.npy')
B = np.load(f'{path}dynamics_B.npy')
alpha = np.load(f'{path}alpha.npy')

projection_matrix = np.load(f'{path}{prefix}_projection_matrix.npy')
projection_mean = np.load(f'{path}{prefix}_PCA_mean.npy')
context_window = 10+1
with open(f'{path}encoding', 'rb') as f: encoding = load(f)
NN_input = list()
NN_output = list()
input_info = list()
for book_index in range(5):
    book = books[book_index]
    with open(f'{path}{book}', 'rb') as f: sentences = load(f)
    CTXTs = [np.zeros(121, dtype=np.float32) for j in range(context_window)]
    for k in range(len(sentences)):
        if len(sentences[k])>2:
            CTXT = np.identity(11)
            vec = np.zeros(55, dtype=np.float32)
            vecs = [np.zeros(55, dtype=np.float32) for j in range(4)]
            tf_input_append(NN_input, CTXTs, vecs, CTXT, vec, projection_matrix, projection_mean)
            NN_output.append(encoding[sentences[k][0]])
            input_info.append([-1, k, book_index]) # position, sentence, book

            for i in range(len(sentences[k])):
                vec = X[sentences[k][i]]
                R = rots[sentences[k][i]]
                CTXT = iteration(CTXT, R, A, B, alpha=alpha, noise=noise, epsilon=2e-2)
                tf_input_append(NN_input, CTXTs, vecs, CTXT, vec, projection_matrix, projection_mean)
                if i+1<len(sentences[k]): NN_output.append(encoding[sentences[k][i+1]])
                else: NN_output.append(np.int64(0))
                input_info.append([i, k, book_index])
            CTXTs.pop(0)
            CTXTs.append(np.zeros(121, dtype=np.float32))

input_info = np.array(input_info, dtype=np.int64)
NN_input = np.array(NN_input, dtype=np.float32)
NN_output = np.array(NN_output, dtype=np.int64)

order = np.arange(NN_input.shape[0])
np.random.shuffle(order)
order = order[:100]

NN_input = NN_input[order]
NN_output = NN_output[order]
input_info = input_info[order]
np.save(f'{path}{prefix}_NN_input', NN_input)
np.save(f'{path}{prefix}_NN_output', NN_output)

import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_lg")
with open(f'{path}tokens.pkl', 'rb') as f: tokens=load(f)
phrases = list()
for i in range(order.shape[0]):
    # I am aware that it could be sorted the books so that I don't have to load everything at every iteration
    # But the computational cost does not seem very big and therefore an optimized code was not performed
    book = books[input_info[i][2]]
    with open(f'{path}{book}', 'rb') as f: sentence = load(f)[input_info[i][1]][0:input_info[i][0]+1]
    sentence = [tokens[s] for s in sentence]
    doc = Doc(nlp.vocab, words=sentence)
    phrases.append(doc.text)
with open(f'{path}{prefix}_NN_phrases.pkl', 'wb+') as f: dump(phrases, f)
