from amaterasu import *
import re
import unicodedata
import spacy
import numpy as np
from pickle import dump
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
with open(f'{path}tokens', 'rb') as f: tokens = load(f)
tokens = {tokens[i]:i for i in range(len(tokens))}

nlp = spacy.load("en_core_web_sm")
books = ['Charles Darwin___A Monograph on the Sub-class Cirripedia (Volume 1 of 2).txt',\
            "Charles Darwin___A Naturalist's Voyage Round the World.txt",\
            'Charles Darwin___Coral Reefs.txt',\
            'Charles Darwin___Geological Observations On South America.txt',\
            'Charles Darwin___Insectivorous Plants.txt',\
            'Charles Darwin___On the Origin of Species 1st Edition.txt',\
            'Charles Darwin___On the Origin of Species 6th Edition.txt',\
            'Charles Darwin___On the Origin of Species by Means of Natural Selection or the Preservation of Favoured Races in the Struggle for Life. (2nd edition).txt',\
            'Charles Darwin___The Autobiography of Charles Darwin.txt',\
            'Charles Darwin___The Descent of Man and Selection in Relation to Sex.txt',\
            'Charles Darwin___The Descent of Man and Selection in Relation to Sex Volume I (1st edition).txt',\
            'Charles Darwin___The Descent of Man and Selection in Relation to Sex Volume II (1st Edition).txt',\
            'Charles Darwin___The Different Forms of Flowers on Plants of the Same Species.txt',\
            'Charles Darwin___The Effects of Cross & Self-Fertilisation in the Vegetable Kingdom.txt',\
            'Charles Darwin___The Expression of Emotion in Man and Animals.txt',\
            'Charles Darwin___The Formation of Vegetable Mould through the action of worms with observations of their habits.txt',\
            'Charles Darwin___The Movements and Habits of Climbing Plants.txt',\
            'Charles Darwin___The Variation of Animals and Plants under Domestication Volume II.txt',\
            'Charles Darwin___The Variation of Animals and Plants under Domestication Volume I.txt',\
            'Charles Darwin___Volcanic Islands.txt',\
            'Grant Allen___Charles Darwin.txt',\
            "Lewis Carroll___Alice's Adventures in Wonderland.txt",\
            "Lewis Carroll___Alice's Adventures Under Ground.txt",\
            'Lewis Carroll___A Tangled Tale.txt',\
            'Lewis Carroll___Eight or Nine Wise Words about Letter-Writing.txt',\
            'Lewis Carroll___Feeding the Mind.txt',\
            'Lewis Carroll___Phantasmagoria and Other Poems.txt',\
            'Lewis Carroll___Songs From Alice in Wonderland and Through the Looking-Glass.txt',\
            'Lewis Carroll___Sylvie and Bruno.txt',\
            'Lewis Carroll___Symbolic Logic.txt',\
            'Lewis Carroll___The Game of Logic.txt',\
            'Lewis Carroll___The Hunting of the Snark.txt',\
            'Lewis Carroll___Three Sunsets and Other Poems.txt',\
            'Lewis Carroll___Through the Looking-Glass.txt',\
            "Charles Dickens___A Child's History of England.txt",\
            'Charles Dickens___A Christmas Carol.txt',\
            'Charles Dickens___American Notes for General Circulation.txt',\
            'Charles Dickens___A Message from the Sea.txt',\
            'Charles Dickens___A Tale of Two Cities.txt',\
            'Charles Dickens___Barnaby Rudge.txt',\
            'Charles Dickens___Bleak House.txt',\
            'Charles Dickens___Captain Boldheart & the Latin-Grammar Master.txt',\
            "Charles Dickens___Charles Dickens' Children Stories.txt",\
            'Charles Dickens___David Copperfield.txt',\
            'Charles Dickens___Doctor Marigold.txt',\
            'Charles Dickens___Dombey and Son.txt',\
            "Charles Dickens___George Silverman's Explanation.txt",\
            'Charles Dickens___Going into Society.txt',\
            'Charles Dickens___Great Expectations.txt',\
            'Charles Dickens___Hard Times.txt',\
            'Charles Dickens___Holiday Romance.txt',\
            'Charles Dickens___Hunted Down.txt',\
            'Charles Dickens___Life And Adventures Of Martin Chuzzlewit.txt',\
            'Charles Dickens___Little Dorrit.txt',\
            "Charles Dickens___Master Humphrey's Clock.txt",\
            'Charles Dickens___Miscellaneous Papers.txt',\
            "Charles Dickens___Mrs. Lirriper's Legacy.txt",\
            "Charles Dickens___Mrs. Lirriper's Lodgings.txt",\
            'Charles Dickens___Mudfog and Other Sketches.txt',\
            'Charles Dickens___Mugby Junction.txt',\
            'Charles Dickens___Nicholas Nickleby.txt',\
            'Charles Dickens___Oliver Twist.txt',\
            'Charles Dickens___Our Mutual Friend.txt',\
            'Charles Dickens___Pictures from Italy.txt',\
            'Charles Dickens___Reprinted Pieces.txt',\
            'Charles Dickens___Sketches by Boz.txt',\
            'Charles Dickens___Sketches of Young Couples.txt',\
            'Charles Dickens___Sketches of Young Gentlemen.txt',\
            "Charles Dickens___Somebody's Luggage.txt",\
            'Charles Dickens___Some Christmas Stories.txt',\
            'Charles Dickens___Sunday Under Three Heads.txt',\
            'Charles Dickens___The Battle of Life.txt',\
            'Charles Dickens___The Chimes.txt',\
            'Charles Dickens___The Cricket on the Hearth.txt',\
            "Charles Dickens___The Haunted Man and the Ghost's Bargin.txt",\
            'Charles Dickens___The Holly-Tree.txt',\
            'Charles Dickens___The Lamplighter.txt',\
            'Charles Dickens___The Lazy Tour of Two Idle Apprentices.txt',\
            'Charles Dickens___The Letters of Charles Dickens Volume 1.txt',\
            'Charles Dickens___The Letters of Charles Dickens Volume 2.txt',\
            'Charles Dickens___The Letters of Charles Dickens Volume 3.txt',\
            'Charles Dickens___The Magic Fishbone.txt',\
            'Charles Dickens___The Mystery of Edwin Drood.txt',\
            'Charles Dickens___The Old Curiosity Shop.txt',\
            'Charles Dickens___The Perils of Certain English Prisoners.txt',\
            'Charles Dickens___The Pickwick Papers.txt',\
            'Charles Dickens___The Poems and Verses of Charles Dickens.txt',\
            'Charles Dickens___The Seven Poor Travellers.txt',\
            'Charles Dickens___The Trial of William Tinkling.txt',\
            'Charles Dickens___The Uncommercial Traveller.txt',\
            'Charles Dickens___The Wreck of the Golden Mary.txt',\
            'Charles Dickens___Three Ghost Stories.txt',\
            'Charles Dickens___To be Read at Dusk.txt',\
            "Charles Dickens___Tom Tiddler's Ground.txt",\
            "Bram Stoker___Dracula's Guest.txt",\
            'Bram Stoker___The Lady of the Shroud.txt',\
            'Bram Stoker___Dracula.txt',\
            'Bram Stoker___The Lair of the White Worm.txt',\
            'Bram Stoker___The Jewel of Seven Stars.txt',\
            'Bram Stoker___The Man.txt',\
            "Sir Arthur Conan Doyle___A Desert Drama.txt",\
            "Sir Arthur Conan Doyle___A Duet.txt",\
            "Sir Arthur Conan Doyle___A Study In Scarlet.txt",\
            "Sir Arthur Conan Doyle___A Visit to Three Fronts.txt",\
            "Sir Arthur Conan Doyle___Beyond the City.txt",\
            "Sir Arthur Conan Doyle___Danger! and Other Stories.txt",\
            "Sir Arthur Conan Doyle___His Last Bow.txt",\
            "Sir Arthur Conan Doyle___Memoirs of Sherlock Holmes.txt",\
            "Sir Arthur Conan Doyle___Micah Clarke.txt",\
            "Sir Arthur Conan Doyle___My Friend The Murderer.txt",\
            "Sir Arthur Conan Doyle___Rodney Stone.txt",\
            "Sir Arthur Conan Doyle___Round the Red Lamp.txt",\
            "Sir Arthur Conan Doyle___Sir Nigel.txt",\
            "Sir Arthur Conan Doyle___Songs of Action.txt",\
            "Sir Arthur Conan Doyle___Songs Of The Road.txt",\
            "Sir Arthur Conan Doyle___Tales of Terror and Mystery.txt",\
            "Sir Arthur Conan Doyle___The Adventure of the Bruce-Partington Plans.txt",\
            "Sir Arthur Conan Doyle___The Adventure of the Cardboard Box.txt",\
            "Sir Arthur Conan Doyle___The Adventure of the Devil's Foot.txt",\
            "Sir Arthur Conan Doyle___The Adventure of the Dying Detective.txt",\
            "Sir Arthur Conan Doyle___The Adventure of the Red Circle.txt",\
            "Sir Arthur Conan Doyle___The Adventure of Wisteria Lodge.txt",\
            "Sir Arthur Conan Doyle___The Adventures of Gerard.txt",\
            "Sir Arthur Conan Doyle___The Adventures of Sherlock Holmes.txt",\
            "Sir Arthur Conan Doyle___The Cabman's Story.txt",\
            "Sir Arthur Conan Doyle___The Captain of the Pole-Star and Other Tales.txt",\
            "Sir Arthur Conan Doyle___The Crime of the Congo.txt",\
            "Sir Arthur Conan Doyle___The Croxley Master.txt",\
            "Sir Arthur Conan Doyle___The Dealings of Captain Sharkey and Other Tales of Pirates.txt",\
            "Sir Arthur Conan Doyle___The Disappearance of Lady Frances Carfax.txt",\
            "Sir Arthur Conan Doyle___The Doings Of Raffles Haw.txt",\
            "Sir Arthur Conan Doyle___The Exploits Of Brigadier Gerard.txt",\
            "Sir Arthur Conan Doyle___The Firm of Girdlestone.txt",\
            "Sir Arthur Conan Doyle___The Great Boer War.txt",\
            "Sir Arthur Conan Doyle___The Great Keinplatz Experiment and Other Tales of Twilight and the Unseen.txt",\
            "Sir Arthur Conan Doyle___The Great Shadow and Other Napoleonic Tales.txt",\
            "Sir Arthur Conan Doyle___The Green Flag.txt",\
            "Sir Arthur Conan Doyle___The Guards Came Through and Other Poems.txt",\
            "Sir Arthur Conan Doyle___The Hound of the Baskervilles.txt",\
            "Sir Arthur Conan Doyle___The Last Galley.txt",\
            "Sir Arthur Conan Doyle___The Last of the Legions and Other Tales of Long Ago.txt",\
            "Sir Arthur Conan Doyle___The Lost World.txt",\
            "Sir Arthur Conan Doyle___The Man from Archangel.txt",\
            "Sir Arthur Conan Doyle___The Mystery of Cloomber.txt",\
            "Sir Arthur Conan Doyle___The New Revelation.txt",\
            "Sir Arthur Conan Doyle___The Parasite.txt",\
            "Sir Arthur Conan Doyle___The Poison Belt.txt",\
            "Sir Arthur Conan Doyle___The Refugees.txt",\
            "Sir Arthur Conan Doyle___The Return of Sherlock Holmes.txt",\
            "Sir Arthur Conan Doyle___The Sign of the Four.txt",\
            "Sir Arthur Conan Doyle___The Tragedy of The Korosko.txt",\
            "Sir Arthur Conan Doyle___The Valley of Fear.txt",\
            "Sir Arthur Conan Doyle___The Vital Message.txt",\
            "Sir Arthur Conan Doyle___The War in South Africa.txt",\
            "Sir Arthur Conan Doyle___The White Company.txt",\
            "Sir Arthur Conan Doyle___Through the Magic Door.txt",\
            "Sir Arthur Conan Doyle___Uncle Bernac.txt"]

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
