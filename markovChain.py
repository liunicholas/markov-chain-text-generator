# import markovify
#
# # Get raw text as string.
# with open("text/markovText.txt") as f:
#     text = f.read()
#
# # Build the model.
# text_model = markovify.Text(text)
#
# # Print five randomly-generated sentences
# for i in range(5):
#     print(text_model.make_sentence())

import random
from random import random
import numpy as np
from scipy.sparse import dok_matrix

corpus = ""
with open("text/markovText.txt", 'r') as f:
        corpus+=f.read()
corpus = corpus.replace('\n',' ')
corpus = corpus.replace('\t',' ')
corpus = corpus.replace('“', ' " ')
corpus = corpus.replace('”', ' " ')
for spaced in ['.','-',',','!','?','(','—',')']:
    corpus = corpus.replace(spaced, ' {0} '.format(spaced))

corpus_words = corpus.split(' ')
corpus_words= [word for word in corpus_words if word != '']

distinct_words = list(set(corpus_words))
word_idx_dict = {word: i for i, word in enumerate(distinct_words)}

k = 3 # adjustable
sets_of_k_words = [ ' '.join(corpus_words[i:i+k]) for i, _ in enumerate(corpus_words[:-k]) ]

sets_count = len(list(set(sets_of_k_words)))
next_after_k_words_matrix = dok_matrix((sets_count, len(distinct_words)))

distinct_sets_of_k_words = list(set(sets_of_k_words))
k_words_idx_dict = {word: i for i, word in enumerate(distinct_sets_of_k_words)}

for i, word in enumerate(sets_of_k_words[:-k]):

    word_sequence_idx = k_words_idx_dict[word]
    next_word_idx = word_idx_dict[corpus_words[i+k]]
    next_after_k_words_matrix[word_sequence_idx, next_word_idx] +=1

def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects',
        the likelihood of the objects is weighted according
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]

def sample_next_word_after_sequence(word_sequence):
    next_word_vector = next_after_k_words_matrix[k_words_idx_dict[word_sequence]]
    likelihoods = next_word_vector/next_word_vector.sum()

    return weighted_choice(distinct_words, likelihoods.toarray())

def stochastic_chain(seed, chain_length=100, seed_length=3):
    current_words = seed.split(' ')
    if len(current_words) != seed_length:
        raise ValueError(f'wrong number of words, expected {seed_length}')
    sentence = seed

    for _ in range(chain_length):
        sentence+=' '
        next_word = sample_next_word_after_sequence(' '.join(current_words))
        sentence+=next_word
        current_words = current_words[1:]+[next_word]
    return sentence
# example use
print(stochastic_chain('STUDENT has made'))
