import nltk
nltk.download('brown')
from collections import defaultdict
import numpy as np
import nltk.corpus
from nltk.corpus import brown

import numpy as np

#########################################################################
# Exercise 1: Create a simple trigram language model and use the model
#########################################################################

""" we´ll save the counts directly in a dictionary
 which defaults to the smoothing factor (note that this is not true smoothing
 as it does not account for the denominator and therefore does not create a
 true probability distribution, but it is enough to get started)
"""
smoothing_factor = .1
counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))


""" We'll also define two helper functions, one to get the log probability of
a single trigram and the second to get the log probability of a full sentence
"""


def logP(u, v, w):
    return np.log(counts[(u, v)][w]) - np.log(sum(counts[(u, v)].values()))


def sentence_logP(S):
    """
    Adds the special tokens to the beginning and end and splits the sentence
    into tokens on whitespace.
    """
    tokens = ['*', '*'] + S + ['STOP']
    return sum([logP(u, v, w) for u, v, w in nltk.ngrams(tokens, 3)])

# we then choose the corpus and sentence split and tokenize the text using spacy
# we'll start with text from the 'news' domain

sentences = brown.sents(categories='news')
test_idx = int(len(sentences) * .7)
train = sentences[:test_idx]
test = sentences[test_idx:]

# Next we'll collect the counts
for sentence in train:
    # add the special tokens to the sentences
    tokens = ['*', '*'] + sentence + ['STOP']
    for u, v, w in nltk.ngrams(tokens, 3):
        # update the counts
        counts[(u, v)][w] += 1

# Now that we have the model we can use it
print(sentence_logP("what is the best sentence ?".split()))

#########################################################################
# Exercise 2: 5 minutes: try and find the sentence (len > 10 tokens) with the highest probability
#########################################################################


#########################################################################
# Exercise 3: 15 minutes:
#########################################################################

# create a function that takes a corpus as a parameter and returns a trigram model trained on the corpus
def estimate_lm(corpus, smoothing_factor=0.001):
    counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))
    for sentence in corpus:
        # add the special tokens to the sentences
        tokens = ['*', '*'] + sentence + ['STOP']
        for u, v, w in nltk.ngrams(tokens, 3):
            # update the counts
            counts[(u, v)][w] += 1
    return counts

def perplexity(corpus):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    p = 0
    token_count = 0
    for sent in corpus:
        token_count += len(sent)
        p += sentence_logP(sent)
    return np.exp(-p/token_count)



test_data = [["I'm", 'not', 'giving', 'you', 'a', 'chance', ',', 'Bill', ',', 'but', 'availing', 'myself', 'of', 'your', 'generous', 'offer', 'of', 'assistance', '.'], ['Good', 'luck', 'to', 'you', "''", '.'], ['``', 'All', 'the', 'in-laws', 'have', 'got', 'to', 'have', 'their', 'day', "''", ',', 'Adam', 'said', ',', 'and', 'glared', 'at', 'William', 'and', 'Freddy', 'in', 'turn', '.'], ['Sweat', 'started', 'out', 'on', "William's", 'forehead', ',', 'whether', 'from', 'relief', 'or', 'disquietude', 'he', 'could', 'not', 'tell', '.'], ['Across', 'the', 'table', ',', 'Hamrick', 'saluted', 'him', 'jubilantly', 'with', 'an', 'encircled', 'thumb', 'and', 'forefinger', '.'], ['Nobody', 'else', 'showed', 'pleasure', '.'], ['Spike-haired', ',', 'burly', ',', 'red-faced', ',', 'decked', 'with', 'horn-rimmed', 'glasses', 'and', 'an', 'Ivy', 'League', 'suit', ',', 'Jack', 'Hamrick', 'awaited', 'William', 'at', 'the', "officers'", 'club', '.'], ['``', 'Hello', ',', 'boss', "''", ',', 'he', 'said', ',', 'and', 'grinned', '.'], ['``', 'I', 'suppose', 'I', 'can', 'never', 'expect', 'to', 'call', 'you', "'", 'General', "'", 'after', 'that', 'Washington', 'episode', "''", '.'], ['``', "I'm", 'afraid', 'not', "''", '.']]


# use this function to train trigram models on each section of the brown corpus
# brown.categories()
lowest_perplex = 100000
domain = ""

for cat in brown.categories():
    train = brown.sents(categories=cat)
    sent_length = len(train)
    tokens = len([t for s in train for t in s])
    counts = estimate_lm(train)
    p = perplexity(test_data)
    if p < lowest_perplex:
        lowest_perplex = p
        domain = cat

print()
print("Exercise 3:")
print("Generation")
print("#"*80)
print("Best domain: {0} with perplexity={1:.3f}".format(domain,
                                                        lowest_perplex))
print()

# Which training corpus gives the lowest perplexity on this test data?


#########################################################################
# Exercise 4: 10 minutes:
#########################################################################

# using the starter code below, train language models for each domain in brown and generate 10 sentences

def sample_next_word(u, v):
    keys, values = zip(* counts[(u, v)]. items())
    values = np.array(values)
    values /= values.sum()
    return keys[np.argmax(np.random.multinomial(1, values))]


def generate():
    result = ['*', '*']
    next_word = sample_next_word(result[-2], result[-1])
    result.append(next_word)
    while next_word != 'STOP':
        next_word = sample_next_word(result[-2], result[-1])
        result.append(next_word)
    return ' '.join(result[2: -1])


print()
print("Exercise 4:")
print("Generation")
print("#"*80)
for x in range(1, 10):
    print(str(x) + ": " + generate())
print()

########################################################################
# Exercise 5
########################################################################

"""
So far, we have been using a kind of stupid smoothing technique, giving up entirely on computing an actual probability distribution. For this section, let's implement a correct version of Laplace smoothing. You'll need to keep track of the vocabulary as well.
"""
def estimate_lm_smoothed(corpus, alpha=1):
    vocab = set(['*', 'STOP'])
    # We will add alpha in in logP_smoothed, so we set lambda to 0
    s_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for sentence in corpus:
        vocab.update(sentence)
        # add the special tokens to the sentences
        tokens = ['*', '*'] + sentence + ['STOP']
        for u, v, w in nltk.ngrams(tokens, 3):
            # update the counts
            s_counts[(u, v)][w] += 1
    return s_counts, vocab



# Fill in the following functions with their smoothed version. You'll need to make sure the defaultdict for the counts has the same name as the output from estimate_lm_smoothed


def logP_smoothed(u, v, w, V, alpha=1):
    return np.log(s_counts[(u, v)][w] + alpha) - np.log(sum(s_counts[(u, v)].values()) + len(V) * alpha)

def sentence_logP_smoothed(S, V, alpha=1):
    """
    Adds the special tokens to the beginning and end and splits the sentence
    into tokens on whitespace.
    """
    tokens = ['*', '*'] + S + ['STOP']
    return sum([logP_smoothed(u, v, w, V, alpha) for u, v, w in nltk.ngrams(tokens, 3)])


def perplexity_smoothed(corpus, V, alpha=1):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    p = 0
    token_count = 0
    for sent in corpus:
        token_count += len(sent)
        p += sentence_logP_smoothed(sent, V, alpha)
    return np.exp(-p/token_count)


# Now train s_counts and compare perplexity with the original version on the heldout test set
s_counts, vocab = estimate_lm_smoothed(train)


print()
print("Exercise 5:")
print("Original vs. Smoothed")
print("#"*80)
print("original perplexity: {0:.3f}".format(perplexity(test)))
print("smoothed perplexity: {0:.3f}".format(perplexity_smoothed(test, vocab, alpha=1)))
print()



########################################################################
# Exercise 6
########################################################################

# now create a development set using some of the training data and find the optimal smoothing factor and retest on the test set

full_corpus = brown.sents()
train_idx = int(len(full_corpus) * .7)
dev_idx = int(len(full_corpus) * .8)
train = full_corpus[:train_idx]
dev = full_corpus[train_idx:dev_idx]
test = full_corpus[dev_idx:]

best_alpha = 1000
best_perplex = 100000

for alpha in [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]:
    s_counts, vocab = estimate_lm_smoothed(train)
    dev_perplex = perplexity_smoothed(dev, vocab, alpha=alpha)
    if dev_perplex < best_perplex:
        best_alpha = alpha
        best_perplex = dev_perplex
        print("current best: {0:.2f} with alpha={1}".format(best_perplex,
                                                            best_alpha))

print()
print("Exercise 6:")
print("Optimal alpha")
print("#"*80)
print("Best alpha: {0}".format(best_alpha))
print("smoothed perplexity: {0:.3f}".format(perplexity_smoothed(test, vocab, alpha=best_alpha)))
print()


##########################################################################
# Exercise 7
##########################################################################

# Now create an interpolated language model (1,2,3 grams)

def estimate_ngram(corpus, N=3):
    vocab = set(['*', 'STOP'])
    if N > 1:
        counts = defaultdict(lambda: defaultdict(lambda: 0))
    else:
        counts = defaultdict(lambda: 0)
    for sentence in corpus:
        vocab.update(sentence)
        # add the special tokens to the sentences
        if N > 1:
            tokens = ['*'] * (N - 1) + sentence + ['STOP']
            for ngram in nltk.ngrams(tokens, N):
                # update the counts
                counts[ngram[:-1]][ngram[-1]] += 1
        else:
            for token in sentence:
                counts[token] += 1
    return counts, vocab

def logP_trigram(counts, u, v, w, vocab, alpha=1):
    return np.log(counts[(u, v)][w] + alpha) - np.log(sum(counts[(u, v)].values()) + len(vocab) * alpha)

def logP_bigram(counts, u, v, vocab, alpha=1):
    return np.log(counts[(u,)][v] + alpha) - np.log(sum(s_counts[(u)].values()) + len(vocab) * alpha)

def logP_unigram(counts, u, vocab, alpha=1):
    return np.log(counts[u] + alpha) - np.log(len(vocab) + len(vocab) * alpha)


def sentence_interpolated_logP(S, vocab, uni_counts, bi_counts, tri_counts, lambdas=[0.5, 0.3, 0.2], alpha=1):
    tokens = ['*', '*'] + S + ['STOP']
    prob = 0
    for u, v, w in nltk.ngrams(tokens, 3):
        tri_prob = logP_trigram(tri_counts, u, v, w, vocab, alpha)
        bi_prob = logP_bigram(bi_counts, u, v, vocab, alpha)
        uni_prob = logP_unigram(uni_counts, u, vocab, alpha)
        prob += lambdas[0] * tri_prob + lambdas[1] * bi_prob + lambdas[2] * uni_prob
    return prob

def interpolated_perplexity(corpus, vocab, uni_counts, bi_counts, tri_counts, lambdas=[0.5, 0.3, 0.2], alpha=1):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    p = 0
    token_count = 0
    for sent in corpus:
        token_count += len(sent)
        p += sentence_interpolated_logP(sent, vocab, uni_counts, bi_counts, tri_counts, lambdas, alpha)
    return np.exp(-p/token_count)


uni, vocab = estimate_ngram(train, 1)
bi, _ = estimate_ngram(train, 2)
tri, _ = estimate_ngram(train, 3)


print()
print("Exercise 7:")
print("Interpolation")
print("#"*80)
print("original perplexity: {0:.3f}".format(perplexity(test)))
print("smoothed perplexity: {0:.3f}".format(perplexity_smoothed(test, vocab, alpha=1)))
print("interpolated perplexity: {0:.3f}".format(interpolated_perplexity(test, vocab, uni, bi, tri, alpha=best_alpha)))


######################################################
# Exercise 8: Build a simple spelling corrector
######################################################

# keys are errors, values are correct
common_errors = {"ei": "ie",  # acheive: achieve
                 "ie": "ei",  # recieve: receive
                 "ant": "ent",  # apparant: apparent
                 "m": "mm",  # accomodate: accommodate
                 "s": "ss",  # profesional: professional
                 "teh": "the",
                 "too": "to",
                 "their": "there",
                 "there": "they're"
                 }

def spell_check(sent, common_errors):
    sents = []
    probs = []
    sents.append(sent)
    probs.append(sentence_interpolated_logP(sent, vocab, uni, bi, tri))
    for i, token in enumerate(sent):
        for error, correct in common_errors.items():
            new_tok = token.replace(error, correct)
            if i < len(sent) - 1:
                new_sent = sent[:i] + [new_tok] + sent[i+1:]
            else:
                new_sent = sent[:i] + [new_tok]
            if new_sent != sent:
                sents.append(new_sent)
                probs.append(sentence_interpolated_logP(new_sent, vocab, uni, bi, tri))
    max_i = np.argmax(probs)
    return sents[max_i], probs[max_i]


test_set = ["I do not know who recieved it".split(),
            "That is not profesional".split(),
            "we saw teh man running".split(),
            "We tried too help them".split(),
            ]


print()
print("Exercise 8:")
print("Spell checker")
print("#"*80)

for sent in test_set:
    new, prob = spell_check(sent, common_errors)
    print("original: {0}".format(" ".join(sent)))
    print("spellchecked: {0}".format(" ".join(new)))
    print("-"*40)
