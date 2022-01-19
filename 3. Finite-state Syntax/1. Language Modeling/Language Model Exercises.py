#!/usr/bin/env python
# coding: utf-8

# **Exercise 1: Train a simple trigram language model**
# ---
# 
# -----
# 
# In the first exercise, we´ll save the counts directly in a dictionary
#  which defaults to the smoothing factor (_note that this is not true smoothing
#  as it does not account for the denominator and therefore does not create a
#  true probability distribution, but it is enough to get started_)

# In[1]:


import nltk
from collections import defaultdict
import numpy as np
import nltk.corpus
from nltk.corpus import brown

import numpy as np

# choose a small smoothing factor
smoothing_factor = 0.001
counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))


# We'll also define two helper functions, one to get the log probability of
# a single trigram and the second to get the log probability of a full sentence

# In[2]:


def logP(u, v, w):
    """
    Compute the log probability of a specific trigram
    """
    return np.log(counts[(u, v)][w]) - np.log(sum(counts[(u, v)]. values()))


def sentence_logP(S):
    """
    Adds the special tokens to the beginning and end.
    Then calculates the sum of log probabilities of
    all trigrams in the sentence.
    """
    tokens = ['*', '*'] + S + ['STOP']
    return sum([logP(u, v, w) for u, v, w in nltk.ngrams(tokens, 3)])


# We then choose the corpus. We'll use the preprocessed Brown corpus (nltk.corpus.brown), which contains many domains.
# To see the domains, you can run brown.categories(). We also split this into train, dev, and test sets, which we will use throughout.

# In[3]:


sentences = brown.sents(categories='news')
dev_idx = int(len(sentences) * .7)
test_idx = int(len(sentences) * .8)
train = sentences[:dev_idx]
dev = sentences[dev_idx:test_idx]
test = sentences[test_idx:]


# Finally, we'll collect the counts in the dictionary we set up before.

# In[4]:


for sentence in train:
    # add the special tokens to the sentences
    tokens = ['*', '*'] + sentence + ['STOP ']
    for u, v, w in nltk.ngrams(tokens, 3):
        # update the counts
        counts[(u, v)][w] += 1


# In[5]:


# Now that we have the model we can use it
print(sentence_logP("what is the best sentence ?".split()))


# **Exercise 2: (3-5 minutes) **
# ---
# 
# -----
# 
# **Try and find the sentence (len > 10 tokens) with the highest probability**
# 
# 1. What is the sentence with the highest probability you could find?
# 2. What is it's log probability?

# **Exercise 3: Function for trigram model, define perplexity, find the best train domain (15-20 minutes) **
# ---
# 
# -----

# First, you'll need to define a function to train the trigram models. It should return the same kind of counts dictionary as in Exercise 1.

# In[6]:



def estimate_lm(corpus, smoothing_factor=0.001):
    """This function takes a corpus and returns a trigram model (counts) trained on the corpus """
    
    # Finish the code here


# Now, you'll need to define a function to measure perplexity, which is defined as the exp(total negative log likelihood / total_number_of_tokens). See https://web.stanford.edu/~jurafsky/slp3/3.pdf for more info.
# 
# Luckily, we already have a function to get the log likelihood of a sentence (sentence_logP). So we can iterate over the sentences in a corpus, summing the log probability of each sentence, and keeping track of the total number of tokens. Finally, you can get the NEGATIVE log likelihood and average this, finally using np.exp to exponentiate the previous result.

# In[7]:


def perplexity(corpus):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. 
    """
    total_log_likelihood = 0
    total_token_count = 0
    
    # Finish the code here


# In[8]:


test_data = [["I'm", 'not', 'giving', 'you', 'a', 'chance', ',', 'Bill', ',', 'but', 'availing', 'myself', 'of', 'your', 'generous', 'offer', 'of', 'assistance', '.'], ['Good', 'luck', 'to', 'you', "''", '.'], ['``', 'All', 'the', 'in-laws', 'have', 'got', 'to', 'have', 'their', 'day', "''", ',', 'Adam', 'said', ',', 'and', 'glared', 'at', 'William', 'and', 'Freddy', 'in', 'turn', '.'], ['Sweat', 'started', 'out', 'on', "William's", 'forehead', ',', 'whether', 'from', 'relief', 'or', 'disquietude', 'he', 'could', 'not', 'tell', '.'], ['Across', 'the', 'table', ',', 'Hamrick', 'saluted', 'him', 'jubilantly', 'with', 'an', 'encircled', 'thumb', 'and', 'forefinger', '.'], ['Nobody', 'else', 'showed', 'pleasure', '.'], ['Spike-haired', ',', 'burly', ',', 'red-faced', ',', 'decked', 'with', 'horn-rimmed', 'glasses', 'and', 'an', 'Ivy', 'League', 'suit', ',', 'Jack', 'Hamrick', 'awaited', 'William', 'at', 'the', "officers'", 'club', '.'], ['``', 'Hello', ',', 'boss', "''", ',', 'he', 'said', ',', 'and', 'grinned', '.'], ['``', 'I', 'suppose', 'I', 'can', 'never', 'expect', 'to', 'call', 'you', "'", 'General', "'", 'after', 'that', 'Washington', 'episode', "''", '.'], ['``', "I'm", 'afraid', 'not', "''", '.']]


# Finally, use *estimate_lm()* to train LMs on each domain in brown.categories() and 
# find which gives the lowest perplexity on test_data. 
# 
# 1. Which domain gives the best perplexity?
# 2. Can you think of a way to use language models to predict domain?

# In[10]:


for domain in brown.categories():
    train = brown.sents(categories=domain)
    
    # Finish the code here


# **Exercise 4: Generation **
# ---
# 
# -----

# For the next exercise, you will need to generate 10 sentences for each domain in the Brown corpus. The first thing we need is code to be able to sample the next word in a trigram. We'll do this by creating a probability distribution over the values in our trigram counts. Remember that each key in the dictionary is a tuple (u, v) and that the values is another dictionary with the count of the continuation w: count. Therefore, we can create a numpy array with the continuation values and divide by the sum of values to get a distribution. Finally, we can use np.random.multinomial to sample from this distribution.

# In[12]:


def sample_next_word(u, v):
    keys, values = zip(* counts[(u, v)]. items())
    # convert values to np.array
    values = np.array(values)
    # divide by sum to create prob. distribution
    values /= values.sum()  
    # return the key (continuation token) for the sample with the highest probability
    return keys[np.argmax(np.random.multinomial(1, values))]  


# Now we can create a function that will generate text using our trigram model. You will need to start out with the two special tokens we used to train the model, and continue adding to this output, sampling the next word at each timestep. If the word sampled is the end token ('STOP'), then stop the generation and return the sequence as a string.

# In[13]:


def generate():
    """
    Sequentially generates text using sample_next_word().
    When the token generated is 'STOP', it returns the generated tokens as a string,
    removing the start and end special tokens.
    """
    result = ['*', '*']
    
    # Finish the code here


# Finally, use the code above to generate 10 sentences per domain in the Brown corpus.
# 
# 1. Do you see any correlation between perplexity scores and generated text?

# **Exercise 5: Smoothing **
# ---
# 
# -----

# So far, we have been using a kind of stupid smoothing technique, giving up entirely on computing an actual probability distribution. For this section, let's implement a correct version of Laplace smoothing. You'll need to keep track of the vocabulary as well, and don't forget to add the special tokens.

# In[15]:


def estimate_lm_smoothed(corpus, alpha=1):
    counts = defaultdict(lambda: defaultdict(lambda: alpha))
    vocab = set()
    
    # Finish the code here

    return counts, vocab


# The main change is not in how we estimate the counts, but in how we calculate log probability for each trigram.
# Specifically, we need to add the size_of_the_vocabulary * alpha to the denominator.

# In[16]:


def logP_smoothed(u, v, w, V, alpha=1):
    # Finish the code here
    pass

def sentence_logP_smoothed(S, V, alpha=1):
    """
    Adds the special tokens to the beginning and end.
    Then calculates the sum of log probabilities of
    all trigrams in the sentence using logP_smoothed.
    """
    # Finish the code here
    pass

def perplexity_smoothed(corpus, V, alpha=1):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    # Finish the code here
    pass


# Now train s_counts and vocab and compare perplexity with the original version on the heldout test set.

# **Exercise 6: Interpolation**
# ---
# 
# -----

# To be able to interpolate unigram, bigram, and trigram models, we first need to train them. So here you need to make a function that takes 1) a corpus and 2) an n-gram (1,2,3) and 3) a smoothing factor and returns the counts and vocabulary. Notice that for the unigram model, you will have to set up the dictionary in a different way than we have done until now.

# In[17]:


def estimate_ngram(corpus, N=3, smoothing_factor=1):
    vocab = set(['*', 'STOP'])
    if N > 1:
        # set up the counts like before
        counts = None
    else:
        # set them up as necessary for the unigram model
        counts = None
        
    # Finish the code here
    return counts, vocab


# You will also need separate functions to get the log probability for each ngram.

# In[18]:


def logP_trigram(counts, u, v, w, vocab, alpha=1):
    # Finish the code here
    pass


def logP_bigram(counts, u, v, vocab, alpha=1):
    # Finish the code here
    pass

def logP_unigram(counts, u, vocab, alpha=1):
    # Finish the code here
    pass


# In this case, the main change is in calculating the log probability of the sentence. 

# In[21]:


def sentence_interpolated_logP(S, vocab, uni_counts, bi_counts, tri_counts, lambdas=[0.5, 0.3, 0.2]):
    tokens = ['*', '*'] + S + ['STOP']
    prob = 0
    for u, v, w in nltk.ngrams(tokens, 3):
        # Finish the code here
        # Calculate the log probabilities for each ngram and then multiply them by the lambdas and sum them.
        pass
    return prob

def interpolated_perplexity(corpus, vocab, uni_counts, bi_counts, tri_counts, smoothing_factor=1, lambdas=[0.5, 0.3, 0.2]):
    """
    Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. 
    In this case, we approximate perplexity over the full corpus as an average of sentence-wise perplexity scores.
    """
    p = 0
    # Finish the code here
    
    pass


# Finally, train unigram, bigram, and trigram models and computer the perplexity of the interpolated model on the test set.

# **Exercise 7: Build a simple spelling corrector**
# ---
# 
# -----

# In this section, we will build a simple spelling corrector with two components: 1) a dictionary of common spelling errors which will allow us to create possible hypothesis sentences and 2) a language model to filter the most likely sentence.

# In[22]:


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


# In[23]:


test_set = ["I do not know who recieved it".split(),
            "That is not profesional".split(),
            "we saw teh man running".split(),
            "We tried too help them".split(),
            ]


# For the spell checker,

# In[25]:


def spell_check(sent, common_errors):
    sents = []
    probs = []
    sents.append(sent)
    probs.append(sentence_logP(sent))
    
    # create new hypothesis sentences by recursively applying all possible spelling mistakes to 
    # each token in the sentence. If the new sentence is not the same as the original, append
    # it to sents and compute its probability and append it to probs.
    for i, token in enumerate(sent):
        for error, correct in common_errors.items():
            
            # Finish the code here
            pass
        
    # Finally take the argmax of the probabilities and return that sentence
    max_i = np.argmax(probs)
    return sents[max_i], probs[max_i]


# It would be a good idea to retrain your langauge model on all of the Brown sentences (brown.sents()) in order to improve it's recall.
# 
# 1. After retraining, do you notice any differences?
