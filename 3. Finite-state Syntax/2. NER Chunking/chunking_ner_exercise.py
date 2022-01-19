import nltk

from nltk.corpus import conll2000


##########################################################################
# Exercise 1: Chunking
##########################################################################

print(conll2000.chunked_sents('train.txt')[99])

""" Let's do some data exploration.
1. First, how many sentences are there?
2. How many NP chunks?
3. How many VP chunks?
4. How many PP chunks?
5. What is the average length of each?
"""




##########################################################################
# Exercise 2: Unigram chunker
##########################################################################


"""
Now, let's concentrate only on NP chunking
1. Create a unigram chunker using the UnigramChunker class below.
Train on the train sentences and evaluate on the test sentences using
the evaluate method, i.e., my_model.evaluate(test_sents).


2. What is the F1 score?


"""


class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

##########################################################################
# Exercise 3: Bigram/Trigram chunker
##########################################################################

"""
Now, modify the code to create Bigram and Trigram taggers

"""



##########################################################################
# Exercise 4: Maximum Entropy model with features
##########################################################################

"""
Finally, we will use a maximum entropy classifier (a discriminative classifier)
to model the chunking task. Remember that discriminative classifiers attempt to
model p(y|x) directly, which allows us more freedom in what x is.
"""

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    return {"pos": pos}


class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, max_iter=10, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)



##########################################################################
# Exercise 5: Add more features to get better performance
##########################################################################

"""Add in more features to get better performance"""



##########################################################################
# Exercise 6: NER Data
##########################################################################

"""
We will use a dataset of tweets annotated for Named Entities.

@inproceedings{derczynski-etal-2016-broad,
    title = "Broad {T}witter Corpus: A Diverse Named Entity Recognition Resource",
    author = "Derczynski, Leon  and
      Bontcheva, Kalina  and
      Roberts, Ian",
    booktitle = "Proceedings of {COLING} 2016, the 26th International Conference on Computational Linguistics: Technical Papers",
    month = dec,
    year = "2016",
    address = "Osaka, Japan",
    publisher = "The COLING 2016 Organizing Committee",
    url = "https://aclanthology.org/C16-1111",
    pages = "1169--1179",
    abstract = "One of the main obstacles, hampering method development and comparative evaluation of named entity recognition in social media, is the lack of a sizeable, diverse, high quality annotated corpus, analogous to the CoNLL{'}2003 news dataset. For instance, the biggest Ritter tweet corpus is only 45,000 tokens {--} a mere 15{\%} the size of CoNLL{'}2003. Another major shortcoming is the lack of temporal, geographic, and author diversity. This paper introduces the Broad Twitter Corpus (BTC), which is not only significantly bigger, but sampled across different regions, temporal periods, and types of Twitter users. The gold-standard named entity annotations are made by a combination of NLP experts and crowd workers, which enables us to harness crowd recall while maintaining high quality. We also measure the entity drift observed in our dataset (i.e. how entity representation varies over time), and compare to newswire. The corpus is released openly, including source text and intermediate annotations.",
}
"""

# load the data (1000 annotated tweets)
import json
from sklearn.metrics import f1_score

ner_data = []
for line in open("twitter_NER.json"):
    ner_data.append(json.loads(line))


# each example in ner_data is a json dictionary that contains two values we are interested in: tokens, entities

print(ner_data[6]["tokens"])
print(ner_data[6]["entities"])

# we need the training data as a list of lists, where the inner list contains tuples of (token, label) i.e., [[(token_1, label_1 ), (token_2, label_2), ...]]

# Test data should be a list of lists with the inner list having tokens

# Test labels should be a flat list of labels ['O', 'B-PER', 'I-PER', 'O'...]

ner_train_data = []
ner_test_data = []
ner_test_labels = []
for s in ner_data[:800]:
    ner_train_data.append(list(zip(s["tokens"], s["entities"])))
for s in ner_data[800:]:
    ner_test_data.append(s["tokens"])
    ner_test_labels.extend(s["entities"])

##########################################################################
# Exercise 6: Hidden Markov Model
##########################################################################

tagger = nltk.HiddenMarkovModelTagger.train(ner_train_data)


# Evaluate the model using the f1_score function from sklearn.metrics
# You should use macro F1 and make sure NOT TO COUNT the 'O' label

f1 = 0.00
print("F1 score: {0:.3f}".format(f1))


##########################################################################
# Exercise 7: Optional further exercises
##########################################################################

"""
The previous F1 is calculated at token level. However, for NER, we often calculate F1 at entity level.

Implement your own code to evaluate entity-level NER

1. You will need to calculate:
    Precision = (number of correctly predicted entities) / (number of predicted entities)

    Recall = (number of correctly predicted entites) / (number of gold entities)

    F1 = (2 * Precision * Recall) / (Precision + Recall)


2. You might want to implement a helper function which, given labels in IOB2 format return a list of all the entities.

"""
