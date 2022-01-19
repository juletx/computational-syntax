import nltk
from nltk.corpus import brown
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm,
                          labels,
                          cmap=plt.cm.BuPu):
    """
    This function plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    x_tick_marks = np.arange(len(labels))
    y_tick_marks = np.arange(len(labels))
    plt.xticks(x_tick_marks, labels, rotation=45)
    plt.yticks(y_tick_marks, labels)
    #
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


##########################################################
# Include your code to train Ngram taggers here
##########################################################
for domain in brown.categories():
    # Create train/dev/test splits

    # train the Ngram taggers

    # get the dev and test accuracy and their difference

    # print out the domain/dev acc/test acc/ and difference

    pass

##########################################################
# Include your code to train Ngram taggers here
##########################################################

# convert test to the correct data format (a flat list of tags)

# remove the tags from the original test data and use tag_sents() to get the predictions from the final model

# convert the predictions to the correct data format (a flat list of predicted tags)

# get a set of the labels (sorted(set(test)))

# create the confusion matrix and plot it using the plot_confusion_matrix function
