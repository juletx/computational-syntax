import numpy as np

class HMM:
    """
    Q = a set of N hidden states
    A = a transition probability matrix where each a_i_j represents the probability of moving from state i to state j and np.sum(A) == 1
    O = a sequence of T observations from V
    V = a vocabulary of seen observations
    B = a sequence of emission probabilities expressing the probability of an observation o_t being generated from a state i
    PI = an initial probability distribution over states
    """

    def __init__(self, Q, V):

        self.Q = Q
        self.V = V

        # Q x Q matrix
        self.A = np.random.random((len(Q), len(Q)))
        # make sure all rows sum to 1
        self.A /= self.A.sum(1)

        # Emission probability matrix, Q x V
        self.B = np.random.random((len(V), len(Q)))
        self.B /= self.B.sum(0)

        # initial probability distribution over states, Q vector
        self.PI = np.random.random((len(Q)))

    def viterbi(self, O):
        """
        O = sequence of T observations
        This code should return an array backpointer with the
        most likely sequence through the trellis.

        e.g. O = [3, 1]
             backpointer = [0, 1]

        """

        # Set up trellis = Q x T matrix
        trellis = np.zeros((len(self.Q), len(O)))

        # set up backpointer
        backpointer = np.zeros(len(O))

        # compute the emission probabilaties for each beginning state
        trellis[0][0] = self.PI[0] * self.B[O[0]-1][0]
        trellis[1][0] = self.PI[1] * self.B[O[0]-1][1]
        backpointer[0] = trellis[:, 0].argmax()

        # compute forward probabilities keeping track of argmaxes with backpointer

        ###############################################################
        # ADD YOUR CODE HERE
        ###############################################################

        return backpointer


def main():

    """
    The example is taken from Jurafsky and Martin Ch. 8.4.

    This HMM predicts whether the day was hot or cold,
    depending on the number of icecreams that the
    author ate that day (1, 2, or 3).
    """
    label_map = {0: "hot", 1: "cold"}

    hmm = HMM(Q=["hot", "cold"], V=[1, 2, 3])

    # We set the values to those in Jurafsky and Martin
    hmm.A = np.array([[0.6, 0.4],
                      [0.5, 0.5]])

    hmm.B = np.array([[0.2, 0.5],
                      [0.4, 0.4],
                      [0.4, 0.1]])

    hmm.PI = np.array([.8, .2])

    # The observed sequence of icecream eaten we want to predict
    O = [3, 1, 3, 2, 2, 2, 2]

    labels = hmm.viterbi(O)

    print([label_map[l] for l in labels])


if __name__ == "__main__":
    main()
