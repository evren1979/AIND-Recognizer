import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
	
    """

    def select(self):
        """ select based on n_constant value
		
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
	
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
		
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Create a dictionary to store model : BIC_score pairs
        scores = {}

        # Calculate a score for each n. n represents the number of states in the model
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # create a model with size n. 
                model = self.base_model(n)

                # Calculate the logL of the model using model.score(...)
                logL = model.score(self.X, self.lengths)

                # The formula below calculates the number of parameters
                # p = n^2 + 2dn - 1, where n : number of states, d : number of features
                p = n**2 + (2*(model.n_features)*n - 1)

                # Calculate the BIC score
                BIC = -2*logL + p*np.log(len(self.X))

                scores[model] = BIC
            except Exception:
                # If exception then just continue with the next n
                continue

            
        # Empty dictionaries evaluate to False in Python.
        # Return the model with the least BIC score using min(scores, key=scores.get)
        # If the scores dictionary is empty then just return None
        return min(scores, key=scores.get) if bool(scores) else None
        

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Create a dictionary to store model : score pairs
        scores = {}

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # Create a model with number of states = n
                model = self.base_model(n)

                # Calculate the logL of the current word 
                logL = model.score(self.X, self.lengths)

                # Calculate the logL of all the other words
                logL_others = 0 
                for word in self.words:
                    if word is self.this_word:
                        continue
                    # self.hwords stores X, lengths for each word
                    X, lengths = self.hwords[word]
                    logL_others += model.score(X, lengths)

                # DIC score is calculated using the following formula
                DIC = logL - (1 / (len(self.hwords)-1))*logL_others

                # Store the model and its corresponding DIC score
                scores[model] = DIC

            except Exception:
                # continue if any exception is raised in the process
                continue

        # Return the model with the highest DIC score or None if the
        # the scores dictionary evaluates to false. Empty dicitonaries
        # in Python evaluate to false. 
        return max(scores, key=scores.get) if bool(scores) else None

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # scores stores model : CV score pairs
        scores = {}
        # Create a fold method from sklearn.model_selection.KFold
        split_method = KFold(n_splits=max(len(self.sequences), 2))

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                # Create a model with number of states = n
                model = self.base_model(n)

                # Create a list to hold all the scores
                fold_scores = []

                # We don't require train_idx here so using _ to not assign an
                # extra variable

                # Create the folds
                for _, test_idx in split_method.split(self.sequences):
                    # use the helper function combine_sequences from asl_utils
                    # In order to run hmmlearn using X, lengths tuples on new folds,
                    # subsets must be combined based on the indices given for the folds
                    X_test, lengths_test = combine_sequences(test_idx, self.sequences)

                    # score and append to fold_scores
                    fold_scores.append(model.score(X_test, lengths_test))

                # Use np.mean to calculate the mean of each score
                scores[model] = np.mean(fold_scores)
                
            except Exception:
                # Move along if any exception occurs
                continue

        # Return the model with the highest score if the scores dictionary contains
        # any key, value pairs otherwise return None
        return max(scores, key=scores.get) if bool(scores) else None

