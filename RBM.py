"""Restricted Boltzmann Machine with softmax visible units.
Based on sklearn's BernoulliRBM class.
"""

# Authors: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
#          Vlad Niculae
#          Gabriel Synnaeve
#          Lars Buitinck
# License: BSD 3 clause

import time
import re

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils import shuffle
from sklearn.utils.extmath import safe_sparse_dot, log_logistic
import scipy
from scipy.special import expit             # logistic function
from sklearn.utils.validation import check_is_fitted

import Utils

# Experiment: when sampling with high temperature (>1), use the softmax probabilities
# of the biases as the prior rather than a uniform distribution. Based on the observation
# that annealing starting from a high temperature often resulted in samples that were
# highly biased toward long strings (because a uniform distribution over the visible
# units will tend to produce strings of the maximum length).
# This kind of helped but wasn't amazing. Possibly I just needed a longer/gentler annealing schedule?
BIASED_PRIOR = 0

class BernoulliRBM(BaseEstimator, TransformerMixin):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Parameters
    ----------

    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    intercept_hidden_ : array-like, shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like, shape (n_features,)
        Biases of the visible units.

    components_ : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """

    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, lr_backoff=False, weight_cost=0):
        self.n_components = n_components
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_backoff = lr_backoff
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.rng_ = check_random_state(self.random_state)
        self.weight_cost = weight_cost
        # A history of some summary statistics recorded at the end of each epoch of training
        # Each key maps to a 2-d array. One row per 'session', one value per epoch.
        # (Another session means this model was pickled, then loaded and fit again.)
        self.history = {'pseudo-likelihood': [], 'overfit': []}

    # TODO
    # Experimental: How many times more fantasy particles compared to minibatch size
    @property
    def fantasy_to_batch(self):
        return 1

    def record(self, name, value):
        if not hasattr(self, 'history'):
            self.history = {'pseudo-likelihood': [], 'overfit': []}
        self.history[name][-1].append(value)

    def _mean_hiddens(self, v, temperature=1.0):
        """Computes the probabilities P(h=1|v).

        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T/temperature)
        p += self.intercept_hidden_/(min(1.0, temperature) if BIASED_PRIOR else temperature)
        return expit(p, out=p)

    def _sample_hiddens(self, v, temperature=1.0):
        """Sample from the distribution P(h|v).

        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v, temperature)
        return (self.rng_.random_sample(size=p.shape) < p)

    def _sample_visibles(self, h, temperature=1.0):
        """Sample from the distribution P(v|h).

        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_/temperature)
        p += self.intercept_visible_/(min(1.0, temperature) if BIASED_PRIOR else temperature)
        expit(p, out=p)
        return (self.rng_.random_sample(size=p.shape) < p)

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        return (- safe_sparse_dot(v, self.intercept_visible_)
                - np.logaddexp(0, safe_sparse_dot(v, self.components_.T)
                               + self.intercept_hidden_).sum(axis=1))

    def gibbs(self, v, temperature=1.0):
        """Perform one Gibbs sampling step.

        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self, "components_")
        h_ = self._sample_hiddens(v, temperature)
        v_ = self._sample_visibles(h_, temperature)

        return v_

    def repeated_gibbs(self, v, niters):
        """Perform n rounds of alternating Gibbs sampling starting from the
        given visible vectors.
        """
        for i in range(niters):
            h = self._sample_hiddens(v)
            v = self._sample_visibles(h, temperature=1.0)
        return v

    def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.

        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float)
        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.rng_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='fortran')
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X)

    def _fit(self, v_pos):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.
        """
        h_pos = self._mean_hiddens(v_pos)
        # TODO: Worth trying with visible probabilities rather than binary states.
        # PG: it is common to use p_i instead of sampling a binary value'... 'it reduces
        # sampling noise this allowing faster learning. There is some evidence that it leads
        # to slightly worse density models'

        # I'm confounded by the fact that we seem to get more effective models WITHOUT
        # softmax visible units. The only explanation I can think of is that it's like
        # a pseudo-version of using visible probabilities. Without softmax, v_neg
        # can have multiple 1s per one-hot vector, which maybe somehow accelerates learning?
        # Need to think about this some more.
        v_neg = self._sample_visibles(self.h_samples_)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg) / self.fantasy_to_batch
        # L2 weight penalty
        update -= self.components_ * self.weight_cost
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0)/self.fantasy_to_batch)
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0)/self.fantasy_to_batch)

        h_neg[self.rng_.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def corrupt(self, v):
        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               self.rng_.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]
        return v_, None

    def uncorrupt(self, visibles, state):
        pass

    @Utils.timeit
    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, "components_")

        v = check_array(X, accept_sparse='csr')
        fe = self._free_energy(v)

        v_, state = self.corrupt(v)
        # TODO: If I wanted to be really fancy here, I would do one of those "with..." things.
        fe_corrupted = self._free_energy(v)
        self.uncorrupt(v, state)

        # See https://en.wikipedia.org/wiki/Pseudolikelihood
        # Let x be some visible vector. x_i is the ith entry. x_-i is the vector except that entry.
        #       x_iflipped is x with the ith bit flipped. F() is free energy.
        # P(x_i | x_-i) = P(x) / P(x_-i) = P(x) / (P(x) + p(x_iflipped))
        # expand def'n of P(x), cancel out the partition function on each term, and divide top and bottom by e^{-F(x)} to get...
        # 1 / (1 + e^{F(x) - F(x_iflipped)})
        # So we're just calculating the log of that. We multiply by the number of
        # visible units because we're approximating P(x) as the product of the conditional likelihood
        # of each individual unit. But we're too lazy to do each one individually, so we say the unit
        # we tested represents an average.
        if hasattr(self, 'codec'):
            normalizer = self.codec.shape()[0]
        else:
            normalizer = v.shape[1]
        return normalizer * log_logistic(fe_corrupted - fe)

    # TODO: No longer used
    def pseudolikelihood_ratio(self, good, bad):
        assert good.shape == bad.shape
        good_energy = self._free_energy(good)
        bad_energy = self._free_energy(bad)
        # Let's do ratio of log probabilities instead
        return (bad_energy - good_energy).mean()

    @Utils.timeit
    def score_validation_data(self, train, validation):
        """Return the energy difference between the given validation data, and a
        subset of the training data. This is useful for monitoring overfitting.
        If the model isn't overfitting, the difference should be around 0. The
        greater the difference, the more the model is overfitting.
        """
        # It's important to use the same subset of the training data every time (per Hinton's "Practical Guide")
        return self._free_energy(train[:validation.shape[0]]).mean(), self._free_energy(validation).mean()

    def fit(self, X, validation=None):
        """Fit the model to the data X.

        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.

        validation : {array-like, sparse matrix}

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float)
        n_samples = X.shape[0]

        if not hasattr(self, 'components_'):
            self.components_ = np.asarray(
                self.rng_.normal(0, 0.01, (self.n_components, X.shape[1])),
                order='fortran')
            self.intercept_hidden_ = np.zeros(self.n_components, )
            # 'It is usually helpful to initialize the bias of visible unit i to log[p_i/(1-p_i)] where p_i is the prptn of training vectors where i is on' - Practical Guide
            # TODO: Make this configurable?
            if 1:
                counts = X.sum(axis=0).A.reshape(-1)
                # There should be no units that are always on
                assert np.max(counts) < X.shape[0], "Found a visible unit always on in the training data. Fishy."
                # There might be some units never on. Add a pseudo-count of 1 to avoid inf
                vis_priors = (counts + 1) / float(X.shape[0])
                self.intercept_visible_ = np.log( vis_priors / (1 - vis_priors) )
            else:
                self.intercept_visible_ = np.zeros(X.shape[1], )

        # If this already *does* have weights and biases before fit() is called,
        # we'll start from them rather than wiping them out. May want to train
        # a model further with a different learning rate, or even on a different
        # dataset.
        else:
            print ("Reusing existing weights and biases")
        # Don't necessarily want to reuse h_samples if we have one leftover from before - batch size might have changed
        self.h_samples_ = np.zeros((self.batch_size * self.fantasy_to_batch, self.n_components))

        # Add new inner lists for this session
        if not hasattr(self, 'history'):
            self.history = {'pseudo-likelihood': [], 'overfit': []}
        for session in self.history.items():
            value = session[1]
            session[1].append([])

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in range(1, self.n_iter + 1):
            if self.lr_backoff:
                # If, e.g., we're doing 10 epochs, use the full learning rate for
                # the first iteration, 90% of the base learning rate for the second
                # iteration... and 10% for the final iteration
                self.learning_rate = ((self.n_iter - (iteration - 1)) / (self.n_iter+0.0)) * self.base_learning_rate
                print ("Using learning rate of {:.3f} (base LR={:.3f})".format(self.learning_rate, self.base_learning_rate))

            for batch_slice in batch_slices:
                self._fit(X[batch_slice])

            if verbose and iteration != self.n_iter:
                end = time.time()
                self.wellness_check(iteration, end - begin, X, validation)
                begin = end
            if iteration != self.n_iter:
                X = shuffle(X)

        return self

    def wellness_check(self, epoch, duration, train, validation):
        """Log some diagnostic information on how the model is doing so far."""
        validation_debug = ''
        if validation is not None:
            t_energy, v_energy = self.score_validation_data(train, validation)
            validation_debug = "\nE(vali):\t{:.2f}\tE(train):\t{:.2f}\tdifference: {:.2f}".format(
                v_energy, t_energy, v_energy-t_energy)
            self.record('overfit', (v_energy, t_energy))

        # TODO: This is pretty expensive. Figure out why? Or just do less often.
        # Also, can use crippling amounts of memory for large datasets. Hack...
        pseudo = self.score_samples(train[:min(train.shape[0], 10**5)])
        self.record('pseudo-likelihood', pseudo.mean())
        print (re.sub('\n *', '\n', """[{}] Iteration {}/{}\tt = {:.2f}s
                Pseudo-log-likelihood sum: {:.2f}\tAverage per instance: {:.2f}{}""".format
                     (type(self).__name__, epoch, self.n_iter, duration,
                      pseudo.sum(), pseudo.mean(), validation_debug,
                      )))


class CharBernoulliRBM(BernoulliRBM):

    def __init__(self, codec, *args, **kwargs):
        """
        codec is the ShortTextCodec used to create the vectors being fit. The
        most important function of the codec is as a proxy to the shape of the
        softmax units in the visible layer (if you're using the CharBernoulliRBMSoftmax
        subclass). It's also used to decode and print
        fantasy particles at the end of each epoch.
        """
        # Attaching this to the object is really helpful later on when models
        # are loaded from pickle in visualize.py and sample.py
        self.codec = codec
        self.softmax_shape = codec.shape()
        # Old-style class :(
        BernoulliRBM.__init__(self, *args, **kwargs)

    def wellness_check(self, epoch, duration, train, validation):
        BernoulliRBM.wellness_check(self, epoch, duration, train, validation)
        fantasy_samples = '|'.join([self.codec.decode(vec) for vec in
                                    self._sample_visibles(self.h_samples_[:3], temperature=0.1)])
        print ("Fantasy samples: {}".format(fantasy_samples))

    def corrupt(self, v):
        n_softmax, n_opts = self.softmax_shape
        # Select a random index in to the indices of the non-zero values of each input
        # TODO: In the char-RBM case, if I wanted to really challenge the model, I would avoid selecting any
        # trailing spaces here. Cause any dumb model can figure out that it should assign high energy to
        # any instance of /  [^ ]/
        meta_indices_to_corrupt = self.rng_.randint(0, n_softmax, v.shape[0]) + np.arange(0, n_softmax * v.shape[0], n_softmax)

        # Offset these indices by a random amount (but not 0 - we want to actually change them)
        offsets = self.rng_.randint(1, n_opts, v.shape[0])
        # Also, do some math to make sure we don't "spill over" into a different softmax.
        # E.g. if n_opts=5, and we're corrupting index 3, we should choose offsets from {-3, -2, -1, +1}
        # 1-d array that matches with meta_i_t_c but which contains the indices themselves
        indices_to_corrupt = v.indices[meta_indices_to_corrupt]
        # Sweet lucifer
        offsets = offsets - (n_opts * (((indices_to_corrupt % n_opts) + offsets.ravel()) >= n_opts))

        v.indices[meta_indices_to_corrupt] += offsets
        return v, (meta_indices_to_corrupt, offsets)

    def uncorrupt(self, visibles, state):
        mitc, offsets = state
        visibles.indices[mitc] -= offsets


class CharBernoulliRBMSoftmax(CharBernoulliRBM):

    def _sample_visibles(self, h, temperature=1.0):
        """Sample from the distribution P(v|h). This obeys the softmax constraint
        on visible units. i.e. sum(v) == softmax_shape[0] for any visible
        configuration v.

        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_/temperature)
        p += self.intercept_visible_/(min(1.0, temperature) if BIASED_PRIOR else temperature)
        nsamples, nfeats = p.shape
        reshaped = np.reshape(p, (nsamples,) + self.softmax_shape)
        return Utils.softmax_and_sample(reshaped).reshape((nsamples, nfeats))
