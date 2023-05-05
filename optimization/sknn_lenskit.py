"""
User-based k-NN collaborative filtering.
"""
from lenskit.util.accum import kvp_minheap_insert
from lenskit.algorithms import Predictor
from lenskit.data import sparse_ratings
from lenskit import util
from numba import njit
import time
import numpy as np
import pandas as pd
import logging
from sys import intern
import os
os.environ["MKL_THREADING_LAYER"] = "tbb"


_logger = logging.getLogger(__name__)


@njit
def _agg_weighted_avg(iur, item, sims, use):
    """
    Weighted-average aggregate.
    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    rates = iur.row_vs(item)
    num = 0.0
    den = 0.0
    for j in use:
        num += rates[j] * sims[j]
        den += np.abs(sims[j])
    return num / den


@njit
def _agg_sum(iur, item, sims, use):
    """
    Sum aggregate
    Args:
        iur(matrix._CSR): the item-user ratings matrix
        item(int): the item index in ``iur``
        sims(numpy.ndarray): the similarities for the users who have rated ``item``
        use(numpy.ndarray): positions in sims and the rating row to actually use
    """
    # added idf
    item_vec = iur.row(item)
    idf = np.log(item_vec.shape[0] / np.sum(item_vec))

    x = 0.0
    for j in use:
        x += sims[j]
    return x * idf


@njit
def _score(items, results, iur, sims, nnbrs, min_sim, min_nbrs, agg):
    h_ks = np.empty(nnbrs, dtype=np.int32)
    h_vs = np.empty(nnbrs)
    used = np.zeros(len(results), dtype=np.int32)

    for i in range(len(results)):
        item = items[i]
        if item < 0:
            continue

        h_ep = 0

        # who has rated this item?
        i_users = iur.row_cs(item)

        # what are their similarities to our target user?
        i_sims = sims[i_users]

        # which of these neighbors do we really want to use?
        for j, s in enumerate(i_sims):
            # threshold
            if np.abs(s) < 1.0e-10:
                continue
            if min_sim is not None and s < min_sim:
                continue
            h_ep = kvp_minheap_insert(0, h_ep, nnbrs, j, s, h_ks, h_vs)

        if h_ep < min_nbrs:
            continue

        results[i] = agg(iur, item, i_sims, h_ks[:h_ep])
        used[i] = h_ep

    return used


@njit
def calculate_idf(iur):
    user_dim = iur.ncols
    item_dim = iur.nrows
    helper_vector = np.ones(user_dim)
    return np.log(user_dim / iur.mult_vec(iur, helper_vector))


class UserUser(Predictor):
    """
    User-user nearest-neighbor collaborative filtering with ratings. This user-user implementation
    is not terribly configurable; it hard-codes design decisions found to work well in the previous
    Java-based LensKit code.
    Args:
        nnbrs(int):
            the maximum number of neighbors for scoring each item (``None`` for unlimited)
        min_nbrs(int): the minimum number of neighbors for scoring each item
        min_sim(float): minimum similarity threshold for considering a neighbor
        feedback(str):
            Control how feedback should be interpreted.  Specifies defaults for the other
            settings, which can be overridden individually; can be one of the following values:
            ``explicit``
                Configure for explicit-feedback mode: use rating values, center ratings, and
                use the ``weighted-average`` aggregate method for prediction.  This is the
                default setting.
            ``implicit``
                Configure for implicit-feedback mode: ignore rating values, do not center ratings,
                and use the ``sum`` aggregate method for prediction.
        center(bool):
            whether to normalize (mean-center) rating vectors.  Turn this off when working
            with unary data and other data types that don't respond well to centering.
        aggregate(str):
            the type of aggregation to do. Can be ``weighted-average`` or ``sum``.
        use_ratings(bool):
            whether or not to use rating values; default is ``True``.  If ``False``, it ignores
            rating values and treates every present rating as 1.
    Attributes:
        user_index_(pandas.Index): User index.
        item_index_(pandas.Index): Item index.
        user_means_(numpy.ndarray): User mean ratings.
        rating_matrix_(matrix.CSR): Normalized user-item rating matrix.
        transpose_matrix_(matrix.CSR): Transposed un-normalized rating matrix.
    """
    IGNORED_PARAMS = ['feedback']
    EXTRA_PARAMS = ['center', 'aggregate', 'use_ratings']
    AGG_SUM = intern('sum')
    AGG_WA = intern('weighted-average')
    RATING_AGGS = [AGG_WA]

    def __init__(self, nnbrs, min_nbrs=1, min_sim=0, feedback='explicit', **kwargs):
        self.nnbrs = nnbrs
        self.min_nbrs = min_nbrs
        self.min_sim = min_sim

        if feedback == 'explicit':
            defaults = {
                'center': True,
                'aggregate': self.AGG_WA,
                'use_ratings': True
            }
        elif feedback == 'implicit':
            defaults = {
                'center': False,
                'aggregate': self.AGG_SUM,
                'use_ratings': False
            }
        else:
            raise ValueError(f'invalid feedback mode: {feedback}')

        defaults.update(kwargs)
        self.center = defaults['center']
        self.aggregate = intern(defaults['aggregate'])
        self.use_ratings = defaults['use_ratings']

    def fit(self, ratings, **kwargs):
        """
        "Train" a user-user CF model.  This memorizes the rating data in a format that is usable
        for future computations.
        Args:
            ratings(pandas.DataFrame): (user, item, rating) data for collaborative filtering.
        """
        util.check_env()
        start = time.time()
        uir, users, items = sparse_ratings(ratings)
        end = time.time()
        print('Mapping build time: ', end - start)

        # mean-center ratings
        if self.center:
            umeans = uir.normalize_rows('center')
        else:
            umeans = None

        # compute centered transpose
        iur = uir.transpose()

        # L2-normalize ratings so dot product is cosine
        # if uir.values is None or not self.use_ratings:
        #     uir.values = np.full(uir.nnz, 1.0)
        # uir.normalize_rows('unit')
        start = time.time()
        # IDF: np.matrix
        self.idf = np.log(uir.nrows / iur.mult_vec(np.ones(uir.nrows)))

        end = time.time()
        print('IDF calculate time: ', end - start)

        self.rating_matrix_ = uir
        self.user_index_ = users
        self.user_means_ = umeans
        self.item_index_ = items
        self.transpose_matrix_ = iur

        return self

    def predict_for_user(self, user, items, ratings=None):
        """
        Compute predictions for a user and items.
        Args:
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series):
                the user's ratings (indexed by item id); if provided, will be used to
                recompute the user's bias at prediction time.
        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """
        watch = util.Stopwatch()
        items = pd.Index(items, name='item')

        ratings, umean = self._get_user_data(user, ratings)

        if ratings is None:
            return pd.Series(index=items, dtype='float64')
        # ratings is a dense vector
        assert len(ratings) == len(self.item_index_)

        # now ratings is normalized to be a mean-centered unit vector
        # this means we can dot product to score neighbors
        # score the neighbors!
        # similarity for every user
        start = time.time()
        nsims = self.rating_matrix_.mult_vec(ratings)
        assert len(nsims) == len(self.user_index_)
        # exclude self
        if user in self.user_index_:
            nsims[self.user_index_.get_loc(user)] = 0

        sim_index = np.argsort(-nsims)
        neighbors = sim_index[:self.nnbrs]
        nsims = nsims[:self.nnbrs]

        _logger.debug('computed user similarities')
        end = time.time()
        n_time = end - start

        # results = np.full(len(items), np.nan, dtype=np.float_)
        # ri_pos = self.item_index_.get_indexer(items.values)
        # if self.aggregate == self.AGG_WA:
        #     agg = _agg_weighted_avg
        # elif self.aggregate == self.AGG_SUM:
        #     # change here
        #     agg = _agg_sum
        # else:
        #     raise ValueError('invalid aggregate ' + self.aggregate)

        # _score(ri_pos, results, self.transpose_matrix_, nsims,
        #        self.nnbrs, self.min_sim, self.min_nbrs, agg)
        
        start = time.time()
        neighbor_score_matrix = self.rating_matrix_.pick_rows(neighbors)
        track_agg_cnt = neighbor_score_matrix.transpose().mult_vec(nsims)
        track_score = track_agg_cnt * self.idf
        candidates = np.argsort(-track_score)
        scores = track_score[candidates]
        end = time.time()
        s_time = end - start

        res_dict = {'track_id': candidates, 'confidence': scores}
        res = pd.DataFrame.from_dict(res_dict)

        # res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True)
        # res = res.reset_index(drop=True)

        res = res.head(500)
        # _logger.debug('scored %d of %d items for %s in %s',
        #               results.notna().sum(), len(items), user, watch)
        return res, n_time, s_time

    def _get_user_data(self, user, ratings):
        "Get a user's data for user-user CF"
        rmat = self.rating_matrix_

        if ratings is None:
            try:
                upos = self.user_index_.get_loc(user)
                ratings = rmat.row(upos)
                umean = self.user_means_[
                    upos] if self.user_means_ is not None else 0
            except KeyError:
                _logger.warning(
                    'user %d has no ratings and none provided', user)
                return None, 0
        else:
            _logger.debug('using provided ratings for user %d', user)
            if self.center:
                umean = ratings.mean()
                ratings = ratings - umean
            else:
                umean = 0
            unorm = np.linalg.norm(ratings)
            ratings = ratings / unorm
            ratings = ratings.reindex(self.item_index_, fill_value=0).values

        return ratings, umean

    def __getstate__(self):
        state = dict(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.aggregate = intern(self.aggregate)

    def __str__(self):
        return 'UserUser(nnbrs={}, min_sim={})'.format(self.nnbrs, self.min_sim)