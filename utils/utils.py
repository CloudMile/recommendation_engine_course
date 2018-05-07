import numpy as np, pandas as pd, pickle
from sklearn.metrics import roc_auc_score

import pandas as pd, numpy as np, os, tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

randomSeed = 88

def get_minibatches_idx(n, batch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def splitIndices(nTotal, ratio, shuffle=False):
    """split index by ratio"""
    assert type(ratio) in (tuple, list), "type of ratio must in (tuple, list)"

    lenAry = [int(nTotal * e) for e in ratio]
    offset = 0
    peice = []
    ary = np.arange(0, nTotal)
    if shuffle:
        ary = np.random.permutation(ary)
        
    for i, len_ in enumerate(lenAry):
        if i < len(lenAry) - 1:
            peice.append(ary[offset : offset + len_])
        # last round
        else:
            peice.append(ary[offset:])
        offset += len_
    return peice

def preview(fpath, chunksize=5, names=None):
    for chunk in pd.read_csv(fpath, chunksize=chunksize, names=names):
        return chunk

def loadPickle(fpath):
    with open(fpath, "rb") as r:
        return pickle.load(r)

def dumpPickle(fpath, obj):
    with open(fpath, "wb") as w:
        pickle.dump(obj, w)


from sklearn.base import BaseEstimator, TransformerMixin
class CounterEncoder(BaseEstimator, TransformerMixin):
    """依照出現頻率進行編碼, 頻率由高到低的index = 0, 1, 2, 3 ..., 以此類推"""
    def fit(self, y):
        counter = pd.Series(y).value_counts()
        self.enc = dict(zip([None] + counter.index.tolist(), range(len(counter) + 1)))
        self.invEnc = dict(zip(self.enc.values(), self.enc.keys()))
        self.classes_ = counter.index.values
        return self

    def transform(self, y):
        return pd.Series(y).map(self.enc).fillna(0).values

    def fit_transform(self, y, **fit_params):
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y):
        return pd.Series(y).map(self.invEnc).values

class OrderedMapper(CounterEncoder):
    def fit(self, y):
        uniq = pd.Series(y).unique()
        self.enc = dict(zip(uniq, range(len(uniq))))
        self.invEnc = dict(zip(range(len(uniq)), uniq))
        self.classes_ = uniq
        return self


from collections import defaultdict, Counter, OrderedDict

def split_ratings(data, pos_thres=4, testRatio=0.3):
    """依照比例切割movielens train test資料"""
    tr, te = [], []
    for u, df in data.groupby("userId"):
        if len(df) < 5: continue

        pos, neg = df.query("rating >= {}".format(pos_thres)), df.query("rating < {}".format(pos_thres))
        pos_len = int(len(pos) * (1 - testRatio))
        tr_pos = pos[:pos_len]
        te_pos = pos[pos_len:]

        neg_len = int(len(neg) * (1 - testRatio))
        tr_neg = neg[:neg_len]
        te_neg = neg[neg_len:]

        tr.append(tr_pos.append(tr_neg))
        te.append(te_pos.append(te_neg))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

def doMovies(movies):
    """處理 movie: genres 轉換成數字"""
    movies = movies.reset_index(drop=True)
    movies.loc[movies.genres == "(no genres listed)", "genres"] = ""
    movies["genres"] = movies.genres.str.split("\|")
    genresMap = Counter()
    movies.genres.map(genresMap.update)
    om = OrderedMapper().fit([e[0] for e in genresMap.most_common()])
    movies["genres"] = movies.genres.map(lambda lst: [om.enc[e] for e in lst])
    return movies, om


def auc_mean(y, pred_mat):
    """mean auc score of each user"""
    tot_auc, cnt = 0, 0
    for i in range(len(y)):
        nnz = y[i].nonzero()[0]
        if len(nnz) <= 1: continue

        labels = y[i][nnz]
        labels = (labels >= 4).astype(int)
        pred = pred_mat[i][nnz]
        if (labels == 1).all() or (labels == 0).all(): continue

        # print(i, ":", labels, predProba[i][nnz])
        tot_auc += roc_auc_score(labels, pred)
        cnt += 1
    return tot_auc / cnt


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def single_user_ndcg(label, score, label_thres=4, k=10):
    """single user ndcg score"""
    nnz = label.nonzero()[0]
    # if np.sum(label >= label_thres) < k: return None
    label, score = label[nnz], score[nnz]
    label = (label >= label_thres).astype(int)
    return ndcg_score(label, score, k)

def all_user_ndcg(label_mat, pred_mat, cond_fn, label_thres=4, k=10):
    """avg of all user ndcg score"""
    tot_ndcg, actual_cnt = 0, 0
    for i, (label, score) in enumerate(zip(label_mat, pred_mat)):
        if not cond_fn(label): continue

        ndcg = single_user_ndcg(label, score, k=10)
        if ndcg is not None:
            tot_ndcg += ndcg
            actual_cnt += 1
    return tot_ndcg / actual_cnt


datapath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data'

def prepare():
    ratings = pd.read_csv("{}/ml-latest-small/ratings.csv".format(datapath))
    movies = pd.read_csv("{}/ml-latest-small/movies.csv".format(datapath))

    uidEnc, midEnc = LabelEncoder(), LabelEncoder()
    # encode user id and movie id to real value
    midEnc.fit(movies.movieId)
    uidEnc.fit(ratings.userId)

    ratings["userId"] = uidEnc.transform(ratings.userId)
    ratings["movieId"] = midEnc.transform(ratings.movieId)

    movies["movieId"] = midEnc.transform(movies.movieId)

    midMap = pd.Series(dict(zip(movies.movieId, movies.title)))

    nUsers, nMovies = len(uidEnc.classes_), len(midEnc.classes_)

    tr = pd.read_csv("./data/ml-latest-small/movielens.tr.csv")
    te = pd.read_csv("./data/ml-latest-small/movielens.te.csv")

    # train data rating matrix
    trRatingMat = np.zeros((nUsers, nMovies))
    # test data rating matrix
    teRatingMat = np.zeros((nUsers, nMovies))
    for idx, r in tr.iterrows():
        trRatingMat[int(r.userId), int(r.movieId)] = r.rating
    for idx, r in te.iterrows():
        teRatingMat[int(r.userId), int(r.movieId)] = r.rating

    return (ratings, movies, uidEnc, midEnc, nUsers,
            nMovies, midMap, tr, te, trRatingMat, teRatingMat)


def loo_preprocess(data, movie_trans, train_hist=None, is_train=True):
    """以leave one out方式產生 train data, test data"""
    queue = []
    data = data.merge(movie_trans, how="left", on="movieId")
    columns=["user_id", "query_movie_ids",
             "genres", "avg_rating", "year", "candidate_movie_id",
             "rating"]
    for u, df in data.groupby("userId"):
        df = df.sort_values("rating", ascending=False)
        if not is_train:
            user_movies_hist = train_hist.query("userId == {}".format(u)).movieId
        for i, (_, r) in enumerate(df.iterrows()):
            if is_train:
                queue.append([int(r.userId),
                              df.movieId[:i].tolist() + df.movieId[i + 1:].tolist(),
                              r.genres, r.avg_rating, r.year, int(r.movieId), r.rating])
            else:
                # queue.append([int(r.userId), df.movieId[:i].tolist() + df.movieId[i + 1:].tolist(), r.genres, r.avg_rating, r.year, int(r.movieId), r.rating])
                # all_hist = set(user_movies_hist.tolist() + df.movieId[:i].tolist())
                all_hist = set(user_movies_hist.tolist())
                queue.append([int(r.userId),
                              list(all_hist - set([int(r.movieId)])),
                              r.genres, r.avg_rating, r.year, int(r.movieId), r.rating])
    return pd.DataFrame(queue, columns=columns)

def do_multi(df, multi_cols):
    """對於multivalent的欄位, 需要增加一個column去描述該欄位的長度"""
    pad = tf.keras.preprocessing.sequence.pad_sequences
    ret = OrderedDict()
    for colname, col in df.iteritems():
        if colname in multi_cols:
            lens = col.map(len)
            ret[colname] = list(pad(col, padding="post", maxlen=lens.max()))
            ret[colname + "_len"] = lens.values
        else:
            ret[colname] = col.values
    return ret


def user_data(data, uids, n_batch=128):
    u_col = ["user_id", "query_movie_ids", "candidate_movie_id"]
    cache = {"u_ary": []}

    def clear(u_ary):
        u_data = do_multi(pd.DataFrame(data=u_ary, columns=u_col), ["query_movie_ids"])
        cache["u_ary"] = []
        return u_data

    for uid, df in data[data.user_id.isin(uids)].groupby("user_id"):
        u_rec, u_ary = df.iloc[0], cache["u_ary"]
        # print(u_rec.query_movie_ids, u_rec.candidate_movie_id)
        u_rec.set_value("query_movie_ids", u_rec.query_movie_ids + [u_rec.candidate_movie_id])
        u_ary.append(u_rec[u_col].values)
        if len(u_ary) >= n_batch:
            yield clear(u_ary)
    yield clear(u_ary)


def user_item_data(data, uids, movie_trans, n_batch=128):
    u_col = ["user_id", "query_movie_ids"]
    cache = {"u_ary": []}
    items = do_multi(movie_trans, ["genres"])
    items["candidate_movie_id"] = items.pop("movieId")

    def clear(u_ary):
        u_data = do_multi(pd.DataFrame(data=u_ary, columns=u_col), ["query_movie_ids"])
        cache["u_ary"] = []
        return u_data

    for uid, df in data[data.user_id.isin(uids)].groupby("user_id"):
        u_rec, u_ary = df.iloc[0], cache["u_ary"]
        # print(u_rec.query_movie_ids, u_rec.candidate_movie_id)
        u_rec.set_value("query_movie_ids", u_rec.query_movie_ids + [u_rec.candidate_movie_id])
        u_ary.append(u_rec[u_col].values)
        if len(u_ary) >= n_batch:
            yield clear(u_ary), items
    yield clear(u_ary), items


def drawRocCurve(y, predProba):
    fprRf, tprRf, _ = roc_curve(y, predProba, pos_label=1)
    aucScr = auc(fprRf, tprRf)
    print("auc:", aucScr)
    f, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fprRf, tprRf, label='ROC CURVE')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Area Under Curve(ROC) (score: {:.4f})'.format(aucScr))
    ax.legend(loc='best')
    plt.show()


def strict_condition(label):
    label = label[label != 0]
    pos, neg = sum(label >= 4), sum(label < 4)
    return len(label) >= 10 and pos <= neg and pos > 0

def norm_condition(label):
    label = label[label != 0]
    return sum(label >= 4) > 0 and sum(label < 4) > 0

def precision_at_k(truth, pred_mat, condition_fn=None, k=10, label_thres=4):
    hits, total = 0, 0
    for label, pr in zip(truth, pred_mat):
        if not condition_fn(label): continue

        top_k_ind = (pr * (label != 0)).argsort()[::-1][:k]
        hits += sum(label[top_k_ind] >= label_thres)
        total += k
    return hits / total