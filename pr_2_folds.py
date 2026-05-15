# -*- coding: utf-8 -*-
"""
pr_2_folds.py
5-fold cross-validation using the pre-provided splits.
Models: UserKNN, ItemKNN, SVD (cornac), CB-TF-IDF
Metrics: MAE, RMSE, Precision@N, Recall@N, F1@N, NDCG@N  (N in {3, 5, 10})
Results cached per model per fold — safe to interrupt and resume.
"""

import os, pickle, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import cornac
from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.models import UserKNN, ItemKNN, SVD, Recommender
from cornac.metrics import MAE, RMSE, Precision, Recall, FMeasure, NDCG

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# SVD++ (cornac lacks it; implemented here via SGD)
# r̂(u,i) = μ + b_u + b_i + q_i^T (p_u + |I_u|^(-1/2) Σ_{j∈I_u} y_j)
# ══════════════════════════════════════════════════════════════════════════
class SVDpp(Recommender):

    def __init__(self, k=20, n_epochs=20, lr=0.005, reg=0.02, name=None, seed=None):
        super().__init__(name=name or f'SVDpp k={k}', trainable=True)
        self.k = k
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        rng = np.random.RandomState(self.seed)

        n_users = train_set.num_users
        n_items = train_set.num_items
        mu = train_set.global_mean
        k, lr, reg = self.k, self.lr, self.reg

        P  = rng.normal(0, 0.1, (n_users, k))
        Q  = rng.normal(0, 0.1, (n_items, k))
        Y  = rng.normal(0, 0.1, (n_items, k))
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)

        u_arr, i_arr, r_arr = train_set.uir_tuple
        u_arr = u_arr.astype(np.int32)
        i_arr = i_arr.astype(np.int32)

        # items rated by each user as numpy arrays (for fast sum)
        user_items = defaultdict(list)
        for u, i in zip(u_arr, i_arr):
            user_items[int(u)].append(int(i))
        user_items_np = {u: np.array(v, dtype=np.int32) for u, v in user_items.items()}

        indices = np.arange(len(u_arr))
        for _ in range(self.n_epochs):
            rng.shuffle(indices)
            for idx in indices:
                u = int(u_arr[idx])
                i = int(i_arr[idx])
                r = float(r_arr[idx])

                Iu = user_items_np.get(u)
                if Iu is not None and len(Iu) > 0:
                    sqrt_Iu = 1.0 / np.sqrt(len(Iu))
                    sum_y = Y[Iu].sum(axis=0)
                else:
                    sqrt_Iu = 1.0
                    sum_y = np.zeros(k)

                p_tilde = P[u] + sqrt_Iu * sum_y
                e = r - (mu + bu[u] + bi[i] + Q[i].dot(p_tilde))
                elr = lr * e

                bu[u] += elr - lr * reg * bu[u]
                bi[i] += elr - lr * reg * bi[i]

                Qi = Q[i].copy()
                Q[i]  += elr * p_tilde - lr * reg * Q[i]
                P[u]  += elr * Qi - lr * reg * P[u]
                if Iu is not None and len(Iu) > 0:
                    Y[Iu] += elr * sqrt_Iu * Qi - lr * reg * Y[Iu]

        self.mu, self.P, self.Q, self.Y = mu, P, Q, Y
        self.bu, self.bi = bu, bi
        self.user_items_np = user_items_np
        self._min_r, self._max_r = train_set.min_rating, train_set.max_rating
        return self

    def score(self, user_idx, item_idx=None):
        Iu = self.user_items_np.get(user_idx)
        sqrt_Iu = 1.0 / np.sqrt(len(Iu)) if (Iu is not None and len(Iu) > 0) else 1.0
        sum_y = self.Y[Iu].sum(axis=0) if (Iu is not None and len(Iu) > 0) else np.zeros(self.k)
        p_tilde = self.P[user_idx] + sqrt_Iu * sum_y
        if item_idx is None:
            scores = self.mu + self.bu[user_idx] + self.bi + self.Q.dot(p_tilde)
            return np.clip(scores, self._min_r, self._max_r)
        s = self.mu + self.bu[user_idx] + self.bi[item_idx] + self.Q[item_idx].dot(p_tilde)
        return float(np.clip(s, self._min_r, self._max_r))

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rs-movie-cour')
FOLDS_DIR = os.path.join(BASE, 'rs-cour-dataset-validation')
PATH_TAGS   = os.path.join(BASE, 'movie-tags.csv')
PATH_TITLES = os.path.join(BASE, 'movie-titles.csv')

THRESHOLD  = 3.5
SEED       = 1
N_FOLDS    = 5
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_folds_cache.pkl')

METRIC_NAMES = [
    'MAE', 'RMSE',
    'Precision@3', 'Recall@3', 'F1@3', 'NDCG@3',
    'Precision@5', 'Recall@5', 'F1@5', 'NDCG@5',
    'Precision@10', 'Recall@10', 'F1@10', 'NDCG@10',
]

# ══════════════════════════════════════════════════════════════════════════
# Content-based TF-IDF recommender
# ══════════════════════════════════════════════════════════════════════════
class ContentBasedTFIDF(Recommender):

    def __init__(self, tags_path, titles_path, name='CB-TF-IDF'):
        super().__init__(name=name, trainable=False)
        self.tags_path   = tags_path
        self.titles_path = titles_path

    @staticmethod
    def _detect_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return df.columns[0]

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        tags_df = pd.read_csv(self.tags_path, encoding='latin-1')
        tag_id  = self._detect_col(tags_df, ('movieId', 'movie_id', 'id', 'item_id'))
        tag_col = next((c for c in tags_df.columns if 'tag' in c.lower()), tags_df.columns[1])

        tags_agg = (
            tags_df.groupby(tag_id)[tag_col]
            .apply(lambda ts: ' '.join(str(t).lower().strip() for t in ts.dropna()))
            .reset_index()
            .rename(columns={tag_id: 'movieId', tag_col: 'tag_text'})
        )
        tags_agg['movieId'] = tags_agg['movieId'].astype(str)

        iid_inv = {v: k for k, v in train_set.iid_map.items()}
        n_items = train_set.num_items
        raw_ids = [str(iid_inv[i]) for i in range(n_items)]

        items_df = pd.DataFrame({'movieId': raw_ids})
        items_df = items_df.merge(tags_agg, on='movieId', how='left')
        items_df['tag_text'] = items_df['tag_text'].fillna('')

        tfidf = TfidfVectorizer(min_df=1, max_features=5000)
        self._item_mat = tfidf.fit_transform(items_df['tag_text'])

        u_arr, i_arr, r_arr = train_set.uir_tuple
        user_items = defaultdict(list)
        for u, i, r in zip(u_arr, i_arr, r_arr):
            user_items[int(u)].append((int(i), float(r)))

        self._user_profiles = {}
        self._user_items    = user_items
        self._global_mean   = train_set.global_mean
        self._rating_scale  = (train_set.min_rating, train_set.max_rating)

        for u_idx, rated in user_items.items():
            idxs = [x[0] for x in rated]
            ws   = np.array([x[1] for x in rated], dtype=float)
            vecs = self._item_mat[idxs].toarray()
            self._user_profiles[u_idx] = np.average(vecs, axis=0, weights=ws)

        return self

    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            return np.array([self._predict(user_idx, i)
                             for i in range(self.train_set.num_items)])
        return self._predict(user_idx, item_idx)

    def _predict(self, u_idx, i_idx):
        mn, mx = self._rating_scale
        if u_idx not in self._user_profiles:
            return self._global_mean
        t_vec  = self._item_mat[i_idx].toarray().ravel()
        norm_t = np.linalg.norm(t_vec)
        if norm_t == 0:
            return self._global_mean
        num, den = 0.0, 0.0
        for j_idx, r in self._user_items[u_idx]:
            j_vec  = self._item_mat[j_idx].toarray().ravel()
            norm_j = np.linalg.norm(j_vec)
            if norm_j == 0:
                continue
            s = float(np.dot(t_vec, j_vec) / (norm_t * norm_j))
            if s > 0:
                num += s * r
                den += s
        if den == 0:
            profile = self._user_profiles[u_idx]
            norm_p  = np.linalg.norm(profile)
            if norm_p == 0:
                return self._global_mean
            sim = float(np.dot(profile, t_vec) / (norm_p * norm_t))
            return float(np.clip(mn + sim * (mx - mn), mn, mx))
        return float(np.clip(num / den, mn, mx))


# ══════════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════════
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    fold_raw = cache.get('fold_raw', defaultdict(list))
    done = {k: len(v) for k, v in fold_raw.items() if v}
    print(f'Cache loaded. Completed folds per model: {done}')
else:
    cache    = {}
    fold_raw = defaultdict(list)

reader = Reader()

ALL_METRICS = [
    MAE(), RMSE(),
    Precision(k=3), Recall(k=3), FMeasure(k=3), NDCG(k=3),
    Precision(k=5), Recall(k=5), FMeasure(k=5), NDCG(k=5),
    Precision(k=10), Recall(k=10), FMeasure(k=10), NDCG(k=10),
]

def make_models():
    models = []
    for k in [10, 20, 30]:
        for sim in ['cosine', 'pearson']:
            models.append(UserKNN(k=k, similarity=sim, name=f'UserKNN k={k} {sim}', seed=SEED))
            models.append(ItemKNN(k=k, similarity=sim, name=f'ItemKNN k={k} {sim}', seed=SEED))
    for nf in [5, 10, 20, 30]:
        models.append(SVD(k=nf, name=f'SVD f={nf}', seed=SEED))
    for nf in [5, 10, 20, 30]:
        models.append(SVDpp(k=nf, name=f'SVDpp f={nf}', seed=SEED))
    models.append(ContentBasedTFIDF(tags_path=PATH_TAGS, titles_path=PATH_TITLES))
    return models


# ══════════════════════════════════════════════════════════════════════════
# 5-fold evaluation loop
# ══════════════════════════════════════════════════════════════════════════
for fold in range(N_FOLDS):
    print(f'\n{"="*62}\n  FOLD {fold+1}/{N_FOLDS}\n{"="*62}')

    train_raw = reader.read(os.path.join(FOLDS_DIR, f'ratings_train_{fold}.csv'), sep=',', skip_lines=1)
    test_raw  = reader.read(os.path.join(FOLDS_DIR, f'ratings_test_{fold}.csv'),  sep=',', skip_lines=1)

    eval_method = BaseMethod.from_splits(
        train_data=train_raw,
        test_data=test_raw,
        fmt='UIR',
        rating_threshold=THRESHOLD,
        exclude_unknowns=True,
        seed=SEED,
        verbose=False,
    )

    for model in make_models():
        name = model.name
        if len(fold_raw.get(name, [])) > fold:
            print(f'  [{name}] fold {fold+1} — cached.')
            continue

        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=ALL_METRICS,
            verbose=True,
        )
        exp.run()

        res = exp.result[0].metric_avg_results
        row = {mn: round(float(res.get(mn, float('nan'))), 4) for mn in METRIC_NAMES}

        fold_raw[name].append(row)
        cache['fold_raw'] = fold_raw
        with open(CACHE_FILE, 'wb') as fh:
            pickle.dump(cache, fh)
        print(f'  [{name}] fold {fold+1} — done.')


# ══════════════════════════════════════════════════════════════════════════
# Average across folds
# ══════════════════════════════════════════════════════════════════════════
print(f'\n{"="*62}\n  RESULTADOS FINALES (media 5-fold)\n{"="*62}')

rows = []
for name, fold_results in fold_raw.items():
    if len(fold_results) < N_FOLDS:
        print(f'  WARNING: {name} has {len(fold_results)}/{N_FOLDS} folds — skipping.')
        continue
    row = {'Modelo': name}
    for mn in METRIC_NAMES:
        vals  = [fm.get(mn, float('nan')) for fm in fold_results[:N_FOLDS]]
        valid = [v for v in vals if not np.isnan(v)]
        row[mn] = round(float(np.mean(valid)), 4) if valid else float('nan')
    rows.append(row)

def _sort_key(r):
    n = r['Modelo']
    for i, f in enumerate(['UserKNN', 'ItemKNN', 'SVD f', 'CB']):
        if n.startswith(f):
            return (i, n)
    return (99, n)
rows.sort(key=_sort_key)

df = pd.DataFrame(rows).set_index('Modelo')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 240)
pd.set_option('display.float_format', '{:.4f}'.format)
print('\n' + df.to_string())

cache['averaged'] = {r['Modelo']: {k: v for k, v in r.items() if k != 'Modelo'} for r in rows}
with open(CACHE_FILE, 'wb') as fh:
    pickle.dump(cache, fh)
print(f'\nSaved to {CACHE_FILE}')
