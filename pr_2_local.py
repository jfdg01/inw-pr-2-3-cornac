# -*- coding: utf-8 -*-
"""
Práctica 2 — Evaluación exhaustiva de SRCF y contenido

Algoritmos evaluados:
- UserKNN (KNNWithMeans) k ∈ {10, 20, 30} × similitud ∈ {coseno, pearson}
- ItemKNN (KNNWithMeans) k ∈ {10, 20, 30} × similitud ∈ {coseno, pearson}
- SVD factores ∈ {5, 10, 20, 30}
- SVD++ factores ∈ {5, 10, 20, 30}
- Recomendador basado en contenido (TF-IDF sobre tags, Práctica 1)

Métricas: MAE, Precision@N, Recall@N, F1@N, NDCG@N para N ∈ {3, 5, 10}
Umbral relevancia: rating ≥ 3.5 · Split 80/20 · Semilla 1
"""

## ── 1. Imports y rutas ────────────────────────────────────────────────────
import warnings
from collections import defaultdict
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import cornac
from cornac.data import Reader
from cornac.eval_methods import RatioSplit
from cornac.models import UserKNN, ItemKNN, SVD, Recommender
from cornac.metrics import MAE, Precision, Recall, FMeasure, NDCG

try:
    from cornac.models import SVDpp
    HAS_SVDPP = True
except ImportError:
    HAS_SVDPP = False
    print('SVDpp no disponible en esta version de cornac — se omitira')

warnings.filterwarnings('ignore')

BASE              = os.path.join(os.path.dirname(__file__), 'rs-movie-cour')
PATH_RATINGS      = os.path.join(BASE, 'ratings.csv')
PATH_MOVIE_TITLES = os.path.join(BASE, 'movie-titles.csv')
PATH_MOVIE_TAGS   = os.path.join(BASE, 'movie-tags.csv')

THRESHOLD = 3.5
N_VALUES  = [3, 5, 10]
SEED      = 1

print('Librerias listas.')

## ── 2. Carga de datos y split 80/20 ──────────────────────────────────────
reader       = Reader()
ratings_data = reader.read(PATH_RATINGS, sep=',', skip_lines=1)

eval_method = RatioSplit(
    data=ratings_data,
    test_size=0.2,
    rating_threshold=THRESHOLD,
    exclude_unknowns=True,
    verbose=True,
    seed=SEED,
)

## ── 3. Recomendador basado en contenido (Practica 1) ─────────────────────
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

        n_with_tags = (items_df['tag_text'] != '').sum()
        print(f'[CB-TF-IDF] Perfiles: {len(self._user_profiles)}/{train_set.num_users} usuarios | '
              f'{n_with_tags}/{n_items} items con tags')
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

print('Clase ContentBasedTFIDF definida.')

## ── 4. Definicion de todos los modelos ───────────────────────────────────
all_models = []

for k in [10, 20, 30]:
    for sim in ['cosine', 'pearson']:
        all_models.append(
            UserKNN(k=k, similarity=sim,
                    name=f'UserKNN k={k} {sim}', seed=SEED)
        )

for k in [10, 20, 30]:
    for sim in ['cosine', 'pearson']:
        all_models.append(
            ItemKNN(k=k, similarity=sim,
                    name=f'ItemKNN k={k} {sim}', seed=SEED)
        )

for n_factors in [5, 10, 20, 30]:
    all_models.append(SVD(k=n_factors, name=f'SVD   f={n_factors}', seed=SEED))

if HAS_SVDPP:
    for n_factors in [5, 10, 20, 30]:
        all_models.append(SVDpp(k=n_factors, name=f'SVD++ f={n_factors}', seed=SEED))

all_models.append(
    ContentBasedTFIDF(tags_path=PATH_MOVIE_TAGS, titles_path=PATH_MOVIE_TITLES)
)

print(f'Total de modelos a evaluar: {len(all_models)}')
for m in all_models:
    print(f'  {m.name}')

## ── 5. Metricas y ejecucion del experimento (con cache por modelo) ────────
import pickle

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'results_cache.pkl')
metric_names = [
    'MAE',
    'Precision@3', 'Recall@3', 'F1@3', 'NDCG@3',
    'Precision@5', 'Recall@5', 'F1@5', 'NDCG@5',
    'Precision@10','Recall@10','F1@10','NDCG@10',
]

all_metrics = [
    MAE(),
    Precision(k=3),  Recall(k=3),  FMeasure(k=3),  NDCG(k=3),
    Precision(k=5),  Recall(k=5),  FMeasure(k=5),  NDCG(k=5),
    Precision(k=10), Recall(k=10), FMeasure(k=10), NDCG(k=10),
]

# Load cache if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        cached_rows = pickle.load(f)
    print(f'Cache cargado: {len(cached_rows)} modelos ya evaluados.')
else:
    cached_rows = {}

rows_ordered = []
for model in all_models:
    if model.name in cached_rows:
        print(f'[{model.name}] Saltado (resultado en cache).')
        rows_ordered.append(cached_rows[model.name])
        continue

    exp = cornac.Experiment(
        eval_method=eval_method,
        models=[model],
        metrics=all_metrics,
        verbose=True,
    )
    exp.run()

    res = exp.result[0]
    row = {'Modelo': res.model_name}
    for mn in metric_names:
        try:
            row[mn] = round(res.metric_avg_results[mn], 4)
        except Exception:
            row[mn] = float('nan')

    cached_rows[model.name] = row
    rows_ordered.append(row)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cached_rows, f)
    print(f'[{model.name}] Resultado guardado en cache.')

rows = rows_ordered

df = pd.DataFrame(rows).set_index('Modelo')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

print('\n' + '='*80)
print('TABLA COMPARATIVA COMPLETA')
print('='*80)
print(df.to_string())

## ── 7. Subtablas por familia de modelo ───────────────────────────────────
families = {
    'UserKNN':   df[df.index.str.startswith('UserKNN')],
    'ItemKNN':   df[df.index.str.startswith('ItemKNN')],
    'SVD  ':     df[df.index.str.startswith('SVD   ')],
    'SVD++':     df[df.index.str.startswith('SVD++')],
    'Contenido': df[df.index.str.startswith('CB')],
}

for fname, fdf in families.items():
    if fdf.empty:
        continue
    print(f"\n{'='*62}\n  {fname.strip()}\n{'='*62}")
    print(fdf.to_string())

## ── 8. Grafico comparativo F1@10 y NDCG@10 ──────────────────────────────
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def model_color(name):
    if name.startswith('UserKNN'): return '#4a90d9'
    if name.startswith('ItemKNN'): return '#e67e22'
    if name.startswith('SVD   '): return '#27ae60'
    if name.startswith('SVD++'): return '#8e44ad'
    return '#c0392b'

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, metric in zip(axes, ['F1@10', 'NDCG@10']):
    vals   = df[metric].sort_values(ascending=False)
    colors = [model_color(i) for i in vals.index]
    bars   = ax.barh(vals.index, vals.values, color=colors)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Comparativa {metric}', fontsize=13)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.invert_yaxis()

legend_elements = [
    Patch(facecolor='#4a90d9', label='UserKNN'),
    Patch(facecolor='#e67e22', label='ItemKNN'),
    Patch(facecolor='#27ae60', label='SVD'),
    Patch(facecolor='#8e44ad', label='SVD++'),
    Patch(facecolor='#c0392b', label='Contenido'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'comparativa_f1_ndcg.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print('Grafico guardado como comparativa_f1_ndcg.png')
