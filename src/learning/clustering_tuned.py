import os
import time
import utils
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)

def get_train(tmp, base, method):
    if method in ['gmm', 'tmmgs', 'tmmgss']:
        train = tmp[~tmp.index.str.startswith(base.split('_')[0] + '_')].copy()
    elif method == 'tmms':
        train = tmp[tmp.index != base].copy()
        
    return train

def get_similar_bases(mf, base, features, k=5, method='gmm'):
    tmp = mf.copy()
    tmp = tmp[features]

    train = tmp[~tmp.index.str.startswith(base.split('_')[0] + '_')].copy()
    test = tmp[tmp.index == base].copy()
    sc = StandardScaler()
    sc.fit(train)
    train.loc[:, :] = sc.transform(train)
    test.loc[:, :] = sc.transform(test)

    knng = NearestNeighbors(n_neighbors=k)
    knng.fit(train)
    dist, ix = knng.kneighbors(test)
    bases = train.iloc[ix.reshape(-1)].index.values
    
    return bases

def clustering_learning(mb, mf, base, features, target, eps, method='tmms'):
    drop_bases = set(mf.index) - set(mb.index)
    mf = mf[~mf.index.isin(drop_bases)]
    features = list(set(features))
    
    mf_new = mf[features].copy()
    mf_new.loc[:, :] = StandardScaler().fit_transform(mf_new)
    train = mf_new[~mf_new.index.str.startswith(base.split('_')[0] + '_')].copy()
    clus = DBSCAN(eps=eps, min_samples=2).fit(train)
    train['label'] = clus.labels_
    
    test = mf_new[mf_new.index == base].copy()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train.drop('label', axis=1), train.label)
    pred = knn.predict(test)
    if pred == -1:
        k = int(train[train.label != -1].label.value_counts().mean())
        train_bases = get_similar_bases(mf, base, features, k=k, method=method)
    else:
        train_bases = train[train.label == pred[0]].index.values
    
    clus_info = pd.DataFrame({
        'base': [base],
        'outlier': True if pred == -1 else False,
        'method': method,
        'target': [target],
        'cluster_size': len(train_bases),
        'n_clusters': [len(train[train.label != -1].label.unique())],
        'n_outliers': [len(train[train.label == -1])],
        'mean_datatsets_per_cluster': [train[train.label != -1].label.value_counts().mean()],
        'std_datatsets_per_cluster': [train[train.label != -1].label.value_counts().std()],
        'max_datasets_per_cluster': [train[train.label != -1].label.value_counts().max()],
        'min_datasets_per_cluster': [train[train.label != -1].label.value_counts().min()],
    })

    df = mb.copy()
    df.reset_index(inplace=True)
    if method == 'tmmgs':
        not_gs = df[
            (df.base == base) &
            ((~df.IndexParams.isin([5., 25., 70., 100.])) |
            (~df.QueryTimeParams.isin([1., 10., 40., 120.])))
        ].index.values
        df.drop(not_gs, axis=0, inplace=True)
        train_bases = list(set(np.append(train_bases, base)))
    elif method == 'tmmgss':
        subset = sorted(df[df.base.str.startswith(base.split('_')[0])].base.unique())
        subset = sorted(list(set(map(lambda x: int(x.split('_')[1]), subset))))[-2]
        subset = base.split('_')[0] + '_' + str(subset)
        not_gs = df[
            (df.base == subset) &
            ((~df.IndexParams.isin([5., 25., 70., 100.])) |
            (~df.QueryTimeParams.isin([1., 10., 40., 120.])))
        ].index.values
        df.drop(not_gs, axis=0, inplace=True)
        train_bases = list(set(np.append(train_bases, subset)))
    elif method == 'tmms':
        subsets = sorted(df[df.base.str.startswith(base.split('_')[0])].base.unique())
        subsets.remove(base)
        train_bases = list(set(np.append(train_bases, subsets)))

    df.set_index('base', inplace=True)
    X_train = df.loc[(df.index.isin(train_bases)), ~(df.columns.isin(META_TARGETS))]
    y_train = df[df.index.isin(train_bases)][target]
    X_test = mb.loc[(mb.index == base), ~(mb.columns.isin(META_TARGETS))]
    y_test = mb[mb.index == base][target]

    return X_train, X_test, y_train, y_test, clus_info

def check_if_done(approach, base_target, k, metatarget, SCORES):
    return len(SCORES[
        (SCORES.method == approach) &
        (SCORES.base == base_target) &
        (SCORES.k_searching == k) &
        (SCORES.target == metatarget)
    ])

mb = utils.read_int_mb()
mb = mb[mb.k_searching != 5]
mb = mb[~mb.QueryTimeParams.isin([220, 240, 2, 1, 3, 4])]
# mb.drop('k_searching', axis=1, inplace=True)
mf = utils.read_mf()

META_TARGETS = ['Recall', 'DistComp', 'QueryTime', 'IndexTime']
REAL_BASES = [
    # 'texture_67940', 'sift_999900', 'moments_67940',
    # 'mnist121d_69900', 'fashion_69900',
    # 'colorHisto_67940', 'mnist_69900', 
    # 'cophir282_999900', 'cophir64_999900', 'base71_999900',

    'texture_32000', 'moments_32000', 'colorHisto_32000',
    'mnist_32000', 'mnist121d_32000', 'fashion_32000',
    'sift_500000', 'cophir64_500000', 'cophir282_500000', 'base71_500000',

    'texture_16000', 'moments_16000', 'colorHisto_16000',
    'mnist_16000', 'mnist121d_16000', 'fashion_16000',
    'sift_100000', 'cophir64_100000', 'cophir282_100000', 'base71_100000',
]

METHODS = ['tmmgs', 'gmm', 'tmms']
K_SEARCHING = mb.k_searching.unique()

SCORES = pd.read_csv('data/results/info_sys_interpolated/clustering_tuned/scores.csv')
CLUSTERS_INFO = pd.read_csv('data/results/info_sys_interpolated/clustering_tuned/info.csv')
PREDICTIONS = pd.read_csv('data/results/info_sys_interpolated/clustering_tuned/predictions.csv')

# PREDICTIONS = pd.DataFrame()
# SCORES = pd.DataFrame()
# CLUSTERS_INFO = pd.DataFrame()

fs_method = 'rf'
_mb = mb.copy()
prods = product(REAL_BASES, META_TARGETS, METHODS, K_SEARCHING)
for base, target, method, k in prods:
    if check_if_done(method, base, k, target, SCORES) or target == 'IndexTime':
        continue
    
    mb = _mb.copy()
    mb = mb[mb.k_searching == k]
    if mb[mb.index == base].empty:
        continue
    if base.startswith('base'): # brute-knng were not runned for all synthetic datasets, resulting in poor recommendations
        mb = mb[mb.graph_type != 2]
    if base.split('_')[-1] == '999900': # 1M datasets have very small subsets, thus we train only the largest ones (0.1M and 0.5M)
        ns = sorted(mb[mb.index.str.startswith(base.split('_')[0])].nr_inst.unique())[:-3]
        bs = [base.split('_')[0] + f'_{i}' for i in np.round(np.exp(ns)).astype(int)]
        mb = mb[~mb.index.isin(bs)]

    features = utils.get_features_for_similarity(target, fs_method=fs_method, mb=mb.copy())
    eps = utils.get_eps(mf, features, 0.9)
    eps = 0.3 if target == 'Recall' else eps

    X_train, X_test, y_train, y_test, clus_info = clustering_learning(mb, mf, base, features, target, eps, method=method)
    if len(X_train) == 0: # 2
        print('2', method, base, k, target)
        continue
    clus_info['k_searching'] = k
    CLUSTERS_INFO = pd.concat([CLUSTERS_INFO, clus_info])

    start = time.time()
    models = utils.fit_models(
        X_train, y_train,
        model=RandomForestRegressor,
        n_models=5
    )
    elapsed_training = time.time() - start

    start = time.time()
    y_pred = utils.ensamble_predictions(models, X_test)
    elapsed_inference = time.time() - start

    kwargs = {
        'feature_selection_method': fs_method,
        'eps': eps,
        'method': method,
        'target': target,
        'cluster_size': len(X_train.index.unique()),
        'elapsed_training': elapsed_training,
        'elapsed_inference': elapsed_inference,
    }
    tmp = utils.format_scores(y_test, y_pred, **kwargs)
    tmp['k_searching'] = k
    SCORES = pd.concat([SCORES, tmp])

    kwargs = {
        'target': target,
        'method': method,
        'feature_selection_method': fs_method,
    }
    if 'k_searching' not in X_test.columns:
        X_test['k_searching'] = 30
    else:
        X_test['k_searching'] = k
    tmp = utils.format_predictions(X_test, y_test, y_pred, **kwargs)
    PREDICTIONS = pd.concat([PREDICTIONS, tmp])

    SCORES.to_csv('data/results/info_sys_interpolated/clustering_tuned/scores.csv', index=False)
    CLUSTERS_INFO.to_csv('data/results/info_sys_interpolated/clustering_tuned/info.csv', index=False)
    PREDICTIONS.to_csv('data/results/info_sys_interpolated/clustering_tuned/predictions.csv', index=False)
