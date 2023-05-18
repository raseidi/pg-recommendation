import os
os.chdir('../../')
import utils
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
pd.set_option('display.max_columns', 100)

mb = utils.read_int_mb()
mf = utils.read_mf()

META_TARGETS = ['IndexTime'] # Recall', 'DistComp', 'QueryTime', 'IndexTime']
REAL_BASES = [
    'texture_67940', 'sift_999900', 'moments_67940',
    'mnist121d_69900', 'fashion_69900',
    'colorHisto_67940', 'mnist_69900',
    'cophir282_999900', 'cophir64_999900'
]
# PREDICTIONS = pd.read_csv('data/results/clustering_predictions.csv')
# SCORES = pd.read_csv('data/results/clustering_scores.csv')
# CLUSTERS_INFO = pd.read_csv('data/results/clustering_info.csv')

# # Fixing IndexTime predictions
# PREDICTIONS = PREDICTIONS[PREDICTIONS.target != 'IndexTime']
# PREDICTIONS = PREDICTIONS[PREDICTIONS.target != 'IndexTime']
# CLUSTERS_INFO = CLUSTERS_INFO[CLUSTERS_INFO.target != 'IndexTime']
PREDICTIONS = pd.DataFrame()
SCORES = pd.DataFrame()
CLUSTERS_INFO = pd.DataFrame()

for fs_method in ['pearson', 'rf', 'pca']:
    prods = product(REAL_BASES, META_TARGETS)
    for base, target in prods:
        print(base, target)
        features = utils.get_features_for_similarity(target, fs_method=fs_method, mb=mb.copy())
        eps = utils.get_eps(mf, features, 0.8)
        eps = 0.3 if target == 'Recall' else eps
        #
        mf_new = mf[features].copy()
        mf_new.loc[:, :] = StandardScaler().fit_transform(mf_new)
        train = mf_new[~mf_new.index.str.startswith(base.split('_')[0])].copy()
        clus = DBSCAN(eps=eps, min_samples=2).fit(train)
        train['label'] = clus.labels_
        
        tmp = pd.DataFrame({
            'base': [base],
            'fs_method': [fs_method],
            'target': [target],
            'n_clusters': [len(train[train.label != -1].label.unique())],
            'n_outliers': [len(train[train.label == -1])],
            'mean_datatsets_per_cluster': [train[train.label != -1].label.value_counts().mean()],
            'std_datatsets_per_cluster': [train[train.label != -1].label.value_counts().std()],
            'max_datasets_per_cluster': [train[train.label != -1].label.value_counts().max()],
            'min_datasets_per_cluster': [train[train.label != -1].label.value_counts().min()],
        })
        CLUSTERS_INFO = pd.concat([CLUSTERS_INFO, tmp])        
        # CLUSTERS_INFO.groupby('fs_method').n_clusters.describe()[['mean', 'std', 'min', 'max']]
        # CLUSTERS_INFO.groupby('fs_method')['mean_datatsets_per_cluster', 'std_datatsets_per_cluster', 'min_datasets_per_cluster', 'max_datasets_per_cluster'].mean()

        # CLUSTERS_INFO.rename(columns={
        #     'mean_datatsets_per_cluster': 'mean.mean',
        #     'std_datatsets_per_cluster': 'std.mean',
        #     'max_datasets_per_cluster': 'max.mean',
        #     'min_datasets_per_cluster': 'min.mean',
        # }).groupby('fs_method').mean().round(4).to_csv('~/Desktop/tmp.csv')
        #

        X_train, X_test, y_train, y_test = utils.clustering_learning(mb, mf, base, features, target, eps)

        models = utils.fit_models(
            X_train, y_train,
            model=RandomForestRegressor,
            n_models=10
        )
        y_pred = utils.ensamble_predictions(models, X_test)

        kwargs = {
            'feature_selection_method': fs_method,
            'method': f'clustering_eps={eps}',
            'target': target,
            'cluster_size': len(X_train.index.unique()),
        }
        tmp = utils.format_scores(y_test, y_pred, **kwargs)
        SCORES = pd.concat([SCORES, tmp])

        kwargs = {
            'target': target,
            'feature_selection_method': fs_method,
        }
        if 'k_searching' not in X_test.columns:
            X_test['k_searching'] = 30
        tmp = utils.format_predictions(X_test, y_test, y_pred, **kwargs)
        PREDICTIONS = pd.concat([PREDICTIONS, tmp])

        print('Done.')
        SCORES.to_csv('data/results/clustering_scores.csv', index=False)
        CLUSTERS_INFO.to_csv('data/results/clustering_info.csv', index=False)
        PREDICTIONS.to_csv('data/results/clustering_predictions.csv', index=False)

