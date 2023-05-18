import os
os.chdir('../../')
import utils
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

mb = utils.read_mb()
mf = utils.read_mf()
not_metafeatures = ['Recall', 'IndexParams', 'QueryTimeParams', 'graph_type', 'k_searching']

# Pearson correlation, where features lie in quantile >= 0.9
pearson_corr = mb.corr().Recall.abs().sort_values(ascending=False)
pearson_corr.drop(not_metafeatures, inplace=True)
pearson_corr[pearson_corr > pearson_corr.quantile(.9)].index
PEARSON_FEATURES = ['lid_mean', 'rv', 'attr_ent.mean', 'min.mean', 'nr_inst']

# PCA evaluation, for the first n features 
# that sum 0.9 of the explained variance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
tmp = mb.copy()
sc = StandardScaler()
tmp.loc[:,:] = sc.fit_transform(tmp)

pca = PCA()
pca.fit(tmp.values)
xvar = pd.DataFrame({
    'ratio': np.abs(pca.explained_variance_ratio_),
    'feature': tmp.columns
}).sort_values(by='ratio', ascending=False)

var_sum = 0
for ix, f in enumerate(xvar.feature.unique()):
    var_sum += xvar[xvar.feature==f].ratio.values[0]
    if var_sum >= .95:
        PCA_FEATURES = xvar.head(ix)[~xvar.feature.isin(not_metafeatures)].feature.values
        break

# RF importance, where features lie in quantile >= 0.9
fi = pd.read_csv('data/feature_importances.csv')
fi.importance = fi.importance/fi.importance.sum()
RF_FEATURES= fi[fi.importance >= fi.importance.quantile(.9)].feature.values

# Measuring similarity according each fs method
fs_methods = {
    # 'pearson': PEARSON_FEATURES,
    # 'pca': PCA_FEATURES,
    'rf': RF_FEATURES
}

out['rf'].keys()
out = dict()
for key, array in fs_methods.items():
    out[key] = utils.get_similar(mf, array)

print(*out.items(), sep='\n\n\n\n')

'''
# Learning based on the dataset similarity for each method
SCORES = pd.DataFrame()
PREDICTIONS = pd.DataFrame()
for method in out.keys():
    for base, similars in out[method].items():
        X_train, X_test, y_train, y_test = utils.similarity_learning(mb, base, similars, 'Recall')
        # X_train, X_test, y_train, y_test = utils.clustering_learning(mb, mf, base, features, target, eps)
        models = utils.fit_models(
            X_train, y_train, 
            model=RandomForestRegressor,
            n_models=10
        )
        y_pred = utils.ensamble_predictions(models, X_test)

        kwargs = {
            'feature_selection_method': method,
            'method': 'similarity_k=5',
            'target': 'Recall'
        }
        tmp = utils.format_scores(y_test, y_pred, **kwargs)
        SCORES = pd.concat([SCORES, tmp])

        kwargs = {
            'method': 'similarity_k=5',
            'target': 'Recall',
            'feature_selection_method': method
        }
        tmp = utils.format_predictions(X_test, y_test, y_pred, **kwargs)
        PREDICTIONS = pd.concat([PREDICTIONS, tmp])

        print(f'Base={base} method={method}')

# SCORES.groupby('feature_selection_method').r2.describe()[['mean', 'std', 'min', 'max']]
SCORES.to_csv('results/similarity_scores', index=False)
PREDICTIONS.to_csv('results/similarity_predictions', index=False)
'''
# Learning from clustering approach
from sklearn.cluster import DBSCAN
eps_values = {
    # 'pearson': np.linspace(2.5, 4, 10),
    'pca': np.linspace(.1, .3, 10),
    # 'rf': np.linspace(0.3, 1, 10),
}
# SCORES = pd.DataFrame()
# PREDICTIONS = pd.DataFrame()
SCORES = pd.read_csv('results/clustering_scores.csv')
PREDICTIONS = pd.read_csv('results/clustering_predictions.csv')
for method in out.keys():
    for base, similars in out[method].items():
        for eps in eps_values[method]:
            X_train, X_test, y_train, y_test = utils.clustering_learning(mb, mf, base, fs_methods[method], 'Recall', eps)
            if type(X_train) == int:
                continue

            models = utils.fit_models(
                X_train, y_train, 
                model=RandomForestRegressor,
                n_models=10
            )
            y_pred = utils.ensamble_predictions(models, X_test)

            kwargs = {
                'feature_selection_method': method,
                'method': f'clustering_eps={eps}',
                'target': 'Recall',
                'pca': 'pca95'
            }
            tmp = utils.format_scores(y_test, y_pred, **kwargs)
            SCORES = pd.concat([SCORES, tmp])

            kwargs = {
                'method': f'clustering_eps={eps}',
                'target': 'Recall',
                'feature_selection_method': method,
                'pca': 'pca95'
            }
            tmp = utils.format_predictions(X_test, y_test, y_pred, **kwargs)
            PREDICTIONS = pd.concat([PREDICTIONS, tmp])

            print(f'Base={base} method={method} eps={eps}')
            SCORES.to_csv('results/clustering_scores.csv', index=False)
            PREDICTIONS.to_csv('results/clustering_predictions.csv', index=False)
