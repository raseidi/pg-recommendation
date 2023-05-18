import os
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
    if var_sum >= .9:
        PCA_FEATURES = xvar.head(ix)[~xvar.feature.isin(not_metafeatures)].feature.values
        break

# RF importance, where features lie in quantile >= 0.9
fi = pd.read_csv('data/feature_importances.csv')
fi.importance = fi.importance/fi.importance.sum()
RF_FEATURES= fi[fi.importance >= fi.importance.quantile(.9)].feature.values


# Measuring similarity according each fs method
fs_methods = {
    'pearson': PEARSON_FEATURES,
    'pca': PCA_FEATURES,
    'rf': RF_FEATURES
}

out = dict()
for key, array in fs_methods.items():
    out[key] = utils.get_similar(mf, array)

print(*out.items(), sep='\n\n\n\n')

# Learning from clustering approach
from sklearn.cluster import DBSCAN
eps_values = {
    'pearson': np.linspace(.5, 4, 10),
    'pca': np.linspace(0.1, 2, 10),
    'pca2': np.linspace(2, 4, 10),
    'rf': np.linspace(0.3, 1, 10),
}

final = pd.DataFrame()
for method in out.keys():
    for base, similars in out[method].items():
        for eps in eps_values[method]:
            X_train, X_test, y_train, y_test, labels = utils.clustering_learning(mb, mf, base, fs_methods[method], 'Recall', eps, return_labels=True)
            if type(X_train) == int:
                print('moio', base, method, eps)
                continue 
        
            aux = pd.DataFrame({
                'cluster_size': [len(X_train.index.unique())],
                'n_clusters': [len(np.unique(labels))],
                'base': [base], 
                'method': [method], 
                'eps': [eps]
            })
            final = pd.concat([final, aux])

# final.groupby('method').cluster_size.mean()
# import seaborn as sns
# sns.boxplot(x='method', y='cluster_size', data=final)
# import matplotlib.pyplot as plt
# plt.yscale('log')
# plt.show()
final.to_csv('src/notebooks/2020.12.07/cluster_size.csv', index=False)