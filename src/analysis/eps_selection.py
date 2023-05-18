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

out = dict()
for key, array in fs_methods.items():
    out[key] = utils.get_similar(mf, array)

print(*out.items(), sep='\n\n\n\n')


''' ---------------------------------------------------- '''
''' ---------------------------------------------------- '''
''' ---------------------------------------------------- '''
''' ---------------------------------------------------- '''
''' ---------------------------------------------------- '''
''' ---------------------------------------------------- '''
from sklearn.neighbors import NearestNeighbors

mf_new = mf[RF_FEATURES].copy()
sc = StandardScaler()
sc.fit(mf_new)
mf_new.loc[:, :] = sc.transform(mf_new)

dists = []
for k in range(1, 100):
    knng = NearestNeighbors(n_neighbors=k+1)
    knng.fit(mf_new)
    dist, ix = knng.kneighbors(mf_new)
    dist = dist[:, 1:]
    dists.append(dist.mean(axis=1).mean())

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(dists)
sns.lineplot(np.arange(len(dists)), dists, marker='o')
plt.yscale('log')
plt.show()
