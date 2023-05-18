'''
    FIRST APPROACH EVALUATING A WIDE RANGE OF EPS VALUES
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def fit_model(X_train, y_train, random_state=0, model=RandomForestRegressor):
    reg = model(random_state=random_state, n_jobs=-1)
    # reg = DecisionTreeRegressor(random_state=random_state)
    reg.fit(X_train, y_train)
    return reg

def ensamble_predictions(models, X_test):
    predictions = np.array([m.predict(X_test) for m in models])
    return predictions.mean(axis=0)


def get_features(mb, target, as_frame=False):
    from sklearn.model_selection import train_test_split, cross_validate

    X = mb.loc[:, ~mb.columns.isin(['Recall', 'QueryTime', 'DistComp', 'IndexTime'])]
    y = mb[target].to_frame()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=X.index)

    cv = cross_validate(DecisionTreeRegressor(), X_train, y_train, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, return_estimator=True)
    sorted(cv.keys())

    count = np.zeros(len(X_train.columns))
    for reg in cv['estimator']:
        count += reg.feature_importances_

    fi = pd.DataFrame({
        'feature': X_train.columns,
        'importance': count / 10 # len cv
    }).sort_values(by='importance', ascending=False)
    features = fi[~fi.feature.isin(['IndexParams', 'QueryTimeParams', 'graph_type', 'k_searching'])].head(5).feature.unique()
    if as_frame:
        return fi[~fi.feature.isin(['IndexParams', 'QueryTimeParams', 'graph_type', 'k_searching'])]
    else:
        return features

def get_similar(mf, features):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    bases_reais = ['texture_67836', 'sift_985462', 'moments_67940', 'mnist121d_69900', 'fashion_69900','colorHisto_67940', 'mnist_69900']
    tmp = mf.copy()
    tmp = tmp[features]

    similar_tasks = dict()
    for b in bases_reais:
        train = tmp[~tmp.index.str.startswith(b.split('_')[0] + '_')].copy()
        test = tmp[tmp.index == b].copy()
        
        sc = StandardScaler()
        sc.fit(train)
        train.loc[:, :] = sc.transform(train)
        test.loc[:, :] = sc.transform(test)

        knng = NearestNeighbors(n_neighbors=5)
        knng.fit(train)
        dist, ix = knng.kneighbors(test)
        bases = train.iloc[ix.reshape(-1)].index.values
        # bases = [x for x in bases if not x.startswith('base80')]
        # print(f'Datasets mais similares ao {b}:')
        # for d, b, in zip(*dist, bases):
        #     print(f'base={b}, dist={d}')

        similar_tasks[b] = bases

    return similar_tasks

mb = pd.read_csv('data/metabase/metabase_v3.csv')
# mb = pd.read_csv('data/metabase_v3.csv')
mb.loc[:, ['QueryTime', 'IndexTime', 'DistComp']]
mb = mb.sample(frac=1)
mf = pd.read_csv('data/metafeatures/new_metafeatures_pp_v3.csv')
drop_cols = mf.columns[(mf.columns.str.startswith('rc_')) | (mf.columns.str.endswith('_median'))]

mb.drop(drop_cols, axis=1, inplace=True)
mf.drop(drop_cols, axis=1, inplace=True)

drop_bases = [
    'nasa', 'mnist400d', 'NusCM55', 'NusCH', 'mnist289d', 'NusEDH',
    'mnist625d', 'NusCORR', 'mnist196d',
    'mnist484d', 'colors', 'NusWT', 'aloi',
    'cifar10', 'deep1M', 'mnistBackground', 'mnistBackgroundRotation'
]
mf = mf[~mf.base.isin(drop_bases)]
mf.base = mf.base + '_' + mf.nr_inst.astype(int).astype(str)

mf.loc[:, ['nr_inst', 'nr_attr']] = mf.loc[:, ['nr_inst', 'nr_attr']].apply(np.log)
selector = VarianceThreshold(0.01)
selector.fit(mf.drop('base', axis=1))
drop_cols_var = mf.drop('base', axis=1).columns[~selector.get_support()].values
mf.drop(drop_cols_var, axis=1, inplace=True)
# mf.drop('n_pcs', axis=1, inplace=True)
# mb.drop('n_pcs', axis=1, inplace=True)
mf.set_index('base', inplace=True)
sc = StandardScaler()
sc.fit(mf)
mf.loc[:, :] = sc.transform(mf.loc[:, :])

mb.base = mb.base + '_' + mb.nr_inst.astype(int).astype(str)
mb.set_index('base', inplace=True)
mb.drop(drop_cols_var, axis=1, inplace=True)
mb = mb.astype(float)
drop_bases = [
    'base80_1000000', 'base48_100000', 'base52_100000',
    'base46_100000', 'base37_100000', 'base77_1000000',
    'base35_100000', 'base53_100000', 'base50_100000',
    'base54_100000', 'base49_100000', 'base51_100000',
    'base47_100000']
mf.drop(drop_bases, inplace=True)
mf.reset_index(inplace=True)
mf.base = mf.base.replace({
    'texture_67836': 'texture_67940',
    'sift_985462': 'sift_999900'
})
mf.set_index('base', inplace=True)

# features = get_features(mb, 'Recall', as_frame=True)
features = pd.read_csv('data/metafeatures/feature_importances.csv')
features.sort_values(by='importance', inplace=True, ascending=False)

mf_new = mf[features.feature.values[:5]].copy()
# mf_new = mf[features.feature.values].copy()
# mf_new = mf[features].copy()
importances = features.importance.values.copy()
importances /= importances.max()

from sklearn.neighbors import KNeighborsClassifier

# mf_new.loc[:, :] = StandardScaler().fit_transform(mf_new)
# mf_weighted = mf_new.multiply(importances)
# mf_weighted.reset_index(inplace=True)
# mf_weighted.base = mf_weighted.base.replace({
#     'texture_67836': 'texture_67940',
#     'sift_985462': 'sift_999900'
# })
# mf_weighted.set_index('base', inplace=True)

# mf = mf_weighted.copy()

bases = ['texture_67940', 'sift_999900', 'moments_67940', 'mnist121d_69900', 'fashion_69900', 'colorHisto_67940', 'mnist_69900']
targets = ['QueryTime']#, 'QueryTime', 'IndexTime', 'DistComp']
final_df = pd.DataFrame()
# mb_int = pd.read_csv('data/metabase/metabase_v3_interpolated.csv')
mb_int = pd.read_csv('data/metabase/metabase_v3.csv')
mb_int.set_index('base', inplace=True)
mb_int[mb_int.index.str.startswith('texture')].index.unique()
mb = mb_int.copy()

preds = pd.read_csv('results/clustering_predictions.csv')
final_df = pd.read_csv('results/clustering_scores.csv')
for b in bases:
    for eps in np.linspace(0.3, 1, 10):
        train = mf_new[~mf_new.index.str.startswith(b.split('_')[0])].copy()
        clus = DBSCAN(eps=eps, min_samples=2).fit(train)
        train['label'] = clus.labels_
        # train[train.index=='sift_999900']
        # train[train.label==12]
        # train.label.unique()
        # train.label.value_counts()

        # clus = DBSCAN(eps=6).fit(train) # default
        test = mf_new[mf_new.index == b].copy()
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train.drop('label', axis=1), train.label)
        pred = knn.predict(test)
        if pred == -1:
            continue 
        print(f'{b} label={pred} eps={eps}')
        assert len(pred) == 1
        train_bases = train[train.label == pred[0]].index.values
        
        X_train = mb[mb.index.isin(train_bases)].drop(targets, axis=1).copy()
        y_train = mb[mb.index.isin(train_bases)].Recall.copy()
        X_test = mb[mb.index == b].drop(targets, axis=1).copy()
        y_test = mb[mb.index == b].Recall.copy()

        models = [
            fit_model(
                X_train, y_train,
                random_state=i
            ) for i in range(5)
        ]
        y_pred = ensamble_predictions(models, X_test)
        
        r2 = r2_score(y_test, y_pred) #92
        tmp = pd.DataFrame({
            'base': [b],
            'target': ['QueryTime'],
            'r2': [r2],
            'rmse': [mean_squared_error(y_test, y_pred) ** (1/2)],
            'weighted': ['False'],
            'eps': [eps],
            'cluster': pred
        })
        final_df = pd.concat([final_df, tmp])

        df_tmp = pd.DataFrame()
        df_tmp['y_true'] = y_test.values
        df_tmp['y_pred'] = y_pred
        df_tmp['IndexParams'] = X_test.IndexParams.values
        df_tmp['QueryTimeParams'] = X_test.QueryTimeParams.values
        df_tmp['k_searching'] = 30 #X_test.k_searching.values
        df_tmp['k_searching'] = X_test.graph_type.values
        df_tmp['base'] = b
        df_tmp['method'] = 'clustering top features, original mb'
        df_tmp['target'] = 'QueryTime'
        df_tmp['eps'] = eps
        preds = pd.concat([preds, df_tmp])
        print(f'Base={b}')

# final_df.r2
final_df.to_csv('src/notebooks/2020.11.30/results/clustering_tf_int_score.csv', index=False)
preds.to_csv('src/notebooks/2020.11.30/results/clustering_tf_int_preds.csv', index=False)


final_df[['base', 'cluster', 'r2']]