import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import sys
#from IPython.core import ultratb
#sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


DROP_COLS_VAR = np.array([
    'attr_conc.mean', 'attr_conc.sd', 'cat_to_num', 'nr_cat',
    'sparsity.mean', 'sparsity.sd', 'lid_hist2', 'lid_hist3',
    'lid_hist4', 'lid_hist5', 'lid_hist6', 'lid_hist7', 'lid_hist8',
    'lid_hist9', 'nr_num', 'nr_bin'], dtype=object)

DROP_REAL_BASES = [
    'nasa', 'mnist400d', 'NusCM55', 'NusCH', 'mnist289d', 'NusEDH',
    'mnist625d', 'NusCORR', 'mnist196d',
    'mnist484d', 'colors', 'NusWT', 'aloi',
    'cifar10', 'deep1M', 'mnistBackground', 'mnistBackgroundRotation'
]

DROP_BASES = np.array([
    'base54_100000', 'base53_100000', 'base46_100000', 
    'base47_100000', 'base35_100000', 'base52_100000', 
    'base49_100000', 'base77_1000000', 'base48_100000', 
    'base50_100000', 'base37_100000', 'base80_1000000', 
    'base51_100000'
])

META_TARGETS = ['Recall_90', 'Recall_95', 'Recall_99', 'Recall', 'DistComp', 'QueryTime', 'IndexTime']

def fit_models(X_train, y_train, model, n_models=10):
    models = []
    for n in range(n_models):
        reg = model(n_jobs=-1, random_state=n)
        models.append(reg.fit(X_train, y_train))
    
    return models

def ensamble_predictions(models, X_test):
    predictions = np.array([m.predict(X_test) for m in models])
    return predictions.mean(axis=0)

def read_mb(path='data/metabase/metabase_v3.csv', recall_only=True, pre_processed=True):
    mb = pd.read_csv(path)
    mb = mb[~mb.base.str.startswith('base38')]
    mb.base = mb.base + '_' + mb.nr_inst.astype(int).astype(str)

    if not pre_processed:
        return mb
    
    mb = mb.sample(frac=1)
    mb.drop(DROP_COLS_VAR, axis=1, inplace=True)
    drop_cols = mb.columns[(mb.columns.str.startswith('rc_')) | (mb.columns.str.endswith('_median'))]
    mb.drop(drop_cols, axis=1, inplace=True)
    mb.loc[:, ['QueryTime', 'DistComp', 'IndexTime', 'nr_inst', 'nr_attr']] = mb.loc[:, ['QueryTime', 'DistComp', 'IndexTime', 'nr_inst', 'nr_attr']].apply(np.log)
    
    mb.set_index('base', inplace=True)
    return mb.astype(float)

def read_int_mb(path='data/metabase/metabase_v3_interpolated.csv', recall_only=True, pre_processed=True):
    mb = pd.read_csv(path)
    mb = mb[~mb.base.str.startswith('base38')]
    
    if not pre_processed:
        return mb
    
    mb = mb.sample(frac=1)
    drop_cols = mb.columns[(mb.columns.str.startswith('rc_')) | (mb.columns.str.endswith('_median'))]
    mb.drop(drop_cols, axis=1, inplace=True)
    mb.loc[:, ['nr_inst', 'nr_attr']] = mb.loc[:, ['nr_inst', 'nr_attr']].apply(np.log)
    
    # mb.base = mb.base + '_' + mb.nr_inst.astype(int).astype(str)
    mb.set_index('base', inplace=True)
    return mb.astype(float)

def read_mf(path='data/metafeatures/new_metafeatures_pp_v3.csv', pre_processed=True):
    mf = pd.read_csv(path)
    drop_cols = mf.columns[(mf.columns.str.startswith('rc_')) | (mf.columns.str.endswith('_median'))]# | (mf.columns.str.endswith('lid_entropy'))]
    mf.drop(drop_cols, axis=1, inplace=True)

    mf = mf[~mf.base.isin(DROP_REAL_BASES)]
    mf.base = mf.base + '_' + mf.nr_inst.astype(int).astype(str)
    mf.base = mf.base.apply(lambda x: 'texture_67940' if x == 'texture_67836' else x)
    mf.base = mf.base.apply(lambda x: 'sift_999900' if x == 'sift_985462' else x)
    mf = mf[~mf.base.str.startswith('base38')]
    mf = mf[~mf.base.str.startswith('base39_')]
    mf.loc[:, ['nr_inst', 'nr_attr']] = mf.loc[:, ['nr_inst', 'nr_attr']].apply(np.log)

    selector = VarianceThreshold(0.01)
    selector.fit(mf.drop('base', axis=1))
    drop_cols_var = mf.drop('base', axis=1).columns[~selector.get_support()].values
    mf.drop(drop_cols_var, axis=1, inplace=True)
    
    mf.set_index('base', inplace=True)
    mf.drop(DROP_BASES, axis=0, inplace=True)
    return mf

def get_similar(mf, features, k=5):
    bases_reais = ['texture_67940', 'sift_999900', 'moments_67940', 'mnist121d_69900', 'fashion_69900','colorHisto_67940', 'mnist_69900']
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

        knng = NearestNeighbors(n_neighbors=k)
        knng.fit(train)
        dist, ix = knng.kneighbors(test)
        bases = train.iloc[ix.reshape(-1)].index.values

        similar_tasks[b] = bases

    return similar_tasks

def format_scores(y_test, y_pred, **kwargs):
    tmp = pd.DataFrame({
        'base': y_test.index.unique()[0],
        'r2': [r2_score(y_test, y_pred)],
        'rmse': [mean_squared_error(y_test, y_pred) ** (1/2)],
    })
    for k, v in kwargs.items():
        tmp[k] = v

    return tmp

def format_predictions(X_test, y_test, y_pred, **kwargs):
    df_tmp = pd.DataFrame()
    df_tmp['y_true'] = y_test.values
    df_tmp['y_pred'] = y_pred
    df_tmp['IndexParams'] = X_test.IndexParams.values
    df_tmp['QueryTimeParams'] = X_test.QueryTimeParams.values
    df_tmp['k_searching'] = X_test.k_searching.values
    df_tmp['base'] = X_test.index.unique()[0]
    df_tmp['graph_type'] = X_test.graph_type.values

    for k, v in kwargs.items():
        df_tmp[k] = v

    return df_tmp

def similarity_learning(mb, base, similars, target='Recall'):
    X_train = mb[mb.index.isin(similars)].drop(target, axis=1).copy()
    y_train = mb[mb.index.isin(similars)][target].copy()
    X_test = mb[mb.index == base].drop(target, axis=1).copy()
    y_test = mb[mb.index == base][target].copy()

    return X_train, X_test, y_train, y_test

def clustering_learning(mb, mf, base, features, target, eps):
    features = list(set(features))
    mf_new = mf[features].copy()
    mf_new.loc[:, :] = StandardScaler().fit_transform(mf_new)
    train = mf_new[~mf_new.index.str.startswith(base.split('_')[0])].copy()
    clus = DBSCAN(eps=eps, min_samples=2).fit(train)
    train['label'] = clus.labels_

    test = mf_new[mf_new.index == base].copy()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train.drop('label', axis=1), train.label)
    pred = knn.predict(test)
    if pred == -1:
        k = int(train[train.label != -1].label.value_counts().mean())
        similars = get_similar(mf, features, k=5)
        train_bases = similars[base]
    else:
        train_bases = train[train.label == pred[0]].index.values
    
    X_train = mb.loc[(mb.index.isin(train_bases)), ~(mb.columns.isin(META_TARGETS))].copy()
    y_train = mb[mb.index.isin(train_bases)][target].copy()
    X_test = mb.loc[(mb.index == base), ~(mb.columns.isin(META_TARGETS))].copy()
    y_test = mb[mb.index == base][target].copy()

    return X_train, X_test, y_train, y_test

def get_features_for_similarity(target='Recall', quantile=0.9, fs_method='rf', mb=None):
    not_metafeatures = ['Recall', 'IndexParams', 'QueryTimeParams', 'graph_type', 'DistComp', 'IndexTime', 'QueryTime']
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    if fs_method == 'rf':
        feat_imp = pd.read_csv('data/feature_importances.csv')
        fi = feat_imp[feat_imp.target == target]
        return fi[fi.importance >= fi.importance.quantile(quantile)].feature.values
    elif fs_method == 'pca':
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
        return PCA_FEATURES

    elif fs_method == 'pearson':
        pearson_corr = mb.corr()[target].abs().sort_values(ascending=False)
        pearson_corr.drop(not_metafeatures, inplace=True)
        return pearson_corr[pearson_corr > pearson_corr.quantile(quantile)].index.values

    else:
        print('fs_method not found')
        return None

def get_eps(mf, features, quantile=0.9):
    from sklearn.neighbors import NearestNeighbors
    features = list(set(features))
    mf_new = mf[features].copy()
    sc = StandardScaler()
    sc.fit(mf_new)
    mf_new.loc[:, :] = sc.transform(mf_new)
    
    knng = NearestNeighbors(n_neighbors=2)
    knng.fit(mf_new)
    
    dist, ix = knng.kneighbors(mf_new)
    dist = dist[:, 1:].reshape(-1)
    dist = np.sort(dist)
    
    return np.quantile(dist, quantile)
