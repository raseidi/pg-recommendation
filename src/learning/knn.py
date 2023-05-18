import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.spatial.distance import pdist, squareform

def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).copy()
    df.set_index('base', inplace=True)
    # df.drop('uscities', inplace=True)

    others = [
        'QueryTime', 'DistComp', 'IndexTime', 'Recall',
        'IndexParams', 'QueryTimeParams', 'k_searching',
    ]
    statistical = df.drop(others, axis=1).columns
    sc = StandardScaler()
    sc.fit(df[statistical])
    # df.loc[:, statistical] = sc.transform(df[statistical])
    others.remove('Recall')
    # df.loc[:, others] = df[others].apply(np.log)
    return df

def fit_model(X_train, y_train, random_state=0, model=RandomForestRegressor):
    reg = model(random_state=random_state)
    reg.fit(X_train, y_train)
    return reg

def ensamble_predictions(models, X_test):
    predictions = np.array([m.predict(X_test) for m in models])
    return predictions.mean(axis=0)

def get_metafeatures(data_path):
    df = pd.read_csv(data_path)
    df = df[df.base != 'uscities']
    df['index_tmp'] = df.base + '_' + df.nr_inst.astype(str)
    meta_features = [
        'attr_ent.mean', 'nr_attr', 'id', 'inst_to_attr', 'nr_inst',
        'iq_range.mean', 'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd',
        'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd',
        'median.mean', 'median.sd', 'min.mean', 'min.sd', 'nr_norm', 'index_tmp',
        'nr_outliers', 'range.mean', 'range.sd', 'sd.mean', 'sd.sd',
        'skewness.sd', 't_mean.mean', 't_mean.sd', 'var.mean', 'var.sd'
    ]
    df = df[meta_features]
    meta_features.pop(meta_features.index('index_tmp'))

    df.drop_duplicates(meta_features, keep='first', inplace=True)
    df.rename(columns={'index_tmp': 'base'}, inplace=True)
    
    sc = StandardScaler()
    sc.fit(df[meta_features])
    df.loc[:, meta_features] = sc.transform(df[meta_features])
    return df

def get_train_test(mb, metatarget, knn_bases):
    # set train/test
    train = mb[mb.index.isin(knn_bases[1:])].copy()
    X_train = train.drop(['Recall', 'QueryTime', 'IndexTime', 'DistComp'], axis=1)
    y_train = train[metatarget]
    test = mb[mb.index == knn_bases[0]].copy()
    X_test = test.drop(['Recall', 'QueryTime', 'IndexTime', 'DistComp'], axis=1)
    y_test = test[metatarget]
    return X_train, y_train, X_test, y_test


# metabase = 'data/metabase/metabase_all_k_search.csv'
# df = get_metafeatures(metabase)
# df.to_csv('data/metafeatures/metafeatures_preprocessed.csv', index=False)
metabase = 'data/metabase/metabase_v2.csv'
mb = read_data(metabase)

mf = pd.read_csv('data/metafeatures/new_metafeatures_pp.csv', index_col='base')
mf.loc[:, :] = StandardScaler().fit_transform(mf.values)
mf = mf.loc[mb.index.unique(), :]
mf.reset_index(drop=False, inplace=True)
n = mf.groupby('base').nr_inst.idxmax()
mf = mf.loc[n, :].copy()
mf.set_index('base', inplace=True)

# computing distance matrix 
dist_matrix = pd.DataFrame(data=squareform(pdist(mf)), columns=mf.index, index=mf.index)

# dist_matrix.loc[:, 'sift'].sort_values().head(n=10)
# knn_bases = dist_matrix.loc[:, 'mnist'].sort_values().head(n=5).index
# X_train, y_train, X_test, y_test = get_train_test(mb, 'Recall', knn_bases)
# reg = RandomForestRegressor(n_jobs=-1)
# reg.fit(X_train, y_train)
# reg.score(X_test, y_test)

# y_pred = reg.predict(X_test)

from itertools import product
k_params = [2,4,8,16]
bases = ['sift', 'colorHisto', 'mnist', 'texture', 'moments']
meta_targets = ['Recall', 'QueryTime', 'IndexTime', 'DistComp']
prods = product(bases, meta_targets, k_params)

SCORES = pd.DataFrame()
for base_target, metatarget, k in prods:
    print('Evaluating {} for {} with k={}'.format(base_target, metatarget, k))
    # define k closest datasets
    knn_bases = dist_matrix.loc[:, base_target].sort_values().head(n=k+1).index
    
    X_train, y_train, X_test, y_test = get_train_test(mb, metatarget, knn_bases)

    # train/test
    models = [
        fit_model(
            X_train,
            y_train,
            random_state=RS,
            model=RandomForestRegressor
        )
        for RS in range(5)
    ]
    y_pred = ensamble_predictions(models, X_test)

    # format scores
    new_score = pd.Series({
        "base": base_target,
        "target": metatarget,
        "k": k,
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
    }).to_frame().T
    SCORES = pd.concat([SCORES, new_score], ignore_index=True)
    SCORES.to_csv('results/csv/knn/scores_max_card_only.csv', index=False)