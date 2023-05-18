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
pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 201)

def get_features(mb, target, as_frame=False):
    from sklearn.model_selection import train_test_split, cross_validate

    X = mb.loc[:, ~mb.columns.isin(['Recall', 'QueryTime', 'DistComp', 'IndexTime'])]
    y = mb[target] #.to_frame()
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

def fit_model(X_train, y_train, random_state=0, model=RandomForestRegressor):
    reg = model(random_state=random_state, n_jobs=-1)
    # reg = DecisionTreeRegressor(random_state=random_state)
    reg.fit(X_train, y_train)
    return reg

def ensamble_predictions(models, X_test):
    predictions = np.array([m.predict(X_test) for m in models])
    return predictions.mean(axis=0)

mb = pd.read_csv('data/metabase/metabase_v3.csv')
mb.loc[:, ['QueryTime', 'IndexTime', 'DistComp']]
mb = mb.sample(frac=1)
mf = pd.read_csv('data/metafeatures/new_metafeatures_pp_v3.csv')
# mf = pd.read_csv('data/metafeatures/new_metafeatures.csv')
drop_cols = mf.columns[(mf.columns.str.startswith('rc_')) | (mf.columns.str.endswith('_median'))]# | (mf.columns.str.endswith('lid_entropy'))]

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
mf.base = mf.base.apply(lambda x: 'texture_67940' if x == 'texture_67836' else x)
mf.base = mf.base.apply(lambda x: 'sift_999900' if x == 'sift_985462' else x)
mf = mf[~mf.base.str.startswith('base38')]
mf.set_index('base', inplace=True)

mb = mb[~mb.base.str.startswith('base38')]
mb.base = mb.base + '_' + mb.nr_inst.astype(int).astype(str)
mb.set_index('base', inplace=True)
mb.drop(drop_cols_var, axis=1, inplace=True)
mb = mb.astype(float)
mb.head()


drop = list(set(mf.index.unique()) - set(mb.index.unique()))
mf.drop(drop, axis=0, inplace=True)
mf = mf[~mf.index.str.startswith('base39_')].copy()

# features = get_features(mb, 'Recall', as_frame=True)
# features = features.feature.values
features = pd.read_csv('data/metafeatures/feature_importances.csv')
features.sort_values(by='importance', inplace=True, ascending=False)
sns.barplot(x='feature', y='importance', data=features)
plt.show()

features.importance.head(5).sum()
features.importance.head(9).sum() / features.importance.sum()
features.importance.values[6:].sum()
# features = features.iloc[:5,:]

mf_new = mf[features.feature.values].copy()
importances = features.importance.values.copy()
importances /= importances.max()
# importances /= importances.sum()

# sc = StandardScaler()
# sc.fit(mf_new)
# mf_new.loc[:, :] = sc.transform(mf_new)
# mf_new.loc[:, :] /= mf_new.max()
# mf_weighted = mf_new.multiply(importances)
# mf_weighted.max()
mf_weighted.drop('min.mean', axis=1, inplace=True)
features = features[features.feature != 'min.mean']

features = ['rv', 'lid_entropy', 'lid_mean','attr_ent.mean','kurtosis.mean']
similars = get_similar(mf_new, features.feature.values[:5])
print(*similars.items(), sep='\n\n-----\n\n')

# mb_int = pd.read_csv('data/metabase/metabase_v3_interpolated.csv')
# mb_int.set_index('base', inplace=True)
# # mb = mb_int.copy()


scores = pd.DataFrame()
preds = pd.DataFrame()
target = 'Recall'
metatargets=['Recall']#, 'QueryTime', 'DistComp', 'IndexTime']
# mb = mb_int.copy()
for k,v in similars.items():
    if k.startswith('texture'):
        k = 'texture_67940'
    elif k.startswith('sift'):
        k = 'sift_999900'

    X_train = mb[mb.index.isin(v[:1])].drop(metatargets, axis=1).copy()
    y_train = mb[mb.index.isin(v[:1])][target].copy()
    X_test = mb[mb.index == k].drop(metatargets, axis=1).copy()
    y_test = mb[mb.index == k][target].copy()
    
    # reg = GradientBoostingRegressor(max_depth=5, learning_rate=0.05, n_estimators=500)
    models = [
    fit_model(
        X_train,
        y_train,
        random_state=RS,
        model=RandomForestRegressor
    )
    for RS in range(10)
    ]
    y_pred = ensamble_predictions(models, X_test)

    tmp = pd.DataFrame({
        'base': [k],
        'target': [target],
        'r2': [r2_score(y_test, y_pred)],
        'rmse': [mean_squared_error(y_test, y_pred) ** (1/2)],
    })
    scores = pd.concat([scores, tmp])

    df_tmp = pd.DataFrame()
    df_tmp['y_true'] = y_test.values
    df_tmp['y_pred'] = y_pred
    df_tmp['IndexParams'] = X_test.IndexParams.values
    df_tmp['QueryTimeParams'] = X_test.QueryTimeParams.values
    df_tmp['k_searching'] = X_test.k_searching.values
    df_tmp['base'] = k
    df_tmp['method'] = 'ifms'
    df_tmp['target'] = 'Recall'
    df_tmp['graph_type'] = X_test.graph_type.values
    preds = pd.concat([preds, df_tmp])

    print(f'Base={k}, target={target}')

scores.r2.mean()
scores[['base', 'r2']]
# scores.to_csv('src/notebooks/2020.11.30/results/scores/similarity_mb=orig_k=1.csv', index=False)
preds.to_csv('src/notebooks/2020.11.30/results/predictions/similarity_mb=orig_k=1.csv', index=False)


''' similarity (k=1) using interpolated sets'''
mb_int = pd.read_csv('data/metabase/metabase_v3_interpolated.csv')
mb_int.set_index('base', inplace=True)
scores = pd.DataFrame()
preds = pd.DataFrame()
for b, sims in similars.items():
    # if b.startswith('texture'):
    #     b = 'texture_67940'
    # elif b.startswith('sift'):
    #     b = 'sift_999900'
    train = mb_int[mb_int.index == sims[0]].copy()
    test = mb_int[mb_int.index == b].copy()
    test = test[['IndexParams', 'QueryTimeParams', 'Recall', 'graph_type']]
    train = train[['IndexParams', 'QueryTimeParams', 'Recall', 'graph_type']]
    test.sort_values(by=['IndexParams', 'QueryTimeParams', 'graph_type'], inplace=True)
    train.sort_values(by=['IndexParams', 'QueryTimeParams', 'graph_type'], inplace=True)
    nn = test.IndexParams.unique()
    r = test.QueryTimeParams.unique()
    train = train[(train.IndexParams.isin(nn)) & (train.QueryTimeParams.isin(r))]

    r2 = r2_score(train.Recall, test.Recall)
    rmse = mean_squared_error(train.Recall, test.Recall) ** (1/2)
    n = len(train)
    p = len(mb.columns) - 4
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    print(f'{b} r2={r2:.2f} adj_r2={adj_r2:.2f} rmse={rmse:.2f} (most similar: {sims[0]})')
    
    df_tmp = pd.DataFrame()
    df_tmp['base'] = [b]
    df_tmp['r2'] = [r2]
    df_tmp['rmse'] = rmse
    df_tmp['method'] = ['ifms']
    df_tmp['target'] = ['Recall']
    scores = pd.concat([scores, df_tmp])
    
    df_tmp = pd.DataFrame()
    df_tmp['y_true'] = train.Recall.values
    df_tmp['y_pred'] = test.Recall.values
    df_tmp['IndexParams'] = train.IndexParams.values
    df_tmp['QueryTimeParams'] = train.QueryTimeParams.values
    df_tmp['k_searching'] = 30
    df_tmp['base'] = b
    df_tmp['method'] = 'ifms'
    df_tmp['target'] = 'Recall'
    df_tmp['graph_type'] = test.graph_type.values
    preds = pd.concat([preds, df_tmp])

# scores.to_csv('src/notebooks/2020.11.30/results/scores/ifms.csv', index=False)
preds.to_csv('src/notebooks/2020.11.30/results/predictions/ifms.csv', index=False)

# percs = np.linspace(0, 100, 50)
# qq_train = np.percentile(train.Recall,percs)
# qq_test = np.percentile(test.Recall,percs)
# plt.plot(qq_train,qq_test, ls="", marker="o")

# x = np.linspace(np.min((qq_train.min(),qq_test.min())), np.max((qq_train.max(),qq_test.max())))
# plt.plot(x,x, color="k", ls="--")

# plt.show()

