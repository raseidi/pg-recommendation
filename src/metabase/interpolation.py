import os
import pandas as pd
import numpy as np
from pyrsistent import b
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold
pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 201)
plt.style.use('bmh')
mb = pd.read_csv('data/metabase/metabase_v3.csv')

mb = mb.sample(frac=1)
mf = pd.read_csv('data/metafeatures/new_metafeatures_pp_v3.csv')
drop_cols = mf.columns[(mf.columns.str.startswith('rc_')) | (mf.columns.str.endswith('_median'))]

mb.drop(drop_cols, axis=1, inplace=True)
mf.drop(drop_cols, axis=1, inplace=True)

# drop_bases = [
#     'nasa', 'NusCM55', 'NusCH', 'NusEDH',
#     'NusCORR', 'colors', 'NusWT', 'aloi',
#     'cifar10', 'deep1M']
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
mf.set_index('base', inplace=True)
# mf.columns

mb.base = mb.base + '_' + mb.nr_inst.astype(int).astype(str)
mb.set_index('base', inplace=True)
mb.drop(drop_cols_var, axis=1, inplace=True)
mb = mb.astype(float)
mb.head()


from itertools import product
from scipy.interpolate import LinearNDInterpolator

qt = [1, 5, 10, 20, 40, 80, 120, 200, 240]
ip = [5, 25, 55, 100]
tmp2 = mb[(mb.QueryTimeParams.isin(qt)) & (mb.IndexParams.isin(ip))].copy()
tmp2.reset_index(inplace=True)
tmp2.loc[:, ['IndexParams', 'QueryTimeParams', 'QueryTime', 'IndexTime', 'DistComp']] = tmp2.loc[:, ['IndexParams', 'QueryTimeParams', 'QueryTime', 'IndexTime', 'DistComp']].apply(np.log)
k = sorted(tmp2.k_searching.unique())
G = tmp2.graph_type.unique()
bases = tmp2.base.unique()

not_mf = ['QueryTime', 'Recall', 'IndexTime', 'DistComp', 'IndexParams', 'QueryTimeParams', 'k_searching', 'graph_type']

METATARGETS = ['QueryTime', 'Recall', 'IndexTime', 'DistComp']
prods = product(bases, G, k)
new_mb = pd.DataFrame()
for base, g, k_param in prods:
    interpolate = tmp2[
        (tmp2.base == base) &
        (tmp2.k_searching == k_param) & 
        (tmp2.graph_type==g)
        ].copy()
    
    if len(interpolate) == 0:
        continue

    mfs = interpolate.drop(not_mf, axis=1)
    mfs.drop_duplicates(inplace=True)
    if len(mfs) != 1:
        print(f'[-] Base="{base}" g={g} k={k_param} deu ruim. Len(mf)={len(mfs)}')
        continue
    
    mfs.set_index('base', inplace=True)
    
    first = True
    for mt in METATARGETS:
        x, y = (interpolate.loc[:, ['IndexParams', 'QueryTimeParams']].values, interpolate.loc[:, mt].values)
        try:
            NN = np.log(np.arange(5, 105, 5))
            # R = sorted(tmp2.QueryTimeParams.unique())
            R = np.log(np.concatenate([np.arange(1, 11, 1), np.arange(0, 241, 20)[1:]]))
            p = np.array([list(r) for r in list(product(NN, R))])

            coefs = LinearNDInterpolator(x, y)
            interpolations = coefs(p)
        except:
            print(f'[-] Base="{base}" g={g} k={k_param} deu ruim na interpolação.')
            continue

        if first:
            df_tmp = pd.DataFrame(p, columns=['IndexParams', 'QueryTimeParams'])
            df_tmp.loc[:, ['IndexParams', 'QueryTimeParams']] = df_tmp.loc[:, ['IndexParams', 'QueryTimeParams']].apply(np.exp).round()
            df_tmp[mt] = interpolations
            df_tmp['base'] = base
            df_tmp['graph_type'] = g
            df_tmp['k_searching'] = k_param
            df_tmp.set_index('base', inplace=True)
            df_tmp = df_tmp.join(mfs)
            first = False
        else:
            df_tmp[mt] = interpolations
    
    df_tmp.dropna(axis=0, inplace=True) # extrapolations are always nan
    new_mb = pd.concat([new_mb, df_tmp])

new_mb.reset_index(inplace=True)
new_mb.to_csv('data/metabase/metabase_v3_interpolated.csv', index=False)

# checking original values vs interpolated values
cols = ['IndexParams', 'QueryTimeParams', 'graph_type', 'Recall', 'QueryTime']
BASE = 'sift_999900' 
NN = 5
R = 10
G = 0
K = 30

interpolated = new_mb[
    (new_mb.base == BASE) &
    (new_mb.IndexParams == NN) &
    (new_mb.QueryTimeParams == R) &
    (new_mb.k_searching == K) &
    (new_mb.graph_type == G)
][['QueryTime', 'DistComp', 'IndexTime']].values

original = mb[
    (mb.index == BASE) &
    (mb.IndexParams == NN) &
    (mb.QueryTimeParams == R) &
    (mb.graph_type == G) &
    (mb.k_searching == 30)
][['QueryTime', 'DistComp', 'IndexTime']].values

np.log(original) == interpolated