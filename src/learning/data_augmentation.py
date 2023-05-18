import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange, LinearNDInterpolator
plt.style.use('bmh')
pd.set_option('display.max_columns', 100)

def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).copy()
    df.set_index('base', inplace=True)
    df.drop('uscities', inplace=True)
    df.graph_type = df.graph_type.astype(int)

    statistical = [
        'attr_ent.mean',
        'iq_range.mean', 'iq_range.sd', 'kurtosis.mean', 'kurtosis.sd',
        'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd',
        'median.mean', 'median.sd', 'min.mean', 'min.sd', 'nr_norm',
        'nr_outliers', 'range.mean', 'range.sd', 'sd.mean', 'sd.sd',
        'skewness.sd', 't_mean.mean', 't_mean.sd', 'var.mean', 'var.sd'
    ]
    others = [
        'QueryTime', 'DistComp', 'IndexTime',
        'nr_inst', 'IndexParams', 'inst_to_attr', 'id',
        'QueryTimeParams', 'k_searching', 'nr_attr'
    ]
    # sc = StandardScaler()
    # sc.fit(df[statistical])
    # df.loc[:, statistical] = sc.transform(df[statistical])
    # df.loc[:, others] = df[others].apply(np.log)
    return df

def get_coefs(x, y):
    interp = lagrange(x, y)
    return Polynomial(interp).coef

def get_interpolations(arr, coefs):
    formula = [(len(coefs) - i - 1, v) for i, v in enumerate(coefs)]
    y_pred = np.array([])
    for t in arr:
        aux = 0
        for exp, c in formula:
            aux += c*t**exp

        y_pred = np.append(y_pred, aux)
    
    return y_pred

metabase = 'data/metabase/metabase_all_k_search.csv'
mb = read_data(metabase)
mb.reset_index(inplace=True)
nn_original = mb.IndexParams.unique()
r_original = mb.QueryTimeParams.unique()

# mfs = [
#     'attr_ent.mean', 'inst_to_attr', 'iq_range.mean', 'iq_range.sd',
#     'kurtosis.mean', 'kurtosis.sd', 'mad.mean', 'mad.sd', 'max.mean',
#     'max.sd', 'mean.mean', 'mean.sd', 'median.mean', 'median.sd',
#     'min.mean', 'min.sd', 'nr_attr', 'nr_norm', 'nr_outliers', 'range.mean',
#     'range.sd', 'sd.mean', 'sd.sd', 'skewness.sd', 't_mean.mean',
#     't_mean.sd', 'var.mean', 'var.sd', 'id'
# ]
# nn_values = np.log(np.arange(2.0, 151.0))
# r_values = np.log(np.arange(1.0, 241.0))
# p = np.array([list(r) for r in list(product(nn_values, r_values))])

# # mb['metainstance'] = 'original'

# BASES = mb.base.unique()
# K_SEARCHING = mb.k_searching.unique()
# GRAPHS = mb.graph_type.unique()
# targets = ['Recall', 'QueryTime', 'IndexTime', 'DistComp']

# prods = product(BASES, GRAPHS, K_SEARCHING)
# for BASE, GRAPH, K in prods:
#     nmslib_format = pd.DataFrame()
#     print(BASE, GRAPH, K)
#     df = mb[
#         (mb.base==BASE) &
#         (mb.graph_type==GRAPH) &
#         (mb.k_searching==K)
#     ].copy()
    
#     if len(df) == 0:
#         continue

#     df = df[df.nr_inst == df.nr_inst.max()].copy()
#     df.loc[:, ['IndexParams', 'QueryTimeParams']] = df.loc[:, ['IndexParams', 'QueryTimeParams']].apply(np.log)

#     last = None
#     for t in targets:
#         x, y = (df.loc[:, ['IndexParams', 'QueryTimeParams']].values, df.loc[:, t].values)
        
#         coefs = LinearNDInterpolator(x, y)
#         interpolations = coefs(p)
#         interpolations = np.nan_to_num(interpolations)

#         df_tmp = pd.DataFrame(p, columns=['IndexParams', 'QueryTimeParams'])
#         df_tmp[t] = interpolations

#         final_df = pd.concat([df[df_tmp.columns], df_tmp])
#         final_df.drop_duplicates(['IndexParams', 'QueryTimeParams'], keep='first', inplace=True)
#         final_df.loc[:, ['IndexParams', 'QueryTimeParams']] = final_df.loc[:, ['IndexParams', 'QueryTimeParams']].apply(np.exp).round()
#         if last is None:
#             last = final_df.copy()
#         else:
#             last[t] = final_df[t]

#     nr_inst = df.nr_inst.max()
#     last['base'] = BASE
#     last['k_searching'] = K
#     last['graph_type'] = GRAPH
#     last['nr_inst'] = nr_inst

#     row = df.drop_duplicates(mfs)[mfs]
#     x = pd.DataFrame([row.values.reshape(-1)] * len(last), columns=row.columns, index=last.index)
#     last[row.columns] = x

#     nmslib_format = pd.concat([nmslib_format, last], ignore_index=True)
#     nmslib_format.to_csv(f'/home/seidi/Repositories/prox_graph_auto_config/data/interpolations/{BASE}_graph={GRAPH}_k={K}.csv', index=False)



import os
os.chdir('data/interpolations')
final_df = pd.DataFrame()
for f in os.listdir():
    if not f.startswith('base'):
        print(f)

    df = pd.read_csv(f)
    tmp = df[(df.IndexParams.isin(nn_original)) & (df.QueryTimeParams.isin(r_original))]
    final_df = pd.concat([final_df, tmp], ignore_index=True)

final_df.to_csv('../metabase/realOnly_interpolated_balanced.csv', index=False)





# r = 1
# nn = 10

# figure, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
# sns.lineplot(x='IndexParams', y=target, data=final_df[final_df.QueryTimeParams==r], ax=axes[0], marker='o')
# sns.lineplot(x='QueryTimeParams', y=target, data=final_df[final_df.IndexParams==nn], ax=axes[1], marker='o')
# axes[0].set_title('R={}'.format(r))
# axes[1].set_title('NN={}'.format(nn))
# plt.show()


# # df = mb[
# #     (mb.index=='mnist') &
# #     (mb.graph_type==0) &
# #     (mb.nr_inst.isin([67940, 69900, 999900])) &
# #     (mb.k_searching==30)
# # ][['IndexParams', 'QueryTimeParams', *targets]].copy()

# figure, axes = plt.subplots(4, 2, figsize=(12, 6), dpi=100, sharex=True)
# flag = True
# axes = iter(axes.reshape(-1))
# targets = ['Recall', 'QueryTime', 'IndexTime', 'DistComp']

# df = last.copy()
# for target in targets:
#     ax = next(axes)
#     sns.lineplot(x='IndexParams', y=target, data=df[df.QueryTimeParams==r], ax=ax, marker='o')
#     if flag:
#         ax.set_title('QueryTimeParams={}'.format(r))

#     ax = next(axes)
#     sns.lineplot(x='IndexParams', y=target, data=df[df.QueryTimeParams==r], ax=ax, marker='o')
#     if flag:
#         ax.set_title('IndexParams={}'.format(nn))
#         flag = False

# plt.savefig('src/notebooks/2020.09.25/plots/interpolation.pdf')
# plt.show()



























