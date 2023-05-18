# Script to join nmslib raw data and extracted metafeatures

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)

columns = np.array(['MethodName', 'Recall', 'QueryTime', 'DistComp',
                    'IndexTime', 'IndexParams', 'QueryTimeParams',
                    'NumData', 'base', 'k_searching'])

# catDat e metafeatures precisam estar pré-processados
catDat_old = pd.read_csv('/home/seidi/Repositories/prox_graph_auto_config/data/nmslib/all_results_preProcessed.csv')
catDat_old = catDat_old[columns]
catDat_new = pd.read_csv('/home/seidi/Repositories/pgac_source/data/dat/all_results_preProcessed.csv')
catDat_new = catDat_new[columns]
catDat = pd.read_csv('/home/seidi/Repositories/prox_graph_auto_config/data/nmslib/all_results_preProcessed_v2.csv')
catDat = catDat[columns]
catDat = pd.concat([catDat_old, catDat, catDat_new])
catDat.reset_index(drop=True, inplace=True)


metafeatures = pd.read_csv('data/metafeatures/new_metafeatures_pp_v3.csv')
metafeatures = metafeatures.rename(columns={'Unnamed: 0': 'base'})
metafeatures[metafeatures.base == 'sift'].nr_inst.max()
# metafeatures.base = metafeatures.base.apply(lambda x: 'fashion' if x.startswith('fashion') else x)
# metafeatures.base = metafeatures.base.apply(lambda x: 'mnist121d' if x.startswith('mnist_121') else x)
# metafeatures.dropna(axis=1, inplace=True)
# metafeatures.drop('elapsed_time(secs)', axis=1, inplace=True)
# metafeatures.n_pcs = metafeatures.n_pcs/metafeatures.nr_attr
# metafeatures[metafeatures.base.str.startswith('mnist_121d')].base.unique()
# mnists = ['mnist_121d_500', 'mnist_121d_69900', 'mnist_121d_32000',
#        'mnist_121d_4000', 'mnist_121d_64000', 'mnist_121d_8000',
#        'mnist_121d_1000', 'mnist_121d_2000', 'mnist_121d_16000']
# nus = ['Normalized_CH_266748','Normalized_CM55_269648','Normalized_CORR_269648','Normalized_EDH_269529','Normalized_WT_269648']
# metafeatures.base = metafeatures.base.apply(lambda x: x.split('_')[0] + x.split('_')[1] + '_'+ x.split('_')[2] if x in mnists else x)
# metafeatures.base = metafeatures.base.apply(lambda x: 'Nus' + x.split('_')[1] + '_' + x.split('_')[2] if x in nus else x)
# metafeatures.nr_inst = metafeatures.base.apply(lambda x: float(x.split('_')[-1]))
# metafeatures.base = metafeatures.base.apply(lambda x: x.split('_')[0])

# como eu sobrescrevi o NumData, tive que subtrair os tamanhos maximos dos arquivos
# old_numData_syn = [10000, 100000, 1000000]
# old_numData_real = pd.unique(metafeatures[~metafeatures.base.str.startswith('base')].groupby('base').nr_inst.max().values)
# old_numData_real = [68040.0, 70000.0, 1000000.0, 25374.0]

# tmp_syn = metafeatures[metafeatures.base.str.startswith('base')].loc[:, 'nr_inst']
# tmp_syn = tmp_syn.apply(lambda x: x-100 if x in old_numData_syn else x)
# tmp_real = metafeatures[~metafeatures.base.str.startswith('base')].loc[:, 'nr_inst']
# # tmp_real = tmp_real.apply(lambda x: x-100 if x in old_numData_real else x)
# metafeatures.loc[:, 'nr_inst'] = pd.concat([tmp_syn, tmp_real])

catDat = catDat[catDat.MethodName != 'RNG']
catDat = catDat[catDat.base != 'uscities']

finished_datasets = catDat.base.unique()
metafeatures = metafeatures[metafeatures.base.isin(finished_datasets)]
metafeatures.base.unique()
metafeatures[metafeatures.base=='mnist121d'].nr_inst

catDat.set_index(['base', 'NumData'], inplace=True)
metafeatures['nr_inst'] = metafeatures['nr_inst'].astype(int)
metafeatures['nr_attr'] = metafeatures['nr_attr'].astype(int)
metafeatures.rename(columns={'nr_inst': 'NumData'}, inplace=True)
metafeatures.set_index(['base', 'NumData'], inplace=True)
new_df2 = catDat.join(metafeatures, how='outer')
new_df2.dropna(axis=0, inplace=True)
new_df2.reset_index(inplace=True)

new_df2.rename(columns={
    'NumData': 'nr_inst',
    'MethodName': 'graph_type'
    }, inplace=True)

new_df2[new_df2.base == 'sift']['nr_inst'].max()


new_df2.graph_type = new_df2.graph_type.apply(
    lambda x: 0 if x == 'sw-graph' else x)
new_df2.graph_type = new_df2.graph_type.apply(
    lambda x: 1 if x == 'NNDescentMethod' else x)
new_df2.graph_type = new_df2.graph_type.apply(
    lambda x: 2 if x == 'Brute-kNNG' else x)

new_df2.base.unique()
new_df2.to_csv('data/metabase/metabase_v3.csv', index=False)