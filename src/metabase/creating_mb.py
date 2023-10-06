# Script to join nmslib raw data and extracted metafeatures

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100)

columns = np.array(['MethodName', 'Recall', 'QueryTime', 'DistComp',
                    'IndexTime', 'IndexParams', 'QueryTimeParams',
                    'NumData', 'base', 'k_searching'])

# catDat_old: ADBIS paper
# catDat and catDat_new: Info. Sys. paper
catDat_old = pd.read_csv('data/nmslib/all_results_preProcessed.csv')
catDat_old = catDat_old[columns]
catDat_new = pd.read_csv('data/dat/all_results_preProcessed.csv')
catDat_new = catDat_new[columns]
catDat = pd.read_csv('data/nmslib/all_results_preProcessed_v2.csv')
catDat = catDat[columns]
catDat = pd.concat([catDat_old, catDat, catDat_new])
catDat.reset_index(drop=True, inplace=True)

metafeatures = pd.read_csv('data/metafeatures/new_metafeatures_pp_v3.csv')
metafeatures = metafeatures.rename(columns={'Unnamed: 0': 'base'})
metafeatures[metafeatures.base == 'sift'].nr_inst.max()

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