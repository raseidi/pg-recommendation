import os
os.chdir('../../')
import utils
import pandas as pd

adbis = pd.read_csv('/home/labkasterdt1/Documents/seidi/adbis_mb.csv')
adbis = adbis.rename(columns={'id':'lid_mean'})
adbis = adbis[adbis.base != 'uscities']
new = utils.read_mb('/home/labkasterdt1/Documents/seidi/mestrado_final/data/metabase/metabase_v3.csv')
new.reset_index(inplace=True)

drop_cols = ['QueryTime', 'IndexTime', 'DistComp']
adbis.drop(drop_cols, axis=1, inplace=True)
new.drop(drop_cols, axis=1, inplace=True)

# Adbis columns
adbis_cols = adbis.columns.values
set(adbis.columns.values) - set(new.columns.values)

# Adbis bases
set(adbis.base.unique()) - set(new.base.unique())
new_bases = list(set(new.base.unique()) - set(adbis.base.unique()))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_scores(data):
    res = dict()
    for b in data[~data.index.str.startswith('base')].index.unique():
        X_train = data[data.index != b].drop('Recall', axis=1)
        y_train = data[data.index != b]['Recall']
        test = data[data.index == b]
        test = test[test.nr_inst == test.nr_inst.max()]
        X_test = test.drop('Recall', axis=1)
        y_test = test['Recall']

        r2 = 0
        for i in range(5):
            reg = RandomForestRegressor(random_state=i, n_jobs=-1)
            # reg = DecisionTreeRegressor(random_state=i)
            reg.fit(X_train, y_train)
            r2 += reg.score(X_test, y_test)
        
        res[b] = r2/5


    res = {k: [v] for k,v in res.items()}
    res = pd.DataFrame.from_dict(res, orient='index')
    return res    

'''
    1) Old mfs + olds datasets (adbis version)
    2) Old mfs + new datasets
    3) New mfs + old datasets
    4) New mfs + New datasets
'''

all_scores = pd.DataFrame()

# 1) Old mfs + olds datasets (adbis version)
# from 84 columns to 35
data = new.loc[:, new.columns.isin(adbis_cols)].copy()
data = data[~data.base.isin(new_bases)]
data.set_index('base', inplace=True)

scores = get_scores(data)
scores = scores.reset_index().rename(columns={'index':'base', 0: 'r2'})
scores['method'] = 1
all_scores = pd.concat([all_scores, pd.DataFrame(scores)])

# 2) Old mfs + new datasets
data = new.loc[:, new.columns.isin(adbis_cols)].copy()
# data = data[~data.base.isin(new_bases)]
data.set_index('base', inplace=True)
scores = get_scores(data)
scores = scores.reset_index().rename(columns={'index':'base', 0: 'r2'})
scores['method'] = 2
all_scores = pd.concat([all_scores, pd.DataFrame(scores)])

# 3) New mfs + old datasets
data = new.copy()
data = data[~data.base.isin(new_bases)]
data.set_index('base', inplace=True)
scores = get_scores(data)
scores = scores.reset_index().rename(columns={'index':'base', 0: 'r2'})
scores['method'] = 3
all_scores = pd.concat([all_scores, pd.DataFrame(scores)])

# 4) New mfs + new datasets
data = new.copy()
# data = data[~data.base.isin(new_bases)]
data.set_index('base', inplace=True)
scores = get_scores(data)
scores = scores.reset_index().rename(columns={'index':'base', 0: 'r2'})
scores['method'] = 4
all_scores = pd.concat([all_scores, pd.DataFrame(scores)])
all_scores.groupby('method').r2.mean()

all_scores.method = all_scores.method.replace({
    1: 'old_mfs+old_bases',
    2: 'old_mfs+new_bases',
    3: 'new_mfs+old_bases',
    4: 'new_mfs+new_bases',
})

all_scores.to_csv('old_new_evaluation.csv', index=False)
