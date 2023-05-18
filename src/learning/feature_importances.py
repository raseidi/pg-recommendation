import os
os.chdir('../../')
import utils
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
mb = utils.read_int_mb()

mb.drop('Recall', axis=1, inplace=True)
META_TARGETS = ['DistComp', 'QueryTime', 'IndexTime']

f_imp = pd.read_csv('data/feature_importances.csv')
f_imp.columns
train = mb.drop(META_TARGETS, axis=1).copy()
mb.DistComp.max()
mb.columns

target = 'Recall'
mb.Recall
for target in META_TARGETS:
    print('\t\t\t' + target)
    y = mb[target].copy()
    cv = cross_validate(DecisionTreeRegressor(), train, y, cv=10, n_jobs=-1, return_estimator=True, scoring='neg_mean_squared_error')

    sum_importances = np.array(0)
    for reg in cv['estimator']:
        sum_importances = sum_importances + reg.feature_importances_

    fi = pd.DataFrame({
        'feature': train.columns,
        'importance': sum_importances / len(cv['estimator']),
        'target': target
    })
    print('dataframe criado')
    fi = fi[~fi.feature.isin(['IndexParams', 'QueryTimeParams', 'graph_type', 'k_searching'])]

    try:
        f_imp = pd.concat([f_imp, fi])
    except:
        print('Error no concat')
        print(f_imp)
        print('--------------')
        print(fi)
        break


f_imp.to_csv('data/feature_importances.csv', index=False)
