import os
# os.chdir('../../')
import utils
import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

mb = utils.read_int_mb()
mf = utils.read_mf()

bases_reais = ['texture_67940', 'sift_999900', 'moments_67940', 'mnist121d_69900', 'fashion_69900','colorHisto_67940', 'mnist_69900']
MTS = ['Recall', 'DistComp', 'QueryTime', 'IndexTime']

res = pd.DataFrame()
for base in bases_reais:
    train = mb[mb.index != base]
    X_test = mb[mb.index == base].drop(MTS, axis=1)
    y_test = mb[mb.index == base].IndexTime

    for base_test in train.index.unique():
        X_train = train[train.index == base_test].drop(MTS, axis=1)
        y_train = train[train.index == base_test]['IndexTime']
        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)
        reg.score(X_test, y_test)

        tmp = pd.DataFrame({
            'base_target': [base],
            'base_model': [base_test],
            'r2_score': [reg.score(X_test, y_test)]
        })
        res = pd.concat([res, tmp])

pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 201)

res.reset_index(drop=True, inplace=True)
ix = res.groupby('base_target').r2_score.nlargest(10).index.get_level_values(1).values
best = res.loc[ix, :][['base_target', 'base_model', 'r2_score']].copy()
best.base_target = best.base_target.apply(lambda x: x.split('_')[0])
best.base_model = best.base_model.apply(lambda x: x.split('_')[0])
ix = best[best.base_target != best.base_model].groupby('base_target').r2_score.idxmax()
best.loc[ix, :]

final = res.loc[ix, :].copy()


fi = pd.DataFrame()
for test, train in final[['base_target', 'base_model']].values:
    X_test = mb[mb.index == test].drop(MTS, axis=1)
    y_test = mb[mb.index == test].IndexTime

    X_train = mb[mb.index == train].drop(MTS, axis=1)
    y_train = mb[mb.index == train].IndexTime

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    tmp = pd.DataFrame({
        'features': X_train.columns,
        'importances': reg.feature_importances_,
        'base': test
    })
    fi = pd.concat([fi, tmp])

fi.groupby('features').importances.mean().sort_values()


