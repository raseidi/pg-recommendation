import os
os.chdir('../../')
import utils
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor

mb = utils.read_int_mb()
mb[~mb.index.str.startswith('base')].nr_num


targets = ['Recall', 'QueryTime', 'IndexTime', 'DistComp']
X = mb.drop(targets, axis=1).copy()
y = mb.loc[:, targets].copy()

PREDICTIONS = pd.DataFrame()
SCORES = pd.DataFrame()
for target in targets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33, stratify=mb.index)
    
    models = utils.fit_models(
        X_train, y_train[target],
        model=RandomForestRegressor,
        n_models=10
    )
    y_pred = utils.ensamble_predictions(models, X_test)
    
    # Formatting scores
    kwargs = {
        'target': target,
    }
    scores_tmp = utils.format_scores(y_test[target], y_pred, **kwargs)
    SCORES = pd.concat([SCORES, scores_tmp])

    # Foramtting predictions
    kwargs = {
        'target': target,
    }
    if 'k_searching' not in X_test.columns:
        X_test['k_searching'] = 30
    preds_tmp = utils.format_predictions(X_test, y_test[target], y_pred, **kwargs)
    PREDICTIONS = pd.concat([PREDICTIONS, preds_tmp])

    SCORES.to_csv('data/results/overall_performance/scores.csv', index=False)
    PREDICTIONS.to_csv('data/results/overall_performance/predictions.csv', index=False)


