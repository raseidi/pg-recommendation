import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from scipy.spatial.distance import pdist, squareform

def read_data(data_path):
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).copy()
    df.set_index('base', inplace=True)
    df.drop('uscities', inplace=True)

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
    sc = StandardScaler()
    sc.fit(df[statistical])
    df.loc[:, statistical] = sc.transform(df[statistical])
    df.loc[:, others] = df[others].apply(np.log)
    return df

def get_train_test(mb, base='mnist', target='Recall'):
    X_train = mb[mb.index!='mnist'].drop(['Recall', 'QueryTime', 'IndexTime', 'DistComp'], axis=1).copy()
    y_train = mb[mb.index!='mnist'][target].copy()
    X_test = mb[mb.index=='mnist'].drop(['Recall', 'QueryTime', 'IndexTime', 'DistComp'], axis=1).copy()
    y_test = mb[mb.index=='mnist'][target].copy()
    return X_train, y_train, X_test, y_test

class Node():
    def __init__(self, X, y, model='clf'):
        self.value = y.mean()
        # TODO: erase these variables for deploy, use it for testing
        # self.X = X
        # self.y = y
        self.model = model
        self.left = None
        self.right = None
        self.model = self._fit(X, y, model)

    @staticmethod
    def _fit(X, y, model):
        if model == 'clf':
            y = np.where(y > y.mean(), 1, 0)
            model = RandomForestClassifier(n_estimators=300)
        elif model == 'reg':
            model = RandomForestRegressor(n_estimators=300)
        else:
            print('Wrong model provided, please enter clf or reg.')
        
        return model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        left = None if self.left is None else self.left.value
        right = None if self.right is None else self.right.value
        return '(Value:{}, L:{}, R:{})'.format(np.round(self.value, 4),left, right)

class CrazyTree():
    def __init__(self, interval_size=1, min_samples_split=500, random_state=0):
        self.root = None
        self.interval_size = interval_size
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        # X = X.reset_index(drop=True).to_numpy()
        # y = X.reset_index(drop=True).to_numpy()
        self.root = self._build_tree(X, y)

    def predict(self, X):
        if self.root is None:
            print('Model not fitted.')
            return None
        
        y_pred = np.array([])
        for x in X:
            curr = self.root
            x = x.reshape(1, -1)
            while True:
                if curr.is_leaf():
                    y_pred = np.append(y_pred, curr.predict(x))
                    break
                else:
                    if curr.predict(x) == 1:
                        curr = curr.right
                    else:
                        curr = curr.left

        return y_pred

    def _build_tree(self, X, y):
        print(len(y))
        if self._is_reg(X, y):
            new_node = Node(X, y, 'reg')
        else:
            new_node = Node(X, y, 'clf')
            ix_left = y <= y.mean()
            ix_right = y > y.mean()
            new_node.left = self._build_tree(X[ix_left, :], y[ix_left])
            new_node.right = self._build_tree(X[ix_right, :], y[ix_right])
            
        return new_node

    def _is_reg(self, X, y):
        return y.max() - y.min() <= self.interval_size or\
            (len(y) <= self.min_samples_split)

    def __str__(self):
        if(self.root is not None):
            self._in_order(self.root)
        return '(in-order)'

    def _in_order(self, current):
        if current is not None:
            self._in_order(current.left)
            print(np.round(current.value, 2))
            self._in_order(current.right)

