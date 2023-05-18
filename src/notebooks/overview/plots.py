import os, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from itertools import product

plt.style.use('ggplot')
os.chdir('/home/seidi/Repositories/mestrado_final/')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

def read_recommendations(path):
    datasets = {
        'texture': 'Texture',
        'moments': 'Moments',
        'colorHisto': 'Histogram',
        'sift': 'SIFT',
        'mnist': 'MNIST',
        'mnist121d': 'MNIST121d',
        'fashion': 'Fashion-MNIST',
        'cophir282': 'Cophir282',
        'cophir64': 'Cophir64',
        'base71': 'Synthetic', 
        'base74': 'Syn74', 
        'base68': 'Syn68',
        'base72': 'Syn72',
        'base70': 'Syn70',
    }

    approaches = {
        'gmm': 'GMM',
        'gmm+': 'GMM+',
        'tmmgs': 'TMM-GS',
        'tmmgs+': 'TMM-GS+',
        'tmms': 'TMM-S',
        'tmms+': 'TMM-S+',  
        'Light': 'Loose',
        'Heavy': 'Tight'  
    }

    graph_type = {
        0: 'NSW',
        1: 'NNDescent',
        2: 'Brute-kNNG',  
    }

    recs = pd.read_csv(path)
    recs.Dataset = recs.Dataset.map(lambda x: datasets.get(x, x))
    recs.approach = recs.approach.map(lambda x: approaches.get(x, x))
    recs.graph_type = recs.graph_type.map(lambda x: graph_type.get(x, x))
    recs = recs.rename(columns={
        'approach': 'Approach',
        'graph_type': 'Graph Algorithm'
    })
    
    data_order = ['Moments', 'Histogram', 'Texture', 'MNIST121d', 'MNIST', 'Fashion-MNIST', 'Cophir64', 'Cophir282', 'SIFT', 'Synthetic']
    drop = list(set(data_order) - set(recs.Dataset.unique()))
    for x in drop:
        data_order.remove(x)
    recs.set_index('Dataset', inplace=True)
    recs = recs.loc[data_order, :].reset_index()
    return recs

root_path = 'src/notebooks/overview/'
recommendations_path = 'src/notebooks/overview/recommendations'
plots_path = 'src/notebooks/overview/plots'
RECOMMENDATIONS_TYPES = {
    'quick': ['Optimal', 'GMM', 'GMM+', 'Tight', 'Loose'], 
    'tuned': ['Optimal', 'GridSearch', 'TMM-GS', 'TMM-GS+', 'TMM-S', 'TMM-S+']
}
OPTIMIZING_VALUES = {
    'qt': 'Query Time (ms)', 
    'nn': 'NN', 
    'dt': 'Distance Computations'
}
REQUIRED_RECALLS = [[0.95], [0.9, 0.99]]

for file in os.listdir(recommendations_path):
    recs = read_recommendations(os.path.join(recommendations_path, file))
    k_values = re.search(r'k=\d+', file)[0]
    k_path = os.path.join(plots_path, k_values)
    try:
        os.mkdir(k_path)
    except:
        pass

    prods = product(RECOMMENDATIONS_TYPES.keys(), OPTIMIZING_VALUES.keys(), REQUIRED_RECALLS)
    for rec_type, opt, req_recall in prods:
        final_recommendations = recs[recs.Approach.isin(RECOMMENDATIONS_TYPES[rec_type])].copy()

        fig, axes = plt.subplots(len(req_recall), 1, figsize=(10, 6), dpi=300, sharex=True)
        axes = axes.reshape(-1) if isinstance(axes, np.ndarray) else [axes]
        for ax, recall in zip(axes, req_recall):
            df = final_recommendations[(final_recommendations.optmizing == opt) & (final_recommendations.required_recall == recall)].copy()
            g = sns.barplot(x='Dataset', y=OPTIMIZING_VALUES[opt], hue='Approach', hue_order=RECOMMENDATIONS_TYPES[rec_type], data=df, ax=ax)
            test = df.copy()
            test.set_index('Approach', inplace=True)
            drop = list(set(RECOMMENDATIONS_TYPES[rec_type]) - set(test.index.unique()))
            o = [h for h in RECOMMENDATIONS_TYPES[rec_type] if h not in drop]
            test = test.loc[o, :].reset_index()
            for ix, p in enumerate(g.patches):
                if ix == len(test):
                    break
                tmp = test.loc[ix, :]
                if tmp.Recall < tmp.required_recall:
                    g.annotate(
                        # format(p.get_height(), '.1f'),
                        tmp.Recall,
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9),
                        size=10,
                        rotation=60,
                        textcoords = 'offset points'
                        ) 
            if OPTIMIZING_VALUES[opt] != 'NN':
                ax.set_yscale('log')
            # if recall != 0.95:
            #     ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title(fr'Required recall$\geq{recall}$') # 
            ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5)
        plt.subplots_adjust(hspace=0.3)
        # plt.suptitle(OPTIMIZING_VALUES[opt])
        recalls = ','.join(map(str, req_recall))
        plt.savefig(f'{k_path}/{rec_type}_{opt}_{recalls}.png', bbox_inches = 'tight', pad_inches = 0, dpi=300)
        fig.clear()
        plt.close(fig)

# tuned_dt_0.9,0.99.png