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

DATASETS = {
    'Original': {
        'texture_67940': 'Texture_70k',
        # 'moments_67940': 'Moments_70k',
        # 'colorHisto_67940': 'Histogram_70k',
        'mnist_69900': 'MNIST_70k',
        'mnist121d_69900': 'MNIST121d_70k',
        'fashion_69900': 'FashionMNIST_70k',
        'sift_999900': 'SIFT_1M',
        'cophir64_999900': 'Cophir64_1M',
        'cophir282_999900': 'Cophir282_1M',
        'base71_999900': 'Synthetic_1M',
    },
    
    'Cardinality=50%': {
        'texture_32000': 'Texture_35k',
        # 'moments_32000': 'Moments_35k',
        # 'colorHisto_32000': 'Histogram_35k',
        'mnist_32000': 'MNIST_35k',
        'mnist121d_32000': 'MNIST121d_35k',
        'fashion_32000': 'FashionMNIST_35k',
        'sift_500000': 'SIFT_500k',
        'cophir64_500000': 'Cophir64_500k',
        'cophir282_500000': 'Cophir282_500k',
        'base71_500000': 'Synthetic_500k',
    },
    
    'Cardinality<=20%': {
        'texture_16000': 'Texture_16k',
        # 'moments_16000': 'Moments_16k',
        # 'colorHisto_16000': 'Histogram_16k',
        'mnist_16000': 'MNIST_16k',
        'mnist121d_16000': 'MNIST121d_16k',
        'fashion_16000': 'FashionMNIST_16k',
        'sift_100000': 'SIFT_100k',
        'cophir64_100000': 'Cophir64_100k',
        'cophir282_100000': 'Cophir282_100k',
        'base71_100000': 'Synthetic_100k',
    }
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

def read_recommendations(path):
    # path = os.path.join(recommendations_path, f)
    recs = pd.read_csv(path)
    ds = {}
    for d in DATASETS.values():
        ds.update(d)

    recs.Dataset = recs.Dataset.map(lambda x: ds.get(x, x))
    recs.approach = recs.approach.map(lambda x: approaches.get(x, x))
    recs.graph_type = recs.graph_type.map(lambda x: graph_type.get(x, x))
    recs = recs.rename(columns={
        'approach': 'Approach',
        'graph_type': 'Graph Algorithm'
    })
    
    recs.set_index('Dataset', inplace=True)
    if re.findall('k=10', path):
        del ds['base71_999900']

    recs = recs.loc[ds.values(), :].reset_index()
    return recs

def annotate_wrong_recs(g, df, req_recall):
    for ix, p in enumerate(g.patches):
        if ix == len(df):
            print('err')
            raise
        tmp = df.iloc[ix, :]
        if tmp.Recall < req_recall:
            g.annotate(
                tmp.Recall.round(2),
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9),
                size=10,
                rotation=60,
                textcoords = 'offset points'
            )

def format_to_plot(df, rec_type):
    # df = final_recommendations[(final_recommendations.k_searching == k)]
    df.set_index('Approach', inplace=True)
    drop = list(set(RECOMMENDATIONS_TYPES[rec_type]) - set(df.index.unique()))
    o = [h for h in RECOMMENDATIONS_TYPES[rec_type] if h not in drop]
    df = df.loc[o, :].reset_index()
    return df

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
REQUIRED_RECALLS = [0.95, 0.90, 0.99]

recs = pd.DataFrame()
for f in os.listdir(recommendations_path):
    df = read_recommendations(os.path.join(recommendations_path, f))
    recs = pd.concat((recs, df))

'''
OPTIMIZING_VALUES.keys(), 
sorted(DATASETS.keys()), 
sorted(recs.k_searching.unique()),
REQUIRED_RECALLS,
RECOMMENDATIONS_TYPES.keys()
for each opt:
    for each cardinality:
        for each k_searching:
            for each recall:
                for each rec_type:

'''
prods = product(sorted(recs.k_searching.unique()), OPTIMIZING_VALUES.keys(), REQUIRED_RECALLS, RECOMMENDATIONS_TYPES.keys())
# prods = product(sorted(recs.k_searching.unique()), OPTIMIZING_VALUES.keys(), REQUIRED_RECALLS, ['tuned'])
for k_searching, opt, req_recall, rec_type in prods:
    req_recall = .95
    final_recommendations = recs[
        (recs.Approach.isin(RECOMMENDATIONS_TYPES[rec_type])) &
        (recs.k_searching == k_searching) &
        # (recs.Dataset.isin(DATASETS[cardinality].values())) & 
        (recs.optmizing == opt) &
        (recs.required_recall == req_recall)
    ].copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 6), dpi=100) #, sharex=True)
    axes = axes.reshape(-1) if isinstance(axes, np.ndarray) else [axes]
    for ax, cardinality in zip(axes, sorted(DATASETS.keys())):
        df = format_to_plot(final_recommendations[(final_recommendations.Dataset.isin(DATASETS[cardinality].values()))], rec_type)
        g = sns.barplot(x='Dataset', y=OPTIMIZING_VALUES[opt], hue='Approach', hue_order=RECOMMENDATIONS_TYPES[rec_type], data=df, ax=ax)
        annotate_wrong_recs(g, df, req_recall)
        if OPTIMIZING_VALUES[opt] != 'NN':
            ax.set_yscale('log')

        ax.set_xlabel('')
        # if cardinality == 'Cardinality<=20%':
        #     if req_recall != 0.95:
        #         ax.set_title(f'Required recall $\geq{req_recall}$, k={k_searching:.0f}')
        #     else:
        #         ax.set_title(f'k={k_searching:.0f}')
        if cardinality != 'Cardinality=50%':
            ax.set_ylabel('')
        # ax.set_title(fr'Required recall $\geq{req_recall}$')
        ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-.05))

    plt.subplots_adjust(hspace=0.3)
    req_recall = req_recall*100
    plt.savefig(f'/home/seidi/Repositories/seidi_pgac_journal/figures/results_section/results/new_recommendations/major_reviews/{rec_type}_{opt}_{req_recall:.0f}_k={k_searching:.0f}.pdf', bbox_inches = 'tight', pad_inches = 0, dpi=100)
    # plt.savefig(f'/home/seidi/Desktop/a.png', bbox_inches = 'tight', pad_inches = 0, dpi=100)
    fig.clear()
    plt.close(fig)
    del final_recommendations
    del df
    del fig



final_recommendations = recs[(recs.required_recall == 0.95) & (recs.k_searching.isin([30]))]
# Melt: columns to rows; Pivot: rows to columns

final_recommendations['Acertou'] = np.where(
    final_recommendations.Recall >= final_recommendations.required_recall, True, False
)

final_recommendations.Dataset.unique()
x=final_recommendations[~final_recommendations.isin(['Optimal'])].groupby('Approach').Acertou.value_counts()
soma = final_recommendations[~final_recommendations.isin(['Optimal'])].Approach.value_counts()
x.name = 'k'
x = x.reset_index(level=1)
a = (x.k/soma).to_frame()
a['Acertou'] = x.Acertou
a.reset_index(inplace=True)
a = a.rename(columns={0: 'rate', 'index': 'approach'})
a = a.append(pd.Series(['GridSearch', False, 0], index=['approach', 'Acertou', 'rate']).T, ignore_index=True)
a.Acertou = np.where(a.Acertou == True, 'Yes', 'No')
a = a.rename(columns={'Acertou': 'Satisfied'})
a.pivot(index='Satisfied', columns='approach', values='rate').loc[:, ['GridSearch', 'Loose', 'Tight', 'GMM', 'TMM-GS', 'TMM-S', 'GMM+', 'TMM-GS+', 'TMM-S+']].T['Yes'].sort_values(ascending=False)


recs[recs.Approach == 'GMM']