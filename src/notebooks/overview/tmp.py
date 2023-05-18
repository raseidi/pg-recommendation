import numpy as np
import pandas as pd

import utils
mf = utils.read_mf()
mb = utils.read_int_mb()

mb[mb.index.str.startswith('base71_')].index.nunique()
mb[mb.index.str.startswith('moments')].index.nunique()

mb.IndexTime = mb.IndexTime.apply(np.exp)
ad2_scores = pd.read_csv('data/results/info_sys_interpolated/adbis/scores.csv')
ad2_scores = ad2_scores[ad2_scores.k_searching == 30]
# ad2_scores = pd.read_csv('data/results/adbis_interpolated/scores.csv')
clus_scores = pd.read_csv('data/results/info_sys_interpolated/clustering_tuned/scores.csv')
clus_scores = clus_scores.rename(columns={'method': 'approach'})
clus_scores.approach = clus_scores.approach + '+'
clus_scores = clus_scores[clus_scores.k_searching == 30]
nn = [5., 25., 70., 150.]
rr = [1., 10., 40., 120.]
REAL_BASES = [
    'texture_67940', 'sift_999900', 'moments_67940',
    'mnist121d_69900', 'fashion_69900',
    'colorHisto_67940', 'mnist_69900',
    'base71_999900', 'cophir64_999900', 'cophir282_999900'
]

mb.reset_index(inplace=True)
mb=mb[mb.base.isin(REAL_BASES)]
mb = mb.drop_duplicates('base')
mb.set_index('base').lid_mean.sort_values().round(2)
mb.set_index('base').nr_attr.apply(np.exp)
mb.lid_mean.max()

cols = ['base', 'approach', 'ix_time', 'qt_time', 'total_time', 'tt_recall', 'rt_recall', 'tt_querytime', 'rt_querytime', 'tt_indextime', 'rt_indextime', 'tt_distcomp', 'rt_distcomp']
recommendation_time = pd.DataFrame(columns=cols)

# GS TIME
for b in REAL_BASES:
    tmp = mb[
        (mb.index == b) &
        (mb.IndexParams.isin(nn)) &
        (mb.QueryTimeParams.isin(rr))
    ]

    # b = b.split('_')[0]
    ix_time = tmp.IndexTime.sum()
    qt_time = tmp.QueryTime.sum()
    total_time = ix_time + qt_time
    new = pd.DataFrame([[b, 'gs', ix_time, qt_time, total_time, 0, 0, 0, 0, 0, 0, 0, 0]], columns=cols)
    recommendation_time = recommendation_time.append(new)

# TMM-GS TIME
for b in REAL_BASES:
    # b = b.split('_')[0]
    tmp = ad2_scores[(ad2_scores.base == b) & (ad2_scores.approach=='tmm_gs')]
    tmp_gs = recommendation_time[(recommendation_time.approach == 'gs') & (recommendation_time.base == b)]
    assert len(tmp_gs) == 1
    new = pd.DataFrame({
        'base': [b],
        'approach': ['tmmgs'],
        'ix_time': tmp_gs.ix_time.values,
        'qt_time': tmp_gs.qt_time.values,
        'total_time': [tmp.elapsed_training.sum() + tmp.elapsed_inference.sum() + tmp_gs.ix_time.values.sum() + tmp_gs.qt_time.values.sum()],
        'tt_recall': tmp[tmp.target=='Recall'].elapsed_training.values,
        'rt_recall': tmp[tmp.target=='Recall'].elapsed_inference.values,
        'tt_querytime': tmp[tmp.target=='QueryTime'].elapsed_training.values,
        'rt_querytime': tmp[tmp.target=='QueryTime'].elapsed_inference.values,
        'tt_indextime': tmp[tmp.target=='IndexTime'].elapsed_training.values,
        'rt_indextime': tmp[tmp.target=='IndexTime'].elapsed_inference.values,
        'tt_distcomp': tmp[tmp.target=='DistComp'].elapsed_training.values,
        'rt_distcomp': tmp[tmp.target=='DistComp'].elapsed_inference.values,
    })
    recommendation_time = recommendation_time.append(new)

# TMM-GS+
for b in REAL_BASES:
    # b = b.split('_')[0]
    tmp = clus_scores[(clus_scores.base == b) & (clus_scores.approach=='tmmgs+')]
    tmp_gs = recommendation_time[(recommendation_time.approach == 'gs') & (recommendation_time.base == b)]
    assert len(tmp_gs) == 1
    new = pd.DataFrame({
        'base': [b],
        'approach': ['tmmgs+'],
        'ix_time': tmp_gs.ix_time.values,
        'qt_time': tmp_gs.qt_time.values,
        'total_time': [tmp.elapsed_training.sum() + tmp.elapsed_inference.sum() + tmp_gs.ix_time.values.sum() + tmp_gs.qt_time.values.sum()],
        'tt_recall': tmp[tmp.target=='Recall'].elapsed_training.values,
        'rt_recall': tmp[tmp.target=='Recall'].elapsed_inference.values,
        'tt_querytime': tmp[tmp.target=='QueryTime'].elapsed_training.values,
        'rt_querytime': tmp[tmp.target=='QueryTime'].elapsed_inference.values,
        'tt_indextime': tmp[tmp.target=='IndexTime'].elapsed_training.values,
        'rt_indextime': tmp[tmp.target=='IndexTime'].elapsed_inference.values,
        'tt_distcomp': tmp[tmp.target=='DistComp'].elapsed_training.values,
        'rt_distcomp': tmp[tmp.target=='DistComp'].elapsed_inference.values,
    })
    recommendation_time = recommendation_time.append(new)

# TMM-S TIME
for b in REAL_BASES:
    # b = b.split('_')[0]
    tmp = ad2_scores[(ad2_scores.base == b) & (ad2_scores.approach=='tmm_s')]
    # subsets
    subsets = sorted(mb[mb.index.str.startswith(b.split('_')[0])].index.unique())
    subsets = sorted(list(set(map(lambda x: int(x.split('_')[1]), subsets))))[:-1]
    subsets = [b.split('_')[0] + '_' + str(s) for s in subsets]
    tmp_s = mb[(mb.index.isin(subsets))]
    ix_time = tmp_s.IndexTime.sum()
    qt_time = tmp_s.QueryTime.sum()
    total_time = ix_time + qt_time
    # subsets
    new = pd.DataFrame({
        'base': [b],
        'approach': ['tmms'],
        'ix_time': [ix_time],
        'qt_time': [qt_time],
        'total_time': [total_time + tmp.elapsed_training.sum() + tmp.elapsed_inference.sum()],
        'tt_recall': tmp[tmp.target=='Recall'].elapsed_training.values,
        'rt_recall': tmp[tmp.target=='Recall'].elapsed_inference.values,
        'tt_querytime': tmp[tmp.target=='QueryTime'].elapsed_training.values,
        'rt_querytime': tmp[tmp.target=='QueryTime'].elapsed_inference.values,
        'tt_indextime': tmp[tmp.target=='IndexTime'].elapsed_training.values,
        'rt_indextime': tmp[tmp.target=='IndexTime'].elapsed_inference.values,
        'tt_distcomp': tmp[tmp.target=='DistComp'].elapsed_training.values,
        'rt_distcomp': tmp[tmp.target=='DistComp'].elapsed_inference.values,
    })
    recommendation_time = recommendation_time.append(new)

# TMM-S+ TIME
for b in REAL_BASES:
    # b = b.split('_')[0]
    tmp = clus_scores[(clus_scores.base == b) & (clus_scores.approach=='tmms+')]
    # subsets
    subsets = sorted(mb[mb.index.str.startswith(b.split('_')[0])].index.unique())
    subsets = sorted(list(set(map(lambda x: int(x.split('_')[1]), subsets))))[:-1]
    subsets = [b.split('_')[0] + '_' + str(s) for s in subsets]
    tmp_s = mb[(mb.index.isin(subsets))]
    ix_time = tmp_s.IndexTime.sum()
    qt_time = tmp_s.QueryTime.sum()
    total_time = ix_time + qt_time
    # subsets
    new = pd.DataFrame({
        'base': [b],
        'approach': ['tmms+'],
        'ix_time': [ix_time],
        'qt_time': [qt_time],
        'total_time': [total_time + tmp.elapsed_training.sum() + tmp.elapsed_inference.sum()],
        'tt_recall': tmp[tmp.target=='Recall'].elapsed_training.values,
        'rt_recall': tmp[tmp.target=='Recall'].elapsed_inference.values,
        'tt_querytime': tmp[tmp.target=='QueryTime'].elapsed_training.values,
        'rt_querytime': tmp[tmp.target=='QueryTime'].elapsed_inference.values,
        'tt_indextime': tmp[tmp.target=='IndexTime'].elapsed_training.values,
        'rt_indextime': tmp[tmp.target=='IndexTime'].elapsed_inference.values,
        'tt_distcomp': tmp[tmp.target=='DistComp'].elapsed_training.values,
        'rt_distcomp': tmp[tmp.target=='DistComp'].elapsed_inference.values,
    })
    recommendation_time = recommendation_time.append(new)

# GMM+ TIME
for b in REAL_BASES:
    # b = b.split('_')[0]
    tmp = clus_scores[(clus_scores.base == b) & (clus_scores.approach=='gmm+')]
    new = pd.DataFrame({
        'base': [b],
        'approach': ['gmm+'],
        'ix_time': [0],
        'qt_time': [0],
        'total_time': [tmp.elapsed_inference.sum()],
        'tt_recall': tmp[tmp.target=='Recall'].elapsed_training.values,
        'rt_recall': tmp[tmp.target=='Recall'].elapsed_inference.values,
        'tt_querytime': tmp[tmp.target=='QueryTime'].elapsed_training.values,
        'rt_querytime': tmp[tmp.target=='QueryTime'].elapsed_inference.values,
        'tt_indextime': tmp[tmp.target=='IndexTime'].elapsed_training.values,
        'rt_indextime': tmp[tmp.target=='IndexTime'].elapsed_inference.values,
        'tt_distcomp': tmp[tmp.target=='DistComp'].elapsed_training.values,
        'rt_distcomp': tmp[tmp.target=='DistComp'].elapsed_inference.values,
    })
    recommendation_time = recommendation_time.append(new)

# GMM TIME
for b in REAL_BASES:
    # b = b.split('_')[0]
    tmp = ad2_scores[(ad2_scores.base == b) & (ad2_scores.approach=='gmm')]
    new = pd.DataFrame({
        'base': [b],
        'approach': ['gmm'],
        'ix_time': [0],
        'qt_time': [0],
        'total_time': [tmp.elapsed_inference.sum()],
        'tt_recall': tmp[tmp.target=='Recall'].elapsed_training.values,
        'rt_recall': tmp[tmp.target=='Recall'].elapsed_inference.values,
        'tt_querytime': tmp[tmp.target=='QueryTime'].elapsed_training.values,
        'rt_querytime': tmp[tmp.target=='QueryTime'].elapsed_inference.values,
        'tt_indextime': tmp[tmp.target=='IndexTime'].elapsed_training.values,
        'rt_indextime': tmp[tmp.target=='IndexTime'].elapsed_inference.values,
        'tt_distcomp': tmp[tmp.target=='DistComp'].elapsed_training.values,
        'rt_distcomp': tmp[tmp.target=='DistComp'].elapsed_inference.values,
    })
    recommendation_time = recommendation_time.append(new)

