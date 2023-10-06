import pandas as pd

mf = pd.read_csv('data/metafeatures/new_metafeatures.csv', index_col=0)
mf.dropna(axis=1, inplace=True)
mf.drop('elapsed_time(secs)', axis=1, inplace=True)
# filtering meta-features (no subsets)
mf.reset_index(drop=False, inplace=True)
mf.n_pcs = mf.n_pcs / mf.nr_attr
mf = mf.rename(columns={'index': 'base'})
mf[mf.base.str.startswith('sift')].base.unique()
nus = ['Normalized_CH_266748', 'Normalized_CM55_269648', 'Normalized_CORR_269648', 'Normalized_EDH_269529', 'Normalized_WT_269648']
mnist = ['fashion_mnist_70000', 'mnist_1000', 'mnist_121d_70000', 'mnist_16000', 'mnist_196d_70000', 'mnist_2000', 'mnist_289d_70000', 'mnist_32000', 'mnist_4000', 'mnist_400d_70000', 'mnist_484d_70000', 'mnist_500', 'mnist_625d_70000', 'mnist_64000', 'mnist_69900', 'mnist_8000', 'mnist_background_50000', 'mnist_background_rotation_50000']
mf.base = mf.base.replace('_(\d+d)', r'\1', regex=True)
mf.base = mf.base.replace('fashion_mnist', 'fashion', regex=True)
mf.base = mf.base.replace('mnist_background_rotation', 'mnistBackgroundRotation', regex=True)
mf.base = mf.base.replace('mnist_background', 'mnistBackground', regex=True)
mf.base = mf.base.replace('deep1M_base_999771', 'deep1M_999900', regex=True)
mf.base = mf.base.apply(lambda x: x.replace('Normalized_', 'Nus') if x in nus else x)
print(*sorted(mf.base.unique()), sep='\n')
mf.nr_inst = mf.base.apply(lambda x: x.split('_')[1]).astype(int)
mf.base = mf.base.apply(lambda x: x.split('_')[0])
mf[mf.base == 'sift'].nr_inst.max()
mf.columns.sort_values()
mf.to_csv('data/metafeatures/new_metafeatures_pp_v3.csv', index=False)
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns
# import matplotlib.pyplot as plt


# mf.base = mf.base.apply(lambda x: 'nus' if x.startswith('Nus') else x)
# mf.base = mf.base.apply(lambda x: 'mnist' if x.startswith('mnist') else x)
# mf.base = mf.base.apply(lambda x: 'syn' if x.startswith('base') else x)
# mf.base = mf.base.apply(lambda x: 'colorHTM' if x in ['colorHisto', 'texture', 'moments'] else x)

# mf.set_index('base', inplace=True)
# # mf.reset_index(inplace=True)

# import utils
# mf = utils.read_mf()
# tmp = mf[(mf.index != 'syn') & (~mf.index.isin(['color', 'nus', 'aloi']))].copy()
# tmp.reset_index(inplace=True)
# tmp.base = tmp.base.apply(lambda x: x.split('_')[0])
# # bases = ['base71', 'base74', 'base68']
# # tmp.base = tmp.base.apply(lambda x: 'syn_' + x.split('base')[1] if x in bases else x)
# tmp.base = tmp.base.apply(lambda x: 'syn' if x.startswith('base') else x)
# tmp.set_index('base', inplace=True)
# sc = StandardScaler()
# tmp.loc[:, :] = sc.fit_transform(tmp.values)
# pca = PCA(2)
# df_pca = pd.DataFrame(pca.fit_transform(tmp.values), columns=['pc1', 'pc2'], index=tmp.index)
# df_pca.reset_index(inplace=True)
# plt.figure(figsize=(14, 8))
# # df_pca = df_pca[df_pca.base != 'syn']
# sns.scatterplot(x='pc1', y='pc2', hue='base', data=df_pca, s=100, palette='bright')
# # plt.savefig('/home/seidi/Desktop/pca.pdf', dpi=100)
# plt.show()

