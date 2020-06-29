
"""
CCA analysis
2019/2020
Author:   
        Jeremy Lefort-Besnard   jlefortbesnard (at) tuta (dot) io
"""


import numpy as np
import pandas as pd
import glob
import nibabel as nib
import nilearn.datasets as ds
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from matplotlib import pylab as plt
import seaborn as sns
from nilearn.signal import clean
from sklearn.preprocessing import LabelBinarizer

##############################
### PREPARING THE DATAFRAME ##
##############################

# extract path to sMRI data
sMRI_paths = glob.glob(
  'CHECHKO_RESTING_STATE/*/*/CAT/mri/mwp1*.nii')
sMRI_subnames = np.array(
  [p.split('mwp1')[-1].split('.nii')[0] for p in sMRI_paths])
df_right = pd.DataFrame(
  data=np.vstack((sMRI_subnames, sMRI_paths)).T,
  columns=['ID', 'data_path'])

# extract behavioral data
df = pd.read_excel('RLS_Gruppen_september8.xlsx')
df_merged = pd.merge(df, df_right, on='ID')
df_merged['RLS'] = df_merged.data_path.str.contains('RLS')



#####################################
##### EXTRACTING BEHAVIORAL DATA ####
#####################################

# extract patients data
df_RLS = df[df["RLS"]==True].reset_index(drop=True)
# 25 RLS patients
# for i in range(0, len(actual_cca.y_weights_)):
#     # pca_brain.inverse_transform(actual_cca.y_weights_[:, 0])
#     AAL_weights = pca_brain.inverse_transform(actual_cca.y_weights_[:, i])
#     out_nii_CCA = masker.inverse_transform(AAL_weights[None, :])
##########
# z scored the continuous features
##########
# cols_continuous = df_RLS.columns[3:-14]
cols_continuous = ['Age', 'Total amount of children', 'Highest degree of education', 'Income', 'days of gestation', 'Weight(gramm)Summe', 'Amount of Stressful live events', 'RLS_Wert_T0', 'RLS_Wert_T1', 'RLS-Beschwerden vorherige SS','Cortisol T0','EPDS T0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7','Q8', 'Q9', 'Q10']
df_RLS_continuous = df_RLS[cols_continuous]
X_continuous = df_RLS_continuous.values
X_continuous_zscored = StandardScaler().fit_transform(X_continuous)
df_RLS[cols_continuous] = X_continuous_zscored

# extract specific data needed for the analysis
col = df_RLS.columns[4:]
df_beh = df_RLS[col]
df_beh = df_beh.drop(['data_path'], axis=1)
# Dataframe into array for the CCA
X_cols = df_beh.columns
X_ss_beh = df_beh.values

#######
# restandardized continuous + categorical
#######
X_ss_beh =  StandardScaler().fit_transform(X_ss_beh)

#########
# PCA the beh
#########
pca_beh = PCA(n_components=10).fit(X_ss_beh)
X_ss_beh = pca_beh.transform(X_ss_beh)


###########################################
##### EXTRACTING STRUCTURAL BRAIN DATA ####
###########################################

tmp_nii = nib.load(df_RLS.data_path[0])
atlas = ds.fetch_atlas_aal()
ratlas_nii = resample_img(
  atlas.maps, target_affine=tmp_nii.affine, interpolation='nearest')
# ratlas_nii.to_filename('debug_ratlas.nii.gz')

# extracting data MRI
FS = []
for i_nii, nii_path in enumerate(df_RLS.data_path.values):
  print(nii_path)
  nii = nib.load(nii_path)
  cur_ratlas_nii = resample_img(
    atlas.maps, target_affine=nii.affine, interpolation='nearest')
  nii_fake4D = nib.Nifti1Image(
    nii.get_data()[:, :, :, None], affine=nii.affine)
  masker = NiftiLabelsMasker(labels_img=ratlas_nii)
  masker.fit()
  cur_FS = masker.transform(nii_fake4D)
  FS.append(cur_FS)
FS = np.array(FS).squeeze()
assert len(df_RLS.data_path.values) == len(FS)
ssFS = StandardScaler().fit_transform(FS)

np.random.seed(42)

# data brain
pca_brain = PCA(n_components=10).fit(ssFS)
Y_ss_brain_pca = pca_brain.transform(ssFS)


#############################
####### CCA Analysis ########
#############################

# test with pca:
# X_ss_beh = X_ss_beh_pca

n_keep = 10
n_permutations = 1001
actual_cca = CCA(n_components=n_keep)
actual_cca.fit(X_ss_beh, Y_ss_brain_pca)
actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
    zip(actual_cca.x_scores_.T, actual_cca.y_scores_.T)])

perm_rs = np.random.RandomState(42)
perm_Rs = []
n_except = 0
for i_iter in range(n_permutations):
    print(i_iter + 1)

    Y_brain_perm = np.array([perm_rs.permutation(sub_row) for sub_row in Y_ss_brain_pca])


    # same procedure, only with permuted subjects on the right side
    try:
        perm_cca = CCA(n_components=n_keep, scale=False)

        # perm_inds = np.arange(len(Y_netmet))
        # perm_rs.shuffle(perm_inds)
        # perm_cca.fit(X_nodenode, Y_netnet[perm_inds, :])
        perm_cca.fit(X_ss_beh, Y_brain_perm)

        perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in
            zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
        perm_Rs.append(perm_R)
    except:
        n_except += 1
        perm_Rs.append(np.zeros(n_keep))
perm_Rs = np.array(perm_Rs)

pvals = []
for i_coef in range(n_keep):
    cur_pval = (1. + np.sum(perm_Rs[1:, 0] > actual_Rs[i_coef])) / n_permutations
    pvals.append(cur_pval)
    print(cur_pval)
pvals = np.array(pvals)
print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))



################
## plot data: ##
################

AAL_weights = pca_brain.inverse_transform(actual_cca.y_weights_[:, 0])
out_nii_CCA = masker.inverse_transform(AAL_weights[None, :])
out_nii_CCA.to_filename('out_nii_CCA{}.nii'.format(1))

out_beh_weights = pca_beh.inverse_transform(actual_cca.x_weights_[:, 0])
out_beh_weights = out_beh_weights.reshape((36, 1))

# save data to dataframe for later analyses
original_beh = {}
original_brain = {}
original_beh["beh_original"] = np.squeeze(out_beh_weights)
original_brain["brain_original"] = AAL_weights
df_brain_vo = pd.DataFrame(data = original_brain)
df_beh_vo = pd.DataFrame(data = original_beh)
df_brain_vo.to_pickle("df_brain_vo")
df_beh_vo["variables"] = df_beh.columns
df_beh_vo = df_beh_vo.set_index("variables")
df_beh_vo.to_pickle("df_beh_vo")

def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


colnames = X_cols

# change colname into less geeky names
colnames = ['Age',
            'Total amount of children',
            'Highest degree of education',
            'Income',
            'days of gestation',
            'baby weight (gramm)',
            'Amount of Stressful live events',
            'RLS strength before child birth',
            'RLS strength after child birth',
            'Level of cortisol',
            'EPDS score',
            'Level of clinical manifestation of RLS',
            'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
            'No birth-related complications',
            'Birth-related complications',
            'No pregnancy-related complications',
            'Pregnancy-related complications',
            'With partner, not married',
            'With partner, married',
            'No breast feeding',
            'Breast feeding',
            'No relocation to other ward',
            'Relocation to other ward',
            'Birth of a single child',
            'Twin birth',
            'No iron supplement',
            'Iron supplement']

f, ax = plt.subplots(figsize=(12, 7))
out_beh_weights = np.round_(out_beh_weights, decimals=1, out=None)
sns.heatmap(out_beh_weights.T, cmap='RdBu_r', square=True, annot=True, cbar_kws={"shrink": .25})
# plt.xticks(range(len(X_colnames)), X_colnames)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_ticklabels(colnames, fontsize=12)
rotateTickLabels(ax, 45, 'x')
y_ticks = [""]
ax.yaxis.set_ticklabels(y_ticks, fontsize=12, rotation=0)
plt.ylabel('Significant component', fontsize=14)
plt.tight_layout()
plt.savefig('CCA_beh_.png')
plt.show()
