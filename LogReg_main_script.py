
"""
Logistic Regression analysis
2019
Author:  
	Danilo Bzdok: danilo (dot) bzdok (at) mcgill (dot) ca
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('RLS_Gruppen_september8.xlsx')


input_data = df[cols].values
input_data_norm = StandardScaler().fit_transform(input_data)
target = df['RLS_Wert_T1'].values


clf = LinearRegression()
clf.fit(input_data_norm, target)
sorted_inds = np.argsort(np.abs(clf.coef_))[::-1]
list(zip(clf.coef_[sorted_inds], np.array(cols)[sorted_inds]))

from matplotlib import pylab as plt
import seaborn as sns

plt.figure(figsize=(8, 8))
sns.heatmap(clf.coef_[:, None], cmap=plt.cm.RdBu_r, center=0)
plt.yticks(np.arange(len(cols)) + 0.5, np.array(cols), rotation=0)
plt.xticks([])
plt.title('Healthy and RLS subjects: classification contributions', fontsize=13)
plt.tight_layout()
plt.savefig('T1.png', DPI=500)
plt.show()


		 ]
