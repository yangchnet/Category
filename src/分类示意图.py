import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from src.planar_utils import plot_decision_boundary

# mat = sio.loadmat('../paint_data/ex6data2.mat')
# data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
# X = np.array([data['X1'], data['X2']])
# data['y'] = mat.get('y')

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')
# ax.set_title('分类示意图')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# clf = sklearn.svm.SVC(random_state = 0, kernel = 'rbf')
# clf.fit(X.T, mat['y'])
# plot_decision_boundary(lambda x: clf.predict(x), X, mat['y'])
# plt.show()
#*************************************************************************

mat = sio.loadmat('../paint_data/ex6data2.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
X = np.array([data['X1'], data['X2']])
sns.set(context="notebook", style="white", palette=sns.diverging_palette(240, 10, n=2))
sns.lmplot('X1', 'X2', hue='y', data=data,
           size=5,
           fit_reg=False,
           scatter_kws={"s": 10}
          )
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')
ax.set_title('分类示意图')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
clf = sklearn.svm.SVC(random_state = 0, kernel = 'poly')
clf.fit(X.T, mat['y'])
plot_decision_boundary(lambda x: clf.predict(x), X, mat['y'])
plt.show()