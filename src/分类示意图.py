import numpy as np
import pandas as pd
import sklearn
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from src.planar_utils import plot_decision_boundary

mat = sio.loadmat('../paint_data/ex6data1.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
X = np.array([data['X1'], data['X2']])
data['y'] = mat.get('y')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'],  cmap='Reds')

ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()#线性数据未分类

svm_clf = sklearn.svm.SVC(random_state = 0, kernel = 'linear')
svm_clf.fit(X.T, mat['y'])
plot_decision_boundary(lambda x: svm_clf.predict(x), X, mat['y'])
plt.show()#线性数据分类

#---------------------------------------------------------------------

mat = sio.loadmat('../paint_data/ex6data2.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
X = np.array([data['X1'], data['X2']])
data['y'] = mat.get('y')

fig1, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'],  cmap='Reds')

ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()#非线性数据

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(data['X1']**2, data['X2']**2, data['y'], s=30, c=data['y'],  cmap='Reds')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()#3维投影

