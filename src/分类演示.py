import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.planar_utils import plot_decision_boundary, load_planar_dataset

X, Y = load_planar_dataset()
X = X + 4
plt.scatter(X[0, :], X[1, :], c = Y, s=40, cmap=plt.cm.Spectral)
# plt.show()

log_clf = LogisticRegressionCV()
log_clf.fit(X.T, Y.T)

mul_clf = MultinomialNB()
mul_clf.fit(X.T, Y.T)

svm_clf = SVC(random_state = 0, kernel = 'rbf')
svm_clf.fit(X.T, Y.T)

tree_clf = DecisionTreeClassifier(random_state = 0)
tree_clf.fit(X.T, Y.T)

plt.subplot(2, 2, 1)
plot_decision_boundary(lambda x: log_clf.predict(x), X, Y)
plt.title(u'logistic回归')


plt.subplot(2, 2, 2)
plot_decision_boundary(lambda x: mul_clf.predict(x), X, Y)
plt.title(u'朴素贝叶斯')


plt.subplot(2, 2, 3)
plot_decision_boundary(lambda x: svm_clf.predict(x), X, Y)
plt.title(u'SVM')


plt.subplot(2, 2, 4)
plot_decision_boundary(lambda x: tree_clf.predict(x), X, Y)
plt.title(u'决策树')

plt.show()

