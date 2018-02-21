# encoding=utf-8
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier #决策树
import matplotlib.pyplot as plt
from src.classification import classfy

data = {"SVM":[],
        "MultinomialNB":[],
        "Tree":[],
        "logistic":[]}

svm_clf = SVC(random_state = 0, kernel = 'linear')
mul_clf = MultinomialNB()
tree_clf = DecisionTreeClassifier(random_state = 0)
logistic_clf = LogisticRegressionCV()


svm_pre, svm_report = classfy(svm_clf)
data["SVM"].append(svm_pre)
data["SVM"].append(float(svm_report[-25:-21]))
data["SVM"].append(float(svm_report[-15:-10]))
print("svm")

mul_pre, mul_report = classfy(mul_clf)
data["MultinomialNB"].append(mul_pre)
data["MultinomialNB"].append(float(mul_report[-25:-20]))
data["MultinomialNB"].append(float(mul_report[-15:-10]))
print("mul")

tree_pre, tree_report = classfy(tree_clf)
data["Tree"].append(tree_pre)
data["Tree"].append(float(tree_report[-25:-20]))
data["Tree"].append(float(tree_report[-15:-10]))
print("tree")

logistic_pre, logistic_report = classfy(logistic_clf)
data["logistic"].append(logistic_pre)
data["logistic"].append(float(logistic_report[-25:-20]))
data["logistic"].append(float(logistic_report[-15:-10]))
print("log")

N = 3
ind = np.arange(N)
X1 = [1, 3, 5]
X2 = [1.3, 3.3, 5.3]
X3 = [1.6, 3.6, 5.6]
X4 = [1.9, 3.9, 5.9]

plt.bar(X1, data["SVM"], color = 'r', width = .3, label ='SVM')
plt.bar(X2, data["MultinomialNB"], color = 'g',  width = .3, label =u'贝叶斯')
plt.bar(X3, data["Tree"], color = 'c', width = .3, label = u'决策树')
plt.bar(X4, data["logistic"], color = 'm', width = .3, label ='logistic')
plt.title(u'算法结果对比图')
plt.legend(loc='upper left', bbox_to_anchor=(0.9, 1.0), borderaxespad=0.)
plt.xticks([1.6, 3.6, 5.6], ['precision', 'recall', 'f1-score'])
plt.subplots_adjust(right = 0.9)
plt.show()
