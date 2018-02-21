# encoding=utf-8
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import scipy as sp
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier #决策树
from src.planar_utils import plot_decision_boundary
import matplotlib.pyplot as plt
from src.classification import classfy

data = {'linear':[],
        'poly':[],
        'rbf':[],
        'sigmoid':[]}

linear_clf = SVC(random_state = 0, kernel = 'linear')
poly_clf = SVC(random_state = 0, kernel = 'poly')
rbf_clf = SVC(random_state = 0, kernel = 'rbf')
sigmoid_clf = SVC(random_state = 0, kernel = 'sigmoid')

for i in range(5):
    data['linear'].append(classfy(linear_clf))
    data['poly'].append(classfy(poly_clf))
    data['rbf'].append(classfy(rbf_clf))
    data['sigmoid'].append(classfy(sigmoid_clf))
x = [1, 2, 3, 4, 5]

plt.plot(x, data['linear'])
plt.plot(x, data['poly'])
plt.plot(x, data['rbf'])
plt.plot(x, data['sigmoid'])
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.show()