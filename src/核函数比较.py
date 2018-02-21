# encoding=utf-8
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier #决策树
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
    a1, r1 = classfy(linear_clf)
    a2, r2 = classfy(poly_clf)
    a3, r3 = classfy(rbf_clf)
    a4, r4 = classfy(sigmoid_clf)
    data['linear'].append(a1)
    data['poly'].append(a2)
    data['rbf'].append(a3)
    data['sigmoid'].append(a4)

x = [1, 2, 3, 4, 5]

plt.plot(x, data['linear'], label = "Linear")
plt.plot(x, data['poly'], label = "Poly")
plt.plot(x, data['rbf'], label = "rbf")
plt.plot(x, data['sigmoid'], label = "sigmoid")
plt.legend(loc=0, shadow=True, fontsize='x-large')
plt.xlabel(u'实验次数')
plt.ylabel(u'预测准确率')
plt.title(u'核函数对比图')
plt.show()