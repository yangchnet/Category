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
import os

os.path.abspath('..')

stopwords = {'的', '我们', '致力与', '公司', '于', '是', '拥有', '展开', '具有', '企业', '以上', '提供', '一家'}

# #读取保存
# data_reviews = load_files("../data1", encoding="utf-8")#训练集
# sp.save('data.npy', data_reviews.data)
# sp.save('target.npy', data_reviews.target)

def classfy(clf):
    '''
    接收分类器对象，返回分类结果
    '''
    data = sp.load('data.npy')
    target = sp.load('target.npy')

    # 初始化TfidfVectorizer
    count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore', stop_words = stopwords)

    # 3 7 分
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.3)
    tf_train = count_vec.fit_transform(data_train)
    tf_test = count_vec.transform(data_test)

    clf.fit(tf_train, target_train)
    class_predicted = clf.predict(tf_test)

    report = classification_report(target_test, class_predicted)

    return np.mean(class_predicted == target_test), report

clf = SVC(random_state = 0, kernel = 'linear')
clf  =  DecisionTreeClassifier(random_state = 0)
a, report = classfy(clf)
print(a)
# print(report[-15:-10])
# print(report[-25:-20])
# print(report[-35:-30])
# # avg / total       0.46******0.44******0.45******1601
# #                      a   6  b
# print(report)