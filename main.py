# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 12:07
# @Author  : quincyqiang
# @File    : main.py
# @Software: PyCharm
from model import bow
from model import tfidf
from model import word2vec
from dataset import load_data
from dataset import load_w2v


if __name__ == '__main__':
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    # 模型测试
    bow_clf=bow(X_train,X_test,y_train,y_test)
    tfidf_clf=tfidf(X_train,X_test,y_train,y_test)

    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec=load_w2v()
    word2vec_clf=word2vec(X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec)