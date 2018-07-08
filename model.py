# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 11:30
# @Author  : quincyqiang
# @File    : model.py
# @Software: PyCharm
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
# 结果评估
def get_metrics(y_test,y_predicted):
    """

    :param y_test: 真实值
    :param y_predicted: 预测值
    :return:
    """
    # 精确度=真阳性/（真阳性+假阳性）
    precision=precision_score(y_test,y_predicted,pos_label=None,average='weighted')
    # 召回率=真阳性/（真阳性+假阴性）
    recall=recall_score(y_test,y_predicted,pos_label=None,average='weighted')

    # F1
    f1=f1_score(y_test,y_predicted,pos_label=None,average='weighted')

    # 精确率
    accuracy=accuracy_score(y_test,y_predicted)
    return accuracy,precision,recall,f1

# 词袋模型
# 声明文本体征提取方法
def bow(X_train,X_test,y_train,y_test):

    def cv(data):
        # 把每一个单词都进行统计，同时计算每个单词出现的次数，默认过滤单词字符
        count_vectorizer=CountVectorizer(token_pattern=r'\b\w+\b')
        embedding=count_vectorizer.fit_transform(data)
        return embedding,count_vectorizer
    # 文本特征提取
    print("正在提取词袋体征")
    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    # print(X_train_counts)
    bow_path='data/result/bow.pkl'
    if os.path.exists(bow_path):
        print("正在加载已经训练的模型...")
        with open(bow_path, 'rb') as out_data:
            clf_bow = pickle.load(out_data)
    else:
        print("正在训练bow模型")
        # 逻辑回归模型进行分类
        clf_bow = LogisticRegression(C=10.0, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
        # 训练模型
        clf_bow.fit(X_train_counts,y_train)

        # 保存训练的模型
        with open(bow_path, 'wb') as in_data:
            pickle.dump(clf_bow, in_data, pickle.HIGHEST_PROTOCOL)
            print("bow model saved:" + bow_path)

    # 模型预测
    y_predicted_counts=clf_bow.predict(X_test_counts)
    # 模型评估
    accuracy,precision,recall,f1=get_metrics(y_test,y_predicted_counts)
    print("accuracy=%.6f,precision=%.6f,recall=%.6f,f1=%.6f" % (accuracy,precision,recall,f1))

    return clf_bow

# tfidf
def tfidf(X_train,X_test,y_train,y_test):

    def tfidf(data):
        tfidf_vectorizer = TfidfVectorizer()
        train = tfidf_vectorizer.fit_transform(data)
        return train, tfidf_vectorizer
    # 文本特征提取
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    tfidf_path = 'data/result/tfidf.pkl'
    if os.path.exists(tfidf_path):
        print("正在加载已经训练的模型...")
        with open(tfidf_path, 'rb') as out_data:
            clf_tfidf = pickle.load(out_data)
    else:
        print("正在训练tfidf模型...")
        # 声明模型
        clf_tfidf = LogisticRegression(C=10.0, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
        # 训练
        clf_tfidf.fit(X_train_tfidf, y_train)
        # 保存训练的模型
        with open(tfidf_path, 'wb') as in_data:
            pickle.dump(clf_tfidf, in_data, pickle.HIGHEST_PROTOCOL)
            print("tfidf model saved:" + tfidf_path)

    # 预测结果
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    ##模型评估
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.6f, precision = %.6f, recall = %.6f, f1 = %.6f" % (
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))

    return clf_tfidf

def word2vec(X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec):

    word2vec_path = 'data/result/word2vec.pkl'
    if os.path.exists(word2vec_path):
        print("正在加载已经训练的word2vec模型...")
        with open(word2vec_path, 'rb') as out_data:
            clf_wordvec = pickle.load(out_data)
        return clf_wordvec
    else:
        clf_wordvec = LogisticRegression(C=10.0, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
        clf_wordvec.fit(X_train_word2vec, y_train_word2vec)
        y_predicted_word2vec = clf_wordvec.predict(X_test_word2vec)
        accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test_word2vec,
                                                                                          y_predicted_word2vec)
        print("accuracy = %.6f, precision = %.6f, recall = %.6f, f1 = %.6f" % (
            accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec))

        # 保存训练的模型
        with open(word2vec_path, 'wb') as in_data:
            pickle.dump(clf_wordvec, in_data, pickle.HIGHEST_PROTOCOL)
            print("word2vec model saved:" + word2vec_path)