# -*- coding: utf-8 -*-
# @Time    : 2018/7/7 20:37
# @Author  : quincyqiang
# @File    : data.py
# @Software: PyCharm

# 导入库
import os
import pickle
import pandas as pd
import numpy as np
import re
import jieba
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# wordd2vec
import gensim
from gensim.models import Word2Vec


def load_data():
    data_path = "data/datasets.pkl"
    if os.path.exists(data_path):
        print("正在加载已处理后的数据...")
        with open(data_path, 'rb') as out_data:
            # 按保存变量的顺序加载变量
            datasets = pickle.load(out_data)
        return datasets
    else:
        print("正在处理数据。。。")
        # 预处理
        data = pd.read_csv('data/dataset7classes.csv', encoding="utf-8", header=None)
        data.columns = ['class_label', 'text']

        # 去除空值
        data['text'].fillna(np.NaN).head(1)
        data.dropna(inplace=True)
        # print(data.info())

        # 如果正则表达式清洗非中文字符
        re_data = []
        for i in data["text"]:
            i_re = ''.join(re.findall(r'[\u4e00-\u9fa5]', i))
            re_data.append(i_re.strip())
        data["text"] = re_data

        # 类别标签数值化
        label = preprocessing.LabelEncoder().fit_transform(data['class_label'])
        data['class_label'] = label

        # 分词
        # 加载自定义词典
        jieba.load_userdict('data/dict_out.csv')
        # 加载停用词表
        lines = open('data/stopwords.dat', 'r', encoding='utf-8')
        stop_words = [line.strip() for line in lines]
        # print(stop_words)
        word_list = []
        words_list = []
        for sent in data['text']:
            try:
                words = jieba.cut(sent)
                words = [word for word in words if word not in stop_words]
                segmented_words = ','.join(words)
            except AttributeError:
                continue
            finally:
                words_list.append(words)
                word_list.append(segmented_words.strip())
        data['tokens'] = word_list
        # 划分数据集和测试集
        X_train, X_test, y_train, y_test = \
            train_test_split(data['tokens'], data['class_label'], test_size=0.2, random_state=1)
        datasets = [X_train, X_test, y_train, y_test]
        # 持久化数据集
        with open(data_path, 'wb') as in_data:
            print("正在保存预处理数据。。。")
            pickle.dump(datasets, in_data, pickle.HIGHEST_PROTOCOL)
            pickle.dump(data, in_data, pickle.HIGHEST_PROTOCOL)
            pickle.dump(words_list, in_data, pickle.HIGHEST_PROTOCOL)
        return datasets


def load_w2v():
    with open('data/datasets.pkl', 'rb') as out_data:
        datasets = pickle.load(out_data)
        del datasets
        data = pickle.load(out_data)
        word_list = pickle.load(out_data)

    word2vec_path = "data/word2vec_model_5.txt"
    if os.path.exists(word2vec_path):
        print("正在加载训练的word2vec。。。")
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
    else:
        # 训练数据
        print(word_list[0])

        # min_count:忽略频率低于该值的词
        # size 训练词向量的维度
        model = Word2Vec(min_count=1, size=5)
        # 构建此货表
        model.build_vocab(word_list)
        # 训练
        model.train(word_list, total_examples=model.corpus_count, epochs=model.iter)

        # 保存模型
        print("正在训练词向量")
        model.wv.save_word2vec_format(word2vec_path)
        # 获取训练结果中指定词的结果向量
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
    print(word2vec.wv["文本"])

    def get_average_word2vec(tokens_list, vector, k=5):
        # 判断数据是否为空，如果为null，则返回1xk的向量
        if len(tokens_list) < 1:
            return np.zeros(k)
        # 如果没在vector则设为零向量
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list.split(',')]
        # print(vectorized)

        length = len(vectorized)
        # 按每一个词的列求和
        summed = np.sum(vectorized, axis=0)
        # 求平均值
        averaged = np.divide(summed, length)
        return averaged

    def get_word2vec_embeddings(vector, data, k):
        embeddings = data['tokens'].apply(lambda x: get_average_word2vec(x, vector, k))
        return list(embeddings)

    embeddings_demo = get_word2vec_embeddings(word2vec, data, k=5)
    X_train_word2vec_demo, X_test_word2vec_demo, y_train_word2vec_demo, y_test_word2vec_demo =\
        train_test_split(embeddings_demo, data["class_label"], test_size=0.2, random_state=1)

    return X_train_word2vec_demo, X_test_word2vec_demo, y_train_word2vec_demo, y_test_word2vec_demo
# if __name__ == '__main__':
#     load_w2v()
