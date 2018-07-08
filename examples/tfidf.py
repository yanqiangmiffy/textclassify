# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 20:05
# @Author  : quincyqiang
# @File    : tfidf.py
# @Software: PyCharm

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
# 读取数据
data=pd.read_csv('demo2.csv',encoding='utf-8',header=0)
data.columns=['word']

data1=data['word']
print(data1)
tfidf_vectorizer=TfidfVectorizer()
train=tfidf_vectorizer.fit(data1)
word_vec=train.transform(data1)
print(word_vec)
print(pd.DataFrame(word_vec.toarray(),columns=tfidf_vectorizer.get_feature_names()))
