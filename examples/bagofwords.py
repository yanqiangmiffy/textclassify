# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 19:50
# @Author  : quincyqiang
# @File    : bagofwords.py
# @Software: PyCharm

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# 读取数据
data=pd.read_csv('demo1.csv',encoding='utf-8',header=0)
data.columns=['word']

data1=data['word']
# 词袋模型
count_vectorizer=CountVectorizer(token_pattern=r"(?u)\b[^，]+\b")
emb=count_vectorizer.fit(data1)
data_vec=emb.transform(data1)
print(data_vec)

# 打印结果
print(pd.DataFrame(data_vec.toarray(),columns=count_vectorizer.get_feature_names()))
