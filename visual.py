# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 20:50
# @Author  : quincyqiang
# @File    : visual.py
# @Software: PyCharm

import pickle
import pandas as pd

#通过词云分别打印不同类别的出现频数前10的词汇
import numpy as  np
import matplotlib.pyplot as plt
#词云生成工具
from wordcloud import WordCloud
#需要对中文进行处理
import matplotlib.font_manager as fm
def word_cloud_data():

    with open('data/datasets.pkl','rb') as out_data:
        datasets=pickle.load(out_data)
        del datasets
        data=pickle.load(out_data)

    # 得到所有的词语
    all_words = []
    for lin, label in zip(data["tokens"], data["class_label"]):
        for word in list(lin.split(",")):
            all_words.append([label, word.strip()])

    # 得到所有的词语
    all_words1 = []
    for i in all_words:
        all_words1.append(i[1])
    VOCAB = len(set(all_words1))
    print("单词总数%s, 词汇量 %s" % (len(all_words), VOCAB))

    da = pd.DataFrame(all_words, columns=["key", "value"])
    print(da.head(5))

    # 按类别汇总词词语
    label_word_dic = {}
    for key, value in zip(da["key"], da["value"]):
        if key in label_word_dic.keys():
            label_word_dic[key].append(value)
        else:
            label_word_dic[key] = [value]

    # 按类别统计词频
    group_word_count = {}
    for key in label_word_dic.keys():
        fdict_word_count = {}
        for words in label_word_dic[key]:
            if words in fdict_word_count.keys():
                fdict_word_count[words] += 1
            else:
                fdict_word_count[words] = 1
        li = sorted(fdict_word_count.items(), key=lambda d: d[1], reverse=True)
        group_word_count[key] = li
    print(group_word_count[1][0:10])

    return label_word_dic

def plot_word_cloud():
    label_word_dic=word_cloud_data()

    fig = plt.figure(figsize=(20, 30))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(7):
        wc = WordCloud(
            background_color="white",  # 设置背景为白色，默认为黑色
            random_state=100,
            max_words=10,
            font_path='C:/Windows/Fonts/simkai.ttf')  # 中文处理，用系统自带的字体
        wc.generate(str(label_word_dic[i]))
        # 为图片设置字体
        my_font = fm.FontProperties(fname='C:/Windows/Fonts/simkai.ttf')
        plt.subplot(7, 2, i + 1)
        plt.imshow(wc)
        plt.title(i, fontsize=20)

    fig.show()
    plt.show()

if __name__ == '__main__':
    plot_word_cloud()