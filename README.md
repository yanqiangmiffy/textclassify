# textclassify
利用bow（词袋特征）、tfidf、word2vec进行中文文本分类
下图为部分数据集
![](https://upload-images.jianshu.io/upload_images/1531909-b79b2dd9dcb8b1d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
第一列为分类标签，第二列为文本数据，是关于七类文学作品的简介
![](https://upload-images.jianshu.io/upload_images/1531909-0f2d2fecf4f26962.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## requirements
- `gensim`
- `sklearn`

## bow
accuracy=0.918533,precision=0.918528,recall=0.918533,f1=0.918515
## tfidf
accuracy = 0.931081, precision = 0.931091, recall = 0.931081, f1 = 0.931071
## word2vec
accuracy = 0.573359, precision = 0.565731, recall = 0.573359, f1 = 0.567236
