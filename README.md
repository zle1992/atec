# atec蚂蚁金服NLP智能客服比赛  16th/2632
https://dc.cloud.alipay.com/index#/topic/intro?id=3


## 目录    
```
project
│   README.md
│      
│
└───data/ 词向量及一些中间文件
│   │ 
│   └───data/ 训练数据
│   │
│   └───log_dir/ log文件
│   │
│   │
│   └───share/  线下训练与线上提交的公共文件
│       └─── jieba/ 分词、停用词
|       |
│       └─── mv_w2v/ 训练词向量
|       |
│       └─── single/ 单模型
|       |
|       └─── stack/ cv模型及中间数据
│   
└───model/
│   │  各种模型
│   │   
└───feature/
│   │  提取的人工特征
│   │   
└───submit/
│   │  线下测试submit
│   │   
└───util/
│   │  辅助文件，分词，训练词向量，拼音转换等
│   │   
    
    
```




## 运行流程：    
参数配置在util/config.py中


python  util/w2v.py  
训练词向量。（自动调用分词，data2id）

python train.py cv  cnn1  
对cnn1模型采用交叉验证。




## 一些实验记录
0612 fix cv bugs offline 数据扩充5 cv  CNN :  0.63407952549263558
0.6155
0613 fix cv bugs offline 数据扩充5 cv ESIM :  0.65816989095636136


0626 CNN CV NO 数据扩充  26个特征：
n :', 0.52228330874667162, 0.52633540429317449, 0.52415987274990683


0629 add checkpoint and best model earlystop and change lr
'mean :', 0.45830773149153237, 0.67153672904068951, 0.54450204294307059)
ON LINE :0.6206
