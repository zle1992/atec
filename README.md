# atec蚂蚁金服NLP智能客服比赛  16th/2632
https://dc.cloud.alipay.com/index#/topic/intro?id=3

+-- _config.yml金服NL
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html


data/
词向量及一些中间文件。

data/data/ 
原始训练数据

data/log_dir/
log文件

data/share
线下训练与线上提交的公共文件

data/share/jieba
分词、停用词

data/share/mv_w2v
训练词向量

data/share/pre_w2v
外部词向量

data/


submit/
测试数据 及 提交文件

model/
模型函数


util/
config.py  一些文件路径及参数
cutword.py  :分词
data2id.py  把词映射成id
help.py 一些辅助函数，如划分训练验证。
w2v.py  训练词向量



运行流程：

python  util/w2v.py  
训练词向量。（自动调用分词，data2id）

python train.py cv  cnn1  
对cnn1模型采用交叉验证。

python submit.py  提交结果  
模型选择及cv 记得修改main()函数




## 0612 fix cv bugs offline 数据扩充5 cv  CNN :  0.63407952549263558
## 0.6155
## 0613 fix cv bugs offline 数据扩充5 cv ESIM :  0.65816989095636136


##0626 CNN CV NO 数据扩充  26个特征：
n :', 0.52228330874667162, 0.52633540429317449, 0.52415987274990683


##0629 add checkpoint and best model earlystop and change lr
'mean :', 0.45830773149153237, 0.67153672904068951, 0.54450204294307059)
 ON LINE :0.6206
