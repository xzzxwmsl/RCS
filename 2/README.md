|姓名|学号|
|-|-|
|向正中|2020223040024|

# 第二次作业 PMF
项目复现了论文《LARA Attribute-to-feature Adversarial Learning for New-item Recommendation》

代码参考开源代码,并修改了实现细节


## 参数
|alpha|batch_size|learning_rate|epoch|attr_num|attr_dim|hidden_dim|user_emb_dim|
|-|-|-|-|-|-|-|-|
|0.0001|1024|0.0001|300|18|5|100|18|

训练集 : 测试集=8:2

## 运行截图
### 运行截图
![](运行结果截图/运行过程.png)
### ndcg,map,precision for top 10,20
![](运行结果截图/ndcg_precision_map.png)
### loss
![](运行结果截图/loss.png)
