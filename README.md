# pytorch_casrel_triple_extraction
基于pytorch的CasRel进行三元组抽取。
这里是使用casrel，先抽取主体，再识别客体和关系。具体使用说明：
- 1、在data/ske/raw_data下是原始数据，新建一个process.py，主要是得到mid_data下的关系的类型。
- 2、针对于不同的数据源，在data_loader.py中修改MyDataset类下，返回的是一个列表，列表中的每个元素是：(text, labels)，其中labels是[[主体，类别，客体]]。
- 3、运行main.py进行训练、验证、测试和预测。

# 依赖

```
pytorch==1.6.0
transformers==4.5.0
CUDA == 10.2 / 10.1
python==3.7 / 3.8
CUDNN == 7.6.5
GPU显卡12G 
```

# 运行

```python
python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/ske/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=49 \  # 关系类别
--seed=123 \
--gpu_ids="0" \
--max_seq_len=256 \  # 句子最大长度
--lr=5e-5 \
--other_lr=5e-5 \
--train_batch_size=32 \
--train_epochs=1 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.1 \
--use_tensorboard="False" \  # 是否使用tensorboardX可视化
--use_dev_num=1000 \ # 使用多少验证数据进行验证
```


