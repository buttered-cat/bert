# 获取预训练的BERT词向量

本项目Fork自[google-research/bert](https://github.com/google-research/bert)，对获取预训练BERT词向量序列的过程进行了封装，把使用TPU Estimator API重构为使用Session API，优化其性能与易用性。

## 与原Repo中`extract_features.py`的区别

* 使用Session API，可复用同一个Session实例。原Repo的代码由于TPU Estimator API的特性，在每次调用`estimator.predict`方法时都会重新生成Session，降低了重复调用的效率。此项目将其重构为使用Session API，把整个过程封装为类，实现可复用的Session。实测重复调用效率提升为 **20%** 左右。
* 不限制输入句子的最大长度。原脚本对长度超过`max_seq_length-2`（单句）或`max_seq_length-3`（句子对）的句子进行截尾（由于[CLS]和[SEP]的加入）。此项目`bert_embedding.py`的`Bert`类中，`get_embedded_vectors`方法不对最大长度做限制。
* 禁用position embedding。由于上条所述，`max_seq_length`长度可能大于公开的预训练模型文件使用的最大长度，导致报错，所以目前禁用。

## 使用

```
from bert_embedding import Bert
model = Bert(
    bert_config_file,
    vocab_file,
    init_checkpoint,
    requested_layers,
    sess,
    do_lower_case,
    batch_size,
    use_one_hot_embeddings
)
result = get_embedded_vectors(['小明', '小明的爸爸', '小明的爸爸的爸爸'])

# result结构: [
#   {
#       'features': [
#           {
#               'token': 'A',
#               'layers': [
#                   {
#                       'index': -1,
#                       'values': [...]
#                   }
#               ]
#           }
#       ],
#       'unique_id': int
#   },
#   ...
# ]
# 返回的结果包括`[CLS]`和`[SEP]`，结构与原repo代码相同。
```

### 参数说明

* bert_config_file：模型config文件路径
* vocab_file：模型词汇表文件路径
* init_checkpoint：模型checkpoint文件路径（如bert_model.ckpt，参考原repo文档）
* requested_layers：获取Transformer的隐层层数，如'-1,-2,-3,-4'代表获取最后四层。参考原repo文档
* sess：Session实例，如不传则在类内自动创建
* do_lower_case：统一字母为小写，参考原repo文档
* batch_size：模型运行的batch大小
* use_one_hot_embeddings：由于不用TPU所以暂时没什么用，默认值为False，参考原repo文档

## 关于WordPiece Tokenization

由于BERT会根据词表里的后缀等entry做word piece tokenization，所以返回的结果长度可能小于传入字符串长度。若想以字符为单位进行tokenization，可以使用`tokenization.py`中的`CharTokenizer`替换`Bert`类使用的`FullTokenizer`。
