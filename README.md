# Sequence-Labeling

### 分词 
### 实体识别
### 语义标注(slots)
### 标点预测  

---

+ V1
    - model: GRU + softmax
    - loss : CRF
+ V2
    - model: enoder: gru, decoder: gru, attention
    - loss : cross_entropy
