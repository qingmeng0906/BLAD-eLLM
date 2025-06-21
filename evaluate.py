import json

# 读取JSON文件
with open('', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化混淆矩阵
TP = 0  # 真实为1，预测为1
TN = 0  # 真实为0，预测为0
FP = 0  # 真实为0，预测为1
FN = 0  # 真实为1，预测为0

for item in data:
    true_label = int(item['output'])
    pred_label = int(item['predict'])
    
    if true_label == 1:
        if pred_label == 1:
            TP += 1
        else:
            FN += 1
    else:
        if pred_label == 1:
            FP += 1
        else:
            TN += 1

# 计算指标
total = TP + TN + FP + FN
accuracy = (TP + TN) / total if total != 0 else 0.0
PRR = FP / (FP + TN)
# 处理除零情况
try:
    recall = TP / (TP + FN)
except ZeroDivisionError:
    recall = 0.0

try:
    precision = TP / (TP + FP)
except ZeroDivisionError:
    precision = 0.0

try:
    f1 = 2 * (precision * recall) / (precision + recall)
except ZeroDivisionError:
    f1 = 0.0

# 打印结果（保留4位小数）
print(TP)
print(TN)
print(FP)
print(FN)
print(f"准确率（Accuracy）: {accuracy:.4f}")
print(f"召回率（Recall）: {recall:.4f}")    
print(f"精确率（Precision）: {precision:.4f}") 
print(f"F1分数: {f1:.4f}")
print(f"PDR: {PRR:.4f}")