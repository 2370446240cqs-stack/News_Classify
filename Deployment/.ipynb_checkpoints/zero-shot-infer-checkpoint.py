import os
# 关键魔法：将 HuggingFace 的默认下载源替换为国内镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import csv
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score

# ==================== 配置参数 ====================
TEST_PATH = "/root/autodl-tmp/THUCNews/test.tsv"          # 测试集路径，格式：标签\t文本
BATCH_SIZE = 32                      # 推理批大小，可根据显存调整
DEVICE = 0                           # 使用 GPU 0，若无 GPU 可设为 -1

candidate_labels = ["财经", "彩票", "房产", "股票", "家居", "教育", 
                    "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
# ================================================

print("正在从国内镜像站下载或加载模型，请稍候...")
# 加载零样本分类 pipeline，支持批量推理
classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=DEVICE,
    batch_size=BATCH_SIZE
)

# 1. 读取测试集文本和真实标签
texts, true_labels = [], []
print(f"正在加载测试集: {TEST_PATH}")
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) < 2:
            continue
        label, text = row[0].strip(), row[1].strip()
        true_labels.append(label)
        texts.append(text)

print(f"测试集样本数: {len(texts)}")

# 2. 批量推理
print("正在进行零样本分类推理...")
results = classifier(texts, candidate_labels)

# 3. 提取预测标签（取概率最高的类别）
pred_labels = [res['labels'][0] for res in results]

# 4. 计算并输出评估指标
acc = accuracy_score(true_labels, pred_labels)
print("\n==================== 测试结果 ====================")
print(f"准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
print("\n分类报告 (Classification Report):")
print(classification_report(true_labels, pred_labels, target_names=candidate_labels, digits=4))