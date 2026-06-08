import os
import ast
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 1. 关键：替换国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 实验配置
MODEL_NAME = "joeddav/xlm-roberta-large-xnli" # 你可以换成 "hfl/chinese-roberta-wwm-ext" 体验更快的速度
NUM_LABELS = 14
MAX_LEN = 60
BATCH_SIZE = 32  # 4090 24G 显存完全吃得下 32

candidate_labels = ["财经", "彩票", "房产", "股票", "家居", "教育", 
                    "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]

# 3. 恢复并适配你原本的 Dataset 类（用于标准的监督训练）
class THUCNewsTrainDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60):
        self.max_len = max_len
        
        # 解析你自己的字典
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.char2id = ast.literal_eval(f.read())
        except UnicodeDecodeError:
            with open(dict_path, 'r', encoding='gbk') as f:
                self.char2id = ast.literal_eval(f.read())
                
        self.unk_id = self.char2id.get('<unk>', 1)
        self.pad_id = self.char2id.get('<pad>', max(self.char2id.values()) + 1 if self.char2id else 0)

        self.data = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        label_id = int(parts[0])
                        text = parts[1]
                        self.data.append((label_id, text))
                    except ValueError:
                        continue 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label_id, text = self.data[index]
        
        # 将文本转换为 Token ID
        input_ids = [self.char2id.get(char, self.unk_id) for char in text]
        
        # 截断与 Padding
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len]
            attention_mask = [1] * self.max_len
        else:
            padding_len = self.max_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_len
            input_ids = input_ids + [self.pad_id] * padding_len
            
        # Trainer 要求的返回格式：字典，包含模型 forward 需要的参数名
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

# 4. 定义评估指标计算函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "macro_f1": f1}

def main():
    print("正在加载数据...")
    # 请替换成你真实的训练集和测试集路径
    train_dataset = THUCNewsTrainDataset(tsv_path="/root/autodl-tmp/THUCNews/train.tsv", dict_path="/root/autodl-tmp/THUCNews/dict.txt", max_len=MAX_LEN)
    test_dataset = THUCNewsTrainDataset(tsv_path="/root/autodl-tmp/THUCNews/test.tsv", dict_path="/root/autodl-tmp/THUCNews/dict.txt", max_len=MAX_LEN)
    print(f"训练集样本: {len(train_dataset)}, 测试集样本: {len(test_dataset)}")

    print("正在初始化模型...")
    # 强制修改模型的分类头数量为 14
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    # ignore_mismatched_sizes=True 会自动丢弃原模型的 NLI 分类权重，重新随机初始化一个 14 类的线性层
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        config=config, 
        ignore_mismatched_sizes=True
    )

    # 5. 专属 4090 的高性能训练参数配置
    training_args = TrainingArguments(
        output_dir="./results",          # 权重保存路径
        eval_strategy="epoch",           # 每个 epoch 结束后进行一次评估
        save_strategy="epoch",           # 每个 epoch 保存一次检查点
        learning_rate=2e-5,              # 经典的微调学习率
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,              # 11万数据，跑3个epoch足够收敛
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,     # 训练结束时加载表现最好的权重
        metric_for_best_model="accuracy",
        
        # 🚀 针对 4090 显卡的核心提速参数：
        fp16=True,                       # 开启半精度加速，速度直接翻倍
        dataloader_num_workers=4,        # 开启多线程数据加载，防止 CPU 瓶颈
        gradient_checkpointing=True,     # 开启梯度检查点，极端省显存（如需追求极限速度，可设为 False）
        report_to="none"
    )

    # 6. 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. 开始训练
    print("\n🚀 开始微调训练...")
    trainer.train()

    # 8. 最终评估与打印详细报告
    print("\n训练完成，正在对测试集做最终专业评估...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    print("\n================== 最终微调测试结果 ==================")
    print(classification_report(labels, preds, target_names=candidate_labels, digits=4))

if __name__ == "__main__":
    main()