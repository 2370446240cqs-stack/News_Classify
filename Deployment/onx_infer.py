import os
import ast  # 【修复2】新增：导入 ast 模块用于解析字典
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader  # 【修复1】新增：把 Dataset 一并导入进来
from sklearn.metrics import classification_report, accuracy_score
from transformers import pipeline

# 【修复3】补充：定义缺失的变量（请将路径替换为你实际的文件路径）
test_path = "/root/autodl-tmp/THUCNews/test.tsv"      # 替换为真实的测试集 TSV 路径
dict_path = "/root/autodl-tmp/THUCNews/dict.txt"     # 替换为真实的字典路径
MAX_LEN = 60
BATCH_SIZE = 32

class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60):
        self.max_len = max_len
        
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
        except UnicodeDecodeError:
            with open(dict_path, 'r', encoding='gbk') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
                
        self.unk_token = '<unk>'
        self.unk_id = self.char2id.get(self.unk_token, 1)
        
        self.pad_token = '<pad>'
        if self.pad_token not in self.char2id and '<PAD>' not in self.char2id:
            self.pad_id = max(self.char2id.values()) + 1
            self.char2id[self.pad_token] = self.pad_id
        else:
            self.pad_id = self.char2id.get(self.pad_token, self.char2id.get('<PAD>'))

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
        return text, torch.tensor(label_id, dtype=torch.long)
        
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

candidate_labels = ["财经", "彩票", "房产", "股票", "家居", "教育", 
                    "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]
label2id = {label: idx for idx, label in enumerate(candidate_labels)}
'''
print("正在从国内镜像站下载或加载模型，请稍候...")
classifier = pipeline("zero-shot-classification", 
                      model="joeddav/xlm-roberta-large-xnli", 
                      device=0 if torch.cuda.is_available() else -1) 
'''
print("正在从国内镜像站下载或加载模型，请稍候...")
classifier = pipeline("zero-shot-classification", 
                      model="joeddav/xlm-roberta-large-xnli", 
                      device=0 if torch.cuda.is_available() else -1,
                      torch_dtype=torch.float16)  # ✅ 新增：开启FP16半精度加速
print("正在加载测试集...")
test_dataset = THUCNewsDataset(tsv_path=test_path, dict_path=dict_path, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
print(f"测试集样本数: {len(test_dataset)}")

all_preds = []
all_trues = []

print("\n开始对测试集进行批量推理...")

for batch in tqdm(test_loader, desc="Inference Progress"):
    texts, true_labels = batch
    
    if isinstance(texts, tuple):
        texts = list(texts)

    #results = classifier(texts, candidate_labels)
    results = classifier(texts, candidate_labels, batch_size=len(texts))
    for i, res in enumerate(results):
        pred_label_str = res['labels'][0]
        pred_label_id = label2id[pred_label_str]
        all_preds.append(pred_label_id)

        true_label = true_labels[i]
        if torch.is_tensor(true_label):
            true_label_id = true_label.item()
        elif isinstance(true_label, str):
            true_label_id = label2id[true_label]
        else:
            true_label_id = int(true_label)
            
        all_trues.append(true_label_id)

acc = accuracy_score(all_trues, all_preds)
print("\n================== 测试结果 ==================")
print(f"总样本数: {len(all_trues)}")
print(f"测试集准确率 (Accuracy): {acc * 100:.2f}%")
print("\n详细分类报告 (Classification Report):")
print(classification_report(all_trues, all_preds, target_names=candidate_labels))