import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. 核心类定义 (复用你的训练代码，保持原样)
# ==========================================
class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60):
        self.max_len = max_len
        
        # 解析字典
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
        except UnicodeDecodeError:
            with open(dict_path, 'r', encoding='gbk') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
                
        # 处理特殊 Token
        self.unk_token = '<unk>'
        self.unk_id = self.char2id.get(self.unk_token, 1)
        
        self.pad_token = '<pad>'
        if self.pad_token not in self.char2id and '<PAD>' not in self.char2id:
            self.pad_id = max(self.char2id.values()) + 1
            self.char2id[self.pad_token] = self.pad_id
        else:
            self.pad_id = self.char2id.get(self.pad_token, self.char2id.get('<PAD>'))

        # 读取 TSV 数据
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
        
        input_ids = []
        for char in text:
            input_ids.append(self.char2id.get(char, self.unk_id))
            
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids = input_ids + [self.pad_id] * (self.max_len - len(input_ids))
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out        

# ==========================================
# 2. 推理与评估逻辑
# ==========================================
def test_model(model, dataloader, device):
    """运行测试集并输出准确率"""
    model.eval()  # 必须切换到评估模式，关闭 Dropout 机制
    
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    with torch.no_grad():  # 推理阶段不需要计算梯度，节约显存和算力
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            
            outputs = model(texts)
            # 取 Logits 中最大值的索引作为预测类别
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    # 计算并打印整体准确率
    acc = accuracy_score(labels_all, predict_all)
    print(f"\n✅ 测试完成！整体准确率 (Accuracy): {acc * 100:.2f}%\n")
    
    # 打印详细分类报告 (包含 Precision, Recall, F1-score)
    print("各类别详细评估报告:")
    print(classification_report(labels_all, predict_all, digits=4))

# ==========================================
# 3. 启动入口
# ==========================================
if __name__ == "__main__":
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用计算设备: {device}")
    
    # 配置路径
    base_dir = "/root/autodl-tmp/THUCNews"
    test_path = f"{base_dir}/test.tsv"      # 测试集路径
    dict_path = f"{base_dir}/dict.txt"      # 字典路径
    model_path = "/root/autodl-tmp/model/lstm-sougou.pth"    # 训练保存的模型权重文件
    
    # 核心超参数 (必须与训练时完全一致)
    EMBED_SIZE = 300
    VOCAB_SIZE = 5308
    PAD_IDX = 5307
    NUM_CLASSES = 14
    MAX_LEN = 60
    BATCH_SIZE = 128
    
    print("正在加载测试集...")
    test_dataset = THUCNewsDataset(tsv_path=test_path, dict_path=dict_path, max_len=MAX_LEN)
    # 测试集不需要 shuffle
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 初始化模型结构
    model = TextRNN(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=128, 
                    num_classes=NUM_CLASSES, pad_idx=PAD_IDX)
    
    # 加载权重 (添加 map_location 参数，确保在纯 CPU 环境下加载 GPU 训练的模型也不会报错)
    print(f"正在加载模型权重: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    print("\n=== 开始推理 ===")
    test_model(model, test_loader, device)