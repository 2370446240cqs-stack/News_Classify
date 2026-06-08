import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. 核心类定义 (完全复用你的训练代码)
# ==========================================
class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60, has_header=False):
        self.max_len = max_len
        
        # 采用 ast 模块安全加载 Python 字典格式的文件
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
        except UnicodeDecodeError:
            with open(dict_path, 'r', encoding='gbk') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
                
        # 动态处理特殊 Token
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
            if has_header:
                next(f) # 如果指定有表头，才跳过第一行
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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w, -0.1, 0.1)

    def forward(self, H):
        M = torch.tanh(H) 
        alpha = torch.matmul(M, self.w) 
        alpha = F.softmax(alpha, dim=1) 
        out = torch.sum(H * alpha, dim=1)
        return out, alpha.squeeze(-1) 

class TextRNN_Att(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx):
        super(TextRNN_Att, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out, att_weights = self.attention(out) # 接收特征和权重
        out = self.fc(out)
        return out, att_weights

# ==========================================
# 2. 推理与核心评估逻辑
# ==========================================
def test_model(model, dataloader, device):
    """运行测试集推理，比对标签并计算正确率"""
    model.eval() # 切换到评估模式（关闭可能存在的 Dropout 或 BatchNorm）
    
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    with torch.no_grad(): # 测试阶段关闭梯度计算，节省显存与耗时
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            
            # ✨ 注意：新模型会同时返回预测 Logits 和 Attention 权重，需要解包接收
            outputs, att_weights = model(texts)
            
            # 取 Logits 中概率最大的索引作为预测的类别标签
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    # 计算整体正确率
    acc = accuracy_score(labels_all, predict_all)
    print(f"\n✅ 测试完成！测试集整体正确率 (Accuracy): {acc * 100:.2f}%\n")
    
    # 打印 0~13 每个具体类别的详细评估报告 (精确率、召回率、F1值)
    print("各类别详细评估报告:")
    print(classification_report(labels_all, predict_all, digits=4))

# ==========================================
# 3. 启动入口
# ==========================================
if __name__ == "__main__":
    # 检测计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用计算设备: {device}")
    
    # 路径配置文件 (请根据你在 AutoDL 上的实际位置微调)
    base_dir = "/root/autodl-tmp/THUCNews"
    test_path = f"{base_dir}/test.tsv"       # 测试集表格文件路径
    dict_path = f"{base_dir}/dict.txt"       # 字典路径
    model_path = "/root/autodl-tmp/model/lstm-attn.pth"     # 训练保存的最优模型权重文件
    
    # 核心超参数 (必须与你训练时的配置完全一致)
    VOCAB_SIZE = 5308
    PAD_IDX = 5307
    NUM_CLASSES = 14
    EMBED_SIZE = 256
    HIDDEN_SIZE = 128
    MAX_LEN = 60
    BATCH_SIZE = 128
    
    print("正在加载测试集 (已开启无表头读取模式)...")
    # has_header=False 确保不漏掉 test.tsv 的第一行
    test_dataset = THUCNewsDataset(tsv_path=test_path, dict_path=dict_path, max_len=MAX_LEN, has_header=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 测试集保持原序，shuffle设为False
    print(f"测试集有效样本数: {len(test_dataset)}")
    
    # 初始化模型结构
    model = TextRNN_Att(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, 
                        num_classes=NUM_CLASSES, pad_idx=PAD_IDX)
    
    # 加载已训练好的权重
    print(f"正在加载模型权重文件: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    print("\n=== 开始进行测试集推理 ===")
    test_model(model, test_loader, device)