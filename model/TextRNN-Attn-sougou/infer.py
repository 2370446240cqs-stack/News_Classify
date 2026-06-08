import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. 核心类定义 (复用你的新训练代码，并微调了表头处理)
# ==========================================
class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60, has_header=False):
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
                next(f)  # 只有指定有表头时才跳过第一行
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
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx, pretrained_weights=None):
        super(TextRNN_Att, self).__init__()
        
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=pad_idx
            )
        else:
            # 推理阶段直接初始化普通 Embedding，稍后由 load_state_dict 覆盖权重，省去重复加载大文件的耗时
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
            
        self.lstm = nn.LSTM(
            input_size=embed_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out, att_weights = self.attention(out)  # 返回特征以及权重
        out = self.fc(out)
        return out, att_weights

# ==========================================
# 2. 推理与准确率评估
# ==========================================
def test_model(model, dataloader, device):
    """运行测试集并输出准确率"""
    model.eval()  # 切换到评估模式
    
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    with torch.no_grad():  # 推理阶段不计算梯度
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            
            # ✨ 注意：新模型解包需要接收两个变量
            outputs, att_weights = model(texts)
            
            # 取最大值的索引作为预测标签
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    # 计算并打印整体准确率
    acc = accuracy_score(labels_all, predict_all)
    print(f"\n✅ 测试完成！测试集整体准确率 (Accuracy): {acc * 100:.2f}%\n")
    
    # 打印详细分类报告 (包含各别标签的精确度、召回率、F1值)
    print("各类别详细评估报告:")
    print(classification_report(labels_all, predict_all, digits=4))

# ==========================================
# 3. 启动入口
# ==========================================
if __name__ == "__main__":
    # 检测计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用计算设备: {device}")
    
    # 配置文件路径
    base_dir = "/root/autodl-tmp/THUCNews"
    test_path = f"{base_dir}/test.tsv"         # 你的测试集文件
    dict_path = f"{base_dir}/dict.txt"         # 字典路径
    model_path = "/root/autodl-tmp/model/lstm-attn-sougou.pth"     # 训练保存的新权重文件
    
    # 核心超参数 (必须与你的新模型训练配置完全一致)
    EMBED_SIZE = 300
    VOCAB_SIZE = 5308
    PAD_IDX = 5307
    NUM_CLASSES = 14
    MAX_LEN = 60
    BATCH_SIZE = 128
    
    print("正在加载测试集 (无表头模式)...")
    # 设置 has_header=False 确保不漏掉 test.tsv 的第一行数据
    test_dataset = THUCNewsDataset(tsv_path=test_path, dict_path=dict_path, max_len=MAX_LEN, has_header=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 初始化模型结构 (此处 pretrained_weights 设为 None)
    model = TextRNN_Att(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=128, 
                        num_classes=NUM_CLASSES, pad_idx=PAD_IDX, pretrained_weights=None)
    
    # 加载已训练好的模型权重
    print(f"正在快捷加载模型权重: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    print("\n=== 开始推理 ===")
    test_model(model, test_loader, device)