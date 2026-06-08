import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ==========================================
# 1. 数据集定义 (保持原样)
# ==========================================
class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60):
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

# ==========================================
# 2. ✨ 新增：加载搜狗预训练词向量的核心函数
# ==========================================
def load_pretrained_embedding(char2id, emb_file, embed_size=300):
    """
    根据给定的字典 char2id 读取预训练字向量文件，构建并返回 PyTorch 嵌入矩阵
    """
    vocab_size = len(char2id)
    # 使用均值为0，方差为0.1的正态分布随机初始化矩阵（比全零或纯 random 效果更好）
    embeddings = np.random.normal(0, 0.1, (vocab_size, embed_size))
    
    # 将包含在字典中的 <pad> 特殊字符强制初始化为全0向量
    pad_id = char2id.get('<pad>', char2id.get('<PAD>'))
    if pad_id is not None:
        embeddings[pad_id] = np.zeros(embed_size)

    hit_count = 0
    print(f"正在读取预训练词向量文件: {emb_file} ...")
    with open(emb_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:  # 大多数词向量文件的第一行是汇总数据（行数 维度），需要跳过
                continue
            
            # 每一行的标准格式：字 v1 v2 v3 ... v300
            tokens = line.strip().split(' ')
            if len(tokens) == embed_size + 1:
                char = tokens[0]
                # 如果这个预训练的汉字正好在你的 THUCNews 字典里，则用它替换随机值
                if char in char2id:
                    idx = char2id[char]
                    embeddings[idx] = np.array([float(x) for x in tokens[1:]])
                    hit_count += 1
                    
    print(f"[*] 预训练字向量加载完毕！字典词总数: {vocab_size}, 成功匹配到预训练字的个数: {hit_count}")
    return torch.tensor(embeddings, dtype=torch.float32)

# ==========================================
# 3. ✨ 修改：适配预训练权重的 TextRNN 模型类
# ==========================================
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx, pretrained_weights=None):
        super(TextRNN, self).__init__()
        
        # 判断是否传入了搜狗预训练权重
        if pretrained_weights is not None:
            # 使用 from_pretrained 快捷加载权重
            # freeze=False 表示允许词向量在后续反向传播中根据你的新闻分类任务被微调优化
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, 
                freeze=False, 
                padding_idx=pad_idx
            )
        else:
            # 如果没有传入则保持以前的随机初始化逻辑
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, 
                embedding_dim=embed_size, 
                padding_idx=pad_idx
            )
        
        # 双向 LSTM 层
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=2,           
            batch_first=True, 
            bidirectional=True      
        )
        
        # 全连接分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :]) # 取序列的最后一个时间步
        return out        

# ==========================================
# 4. 评估与训练主循环逻辑 (保持原样)
# ==========================================
def evaluate(model, dataloader, device):
    """评估模型，计算 Loss、Accuracy 和 Macro-F1"""
    model.eval() 
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad(): 
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    acc = accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / len(dataloader), acc, f1

def train(model, train_loader, dev_loader, device, num_epochs=10):
    """模型训练主循环"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() 
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        model.train() 
        
        for i, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'  Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
                
        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, device)
        print(f'  [Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './lstm-sougou.pth')
            print('  [*] Best model saved!')
            
    print("训练完成！")

# ==========================================
# 5. ✨ 修改：启动入口（加载预训练权重并注入）
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用计算设备: {device}")
    
    base_dir = "/root/autodl-tmp/THUCNews"
    train_path = f"{base_dir}/train.tsv"  
    val_path = f"{base_dir}/val.tsv"      
    dict_path = f"{base_dir}/dict.txt"
    
    # 搜狗词向量的路径
    emb_file_path = "/root/autodl-tmp/sgns.sogou.char"
    
    print("正在加载数据集以提取字典...")
    train_dataset = THUCNewsDataset(tsv_path=train_path, dict_path=dict_path, max_len=60)
    val_dataset = THUCNewsDataset(tsv_path=val_path, dict_path=dict_path, max_len=60)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # 💡 关键修改点一：设置搜狗词向量的标准维度 300 维
    EMBED_SIZE = 300  
    VOCAB_SIZE = len(train_dataset.char2id) # 动态获取真实的字典字数
    PAD_IDX = train_dataset.pad_id          # 动态获取真实的 PAD 索引
    NUM_CLASSES = 14  
    
    # 💡 关键修改点二：在初始化模型前，运行函数生成预训练权重矩阵
    print("\n=== 开始加载并匹配搜狗预训练词向量 ===")
    pretrained_matrix = load_pretrained_embedding(
        char2id=train_dataset.char2id, 
        emb_file=emb_file_path, 
        embed_size=EMBED_SIZE
    )
    
    # 💡 关键修改点三：将生成的矩阵通过 pretrained_weights 参数喂给模型
    model = TextRNN(
        vocab_size=VOCAB_SIZE, 
        embed_size=EMBED_SIZE, 
        hidden_size=128, 
        num_classes=NUM_CLASSES, 
        pad_idx=PAD_IDX,
        pretrained_weights=pretrained_matrix
    )
    
    print("\n=== 开始训练 ===")
    train(model, train_loader, val_loader, device, num_epochs=10)