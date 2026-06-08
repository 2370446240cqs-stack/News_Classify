import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 假设 THUCNewsDataset 和 TextRNN 已经定义好
# ... [前面的 Dataset 和 Model 代码] ...
class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=60):
        self.max_len = max_len
        
        # 1. 采用 ast 模块安全加载 Python 字典格式的文件
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # ast.literal_eval 可以安全地将字符串 '{'汀': 0, ...}' 变回真实的字典对象
                self.char2id = ast.literal_eval(content)
        except UnicodeDecodeError:
            with open(dict_path, 'r', encoding='gbk') as f:
                content = f.read()
                self.char2id = ast.literal_eval(content)
                
        # 2. 动态处理特殊 Token
        # 你的字典里使用的是小写的 '<unk>'
        self.unk_token = '<unk>'
        self.unk_id = self.char2id.get(self.unk_token, 1) # 如果没有就默认用1
        
        # 检查是否已有 <PAD>。由于 0 已经被 '汀' 占用，我们需要给 PAD 分配一个全新的、不重复的 ID
        self.pad_token = '<pad>'
        if self.pad_token not in self.char2id and '<PAD>' not in self.char2id:
            # 取当前字典里最大的 ID 值加 1，作为 PAD 的 ID（比如你的 unk 是 5306，那 pad 就是 5307）
            self.pad_id = max(self.char2id.values()) + 1
            self.char2id[self.pad_token] = self.pad_id
        else:
            self.pad_id = self.char2id.get(self.pad_token, self.char2id.get('<PAD>'))

        # 3. 读取 TSV 数据（保持原样）
        self.data = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            #next(f) # 如果第一行是表头 label \t text_a，取消注释这行；如果没有表头，注释掉这行
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        label_id = int(parts[0])
                        text = parts[1]
                        self.data.append((label_id, text))
                    except ValueError:
                        continue # 跳过无法转换为整型的异常标签行

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label_id, text = self.data[index]
        
        # 4. 文本数值化（查表）
        input_ids = []
        for char in text:
            # 查不到的字统一用 self.unk_id
            input_ids.append(self.char2id.get(char, self.unk_id))
            
        # 5. 截断与填充 (改用动态获取的 pad_id)
        if len(input_ids) >= self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids = input_ids + [self.pad_id] * (self.max_len - len(input_ids))
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx):
        super(TextRNN, self).__init__()
        
        # 1. 词嵌入层 (Embedding Layer)
        # 作用：将离散的整数ID (如3379) 映射为连续的稠密向量
        # padding_idx=pad_idx 会让 PAD 的向量始终保持为全0，不参与训练
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_size, 
            padding_idx=pad_idx
        )
        
        # 2. 双向 LSTM 层
        # batch_first=True 表示输入数据的维度是 [batch_size, seq_len, embed_size]
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=2,           # 使用2层LSTM增加模型拟合能力
            batch_first=True, 
            bidirectional=True      # 开启双向，同时捕捉前向和后向语义
        )
        
        # 3. 全连接分类层 (Linear Layer)
        # 因为是双向LSTM，所以输出的隐藏层维度是 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]  (例如: [64, 30])
        
        out = self.embedding(x)
        # out shape: [batch_size, seq_len, embed_size]
        
        # LSTM 返回两个值：所有时间步的输出(out) 和 最后一个时间步的隐状态(_)
        out, _ = self.lstm(out)
        # out shape: [batch_size, seq_len, hidden_size * 2]
        
        # 我们只需要序列的最后一个时间步的特征来进行分类
        # out[:, -1, :] 表示取所有batch的最后一个词(seq_len维度上的最后一个元素)
        out = self.fc(out[:, -1, :])
        # out shape: [batch_size, num_classes] (例如: [64, 14])
        
        return out        
def evaluate(model, dataloader, device):
    """评估模型，计算 Loss、Accuracy 和 Macro-F1"""
    model.eval() # 切换到评估模式，关闭 Dropout (如果有)
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad(): # 验证阶段不计算梯度
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            
            # 取 Logits 中最大值的索引作为预测类别
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
    # 计算准确率和 Macro-F1
    acc = accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / len(dataloader), acc, f1

def train(model, train_loader, dev_loader, device, num_epochs=10):
    """模型训练主循环"""
    # 将模型加载到 GPU
    model = model.to(device)
    
    # 定义优化器和损失函数 (文本分类标准配置)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # CrossEntropyLoss 内部自带了 LogSoftmax 操作，所以模型最后一层不需要加 Softmax
    criterion = nn.CrossEntropyLoss() 
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        model.train() # 切换到训练模式
        
        for i, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            # 1. 梯度清零
            model.zero_grad()
            # 2. 前向传播
            outputs = model(texts)
            # 3. 计算损失
            loss = criterion(outputs, labels)
            # 4. 反向传播
            loss.backward()
            # 5. 更新参数
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'  Batch [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
                
        # 每个 Epoch 结束后在验证集上评估
        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, device)
        print(f'  [Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}')
        
        # 保存最优模型 (保存在当前数据盘目录下，避免占用系统盘)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './lstm.pth')
            print('  [*] Best model saved!')
            
    print("训练完成！")

# === 启动入口 ===
if __name__ == "__main__":
    # 1. 检测设备 (优先使用 GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用计算设备: {device}")
    
    # 2. 配置文件路径 (建议在 AutoDL 中使用绝对路径，避免运行目录不一致导致找不到文件)
    # 假设你的 train.tsv 和 dict.txt 也与 Val.tsv 在同一目录下
    base_dir = "/root/autodl-tmp/THUCNews"
    train_path = f"{base_dir}/train.tsv"  # 请确认实际训练集文件名
    val_path = f"{base_dir}/val.tsv"      # 你指定的验证集路径
    dict_path = f"{base_dir}/dict.txt"
    
    print("正在加载数据集...")
    # 3. 分别独立加载训练集和验证集
    train_dataset = THUCNewsDataset(tsv_path=train_path, dict_path=dict_path, max_len=60)
    val_dataset = THUCNewsDataset(tsv_path=val_path, dict_path=dict_path, max_len=60)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 4. 构建 DataLoader
    # 训练集需要打乱 (shuffle=True)，验证集不需要打乱 (shuffle=False) 以保证评估顺序一致
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # 根据你之前输出的字典信息配置超参数
    VOCAB_SIZE = 5308
    PAD_IDX = 5307
    NUM_CLASSES = 14  # 请确认你的实际类别数量
    
    # 5. 初始化模型
    model = TextRNN(vocab_size=VOCAB_SIZE, embed_size=256, hidden_size=128, 
                    num_classes=NUM_CLASSES, pad_idx=PAD_IDX)
    
    # 6. 开始训练
    print("\n=== 开始训练 ===")
    train(model, train_loader, val_loader, device, num_epochs=10)