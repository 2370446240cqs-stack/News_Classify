import torch
import torch.nn as nn

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