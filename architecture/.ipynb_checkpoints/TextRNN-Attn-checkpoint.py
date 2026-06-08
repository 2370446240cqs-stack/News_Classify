import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 这里的 hidden_size * 2 是因为我们用的是双向 LSTM
        # 定义一个可学习的权重矩阵 w，用于计算注意力得分
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        # 初始化权重
        nn.init.uniform_(self.w, -0.1, 0.1)

    def forward(self, H):
        """
        H: 双向 LSTM 的完整输出序列
        Shape: [batch_size, seq_len, hidden_size * 2]
        """
        # 1. 非线性变换
        # M shape: [batch_size, seq_len, hidden_size * 2]
        M = torch.tanh(H) 
        
        # 2. 计算原始注意力得分 (Score)
        # alpha shape: [batch_size, seq_len, 1]
        alpha = torch.matmul(M, self.w) 
        
        # 3. 在 seq_len 维度上进行 Softmax 归一化，得到注意力权重
        # 这样保证了一句话中所有词的权重加起来等于 1
        alpha = F.softmax(alpha, dim=1) 
        
        # 4. 根据权重对 LSTM 的输出进行加权求和，得到最终的句子级上下文向量
        # out shape: [batch_size, hidden_size * 2]
        out = torch.sum(H * alpha, dim=1)
        
        # 注意：我们同时返回 out 和 alpha
        # out 用于后续全连接层分类；alpha 用于大作业要求的可视化
        return out, alpha.squeeze(-1) 

# ================= 整合进 LSTM 模型 =================

class TextRNN_Att(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx):
        super(TextRNN_Att, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=2,           
            batch_first=True, 
            bidirectional=True      
        )
        
        # 引入刚才定义的 Attention 层
        self.attention = Attention(hidden_size)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        
        out = self.embedding(x)
        # out shape: [batch_size, seq_len, embed_size]
        
        # 获取 LSTM 所有时间步的输出
        out, _ = self.lstm(out)
        # out shape: [batch_size, seq_len, hidden_size * 2]
        
        # ✨ 关键改变：不再只取最后一个时间步 out[:, -1, :]
        # 而是把整个序列扔给 Attention 层进行加权融合
        out, att_weights = self.attention(out)
        # out shape: [batch_size, hidden_size * 2]
        # att_weights shape: [batch_size, seq_len]
        
        # 全连接层分类
        out = self.fc(out)
        # out shape: [batch_size, num_classes]
        
        # 返回分类 Logits 和注意力权重
        return out, att_weights