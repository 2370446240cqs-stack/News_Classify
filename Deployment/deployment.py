import torch
import torch.nn as nn
import torch.nn.functional as F
import ast

# 1. 拷贝你最新的 Attention 模型结构
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w, -0.1, 0.1)

    def forward(self, H):
        M = torch.tanh(H) 
        alpha = F.softmax(torch.matmul(M, self.w), dim=1) 
        out = torch.sum(H * alpha, dim=1)
        return out, alpha.squeeze(-1) 

class TextRNN_Att(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, pad_idx):
        super(TextRNN_Att, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out, att_weights = self.attention(out)
        out = self.fc(out)
        return out, att_weights

if __name__ == "__main__":
    device = torch.device('cpu') # 导出模型通常在 CPU 上进行即可
    
    # 获取动态词典大小和 PAD_ID
    dict_path = "/root/autodl-tmp/THUCNews/dict.txt"
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            char2id = ast.literal_eval(f.read())
    except UnicodeDecodeError:
        with open(dict_path, 'r', encoding='gbk') as f:
            char2id = ast.literal_eval(f.read())
            
    pad_id = max(char2id.values()) + 1 if '<pad>' not in char2id and '<PAD>' not in char2id else char2id.get('<pad>', char2id.get('<PAD>'))
    VOCAB_SIZE = len(char2id) + (1 if '<pad>' not in char2id and '<PAD>' not in char2id else 0)
    
    # 2. 初始化模型并加载权重
    model = TextRNN_Att(vocab_size=VOCAB_SIZE, embed_size=300, hidden_size=128, num_classes=14, pad_idx=pad_id)
    # 注意：如果你之前没用搜狗预训练词向量，请把上面的 embed_size 改回 256
    model.load_state_dict(torch.load("/root/autodl-tmp/lstm_sougou_model.pth", map_location=device))
    model.eval() # 务必切换到 eval 模式

    # 3. 构造一个 Dummy Input (Batch Size = 1, Sequence Length = 30)
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, 30), dtype=torch.long)

    # 4. 执行导出
    onnx_file_path = "/root/autodl-tmp/textrnn_att.onnx"
    print("正在导出为 ONNX 格式...")
    
    torch.onnx.export(
        model,                      # 要导出的模型
        dummy_input,                # 假数据
        onnx_file_path,             # 导出的文件名
        export_params=True,         # 将训练好的权重一并导出
        opset_version=14,           # ONNX 的算子集版本，14 是比较稳定支持 LSTM 的版本
        do_constant_folding=True,   # 开启常量折叠优化
        input_names=['input_ids'],  # 给输入节点起个名字
        output_names=['logits', 'attention_weights'], # 给输出节点起个名字
        dynamic_axes={              # 设置动态维度：允许在推理时改变 Batch Size
            'input_ids': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'attention_weights': {0: 'batch_size'}
        }
    )
    
    print(f"[*] 导出成功！模型已保存为: {onnx_file_path}")