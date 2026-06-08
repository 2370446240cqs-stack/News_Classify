import ast
import torch
from torch.utils.data import Dataset, DataLoader

class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=30):
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
            next(f) # 如果第一行是表头 label \t text_a，取消注释这行；如果没有表头，注释掉这行
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

# 使用示例
if __name__ == "__main__":
    dataset = THUCNewsDataset("/root/autodl-tmp/THUCNews/Train.tsv", "/root/autodl-tmp/THUCNews/dict.txt", max_len=30)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"词典总大小: {len(dataset.char2id)}")
    print(f"<UNK> 的 ID 是: {dataset.unk_id}")
    print(f"<PAD> 的 ID 是: {dataset.pad_id}")
    
    for batch_x, batch_y in dataloader:
        print("\n=== 首个 Batch 数据预览 ===")
        print("样本ID (batch_x[0]):", batch_x[0])
        print("标签ID (batch_y[0]):", batch_y[0])
        break