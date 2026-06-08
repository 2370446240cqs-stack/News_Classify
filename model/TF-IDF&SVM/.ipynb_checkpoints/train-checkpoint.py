import ast
import torch
from torch.utils.data import Dataset
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# ==========================================
# 1. 完整保留的数据集定义
# ==========================================
class THUCNewsDataset(Dataset):
    def __init__(self, tsv_path, dict_path, max_len=30):
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
            next(f)  # 跳过表头
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
# 2. 【核心修复】将分词函数移至全局作用域
# ==========================================
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# ==========================================
# 3. 训练脚本主体（TF-IDF + SVM）
# ==========================================
def train_svm_pipeline():
    print("正在通过 Dataset 类加载数据...")
    # 实例化数据集
    train_dataset = THUCNewsDataset(tsv_path='/root/autodl-tmp/THUCNews/train.tsv', dict_path='/root/autodl-tmp/THUCNews/dict.txt')
    test_dataset = THUCNewsDataset(tsv_path='/root/autodl-tmp/THUCNews/val.tsv', dict_path='/root/autodl-tmp/THUCNews/dict.txt')

    # 从数据集中提取出原始文本字符串和标签
    X_train = [item[1] for item in train_dataset.data]
    y_train = [item[0] for item in train_dataset.data]
    
    X_test = [item[1] for item in test_dataset.data]
    y_test = [item[0] for item in test_dataset.data]

    print(f"数据加载完成。训练集样本数: {len(X_train)}, 测试集(验证集)样本数: {len(X_test)}")

    # 构建机器学习流水线 (这里会自动在全局寻找 chinese_tokenizer)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            max_df=0.9,
            min_df=2,
            ngram_range=(1, 2)
        )),
        ('clf', LinearSVC(C=1.0, dual=False, random_state=42))
    ])

    # 开始训练
    print("开始训练 SVM 模型（这可能需要几分钟，取决于数据集大小）...")
    pipeline.fit(X_train, y_train)
    print("模型训练完成！")

    # 预测与评估
    print("开始在测试集上进行评估...")
    y_pred = pipeline.predict(X_test)
    
    # 打印评估报告
    print("\n分类性能报告:")
    print(classification_report(y_test, y_pred))

    # 保存模型权重
    save_path = '/root/autodl-tmp/IFDF-SVM.pkl'
    print(f"\n正在将模型保存至: {save_path}")
    joblib.dump(pipeline, save_path)
    print("✅ 模型权重及TF-IDF词表已成功保存！")

if __name__ == '__main__':
    train_svm_pipeline()