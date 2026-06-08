import csv
import random
import collections
import os

input_path = '/root/autodl-tmp/THUCNews/Train.tsv' # 请确保读取的是清洗后的文件
train_path = '/root/autodl-tmp/THUCNews/train.tsv'
dev_path   = '/root/autodl-tmp/THUCNews/dev.tsv'
test_path  = '/root/autodl-tmp/THUCNews/test.tsv'

RANDOM_SEED = 42

if not os.path.exists(input_path):
    print(f"❌ 找不到文件: {input_path}，请检查路径。")
else:
    print(f"🚀 正在读取并按类别分组文件: {input_path} ...")
    
    data_by_class = collections.defaultdict(list)
    skipped_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) >= 2:
                # 获取第一列标签并去除两端可能的空格
                label = str(row[0]).strip()
                
                # 🛡️ 核心修复：只保留纯数字标签，过滤掉表头（如 'label'）或脏数据
                if not label.isdigit():
                    if skipped_lines == 0:
                        print(f"⚠️ 发现并跳过非数字标签 (可能是表头): '{label}'")
                    skipped_lines += 1
                    continue
                    
                data_by_class[label].append(row)
                
    train_data = []
    dev_data = []
    test_data = []
    
    print("\n📊 各类别数据统计及划分详情：")
    print("-" * 65)
    print(f"{'类别':<5} | {'总数':<10} | {'训练集(70%)':<12} | {'验证集(15%)':<12} | {'测试集(15%)'}")
    print("-" * 65)
    
    # 🛡️ 核心修复：因为前面已经过滤了非数字，这里可以直接放心地用 int 排序
    sorted_labels = sorted(data_by_class.keys(), key=int)
    
    random.seed(RANDOM_SEED)
    
    for label in sorted_labels:
        items = data_by_class[label]
        total_count = len(items)
        
        random.shuffle(items)
        
        train_split = int(total_count * 0.70)
        dev_split   = int(total_count * 0.85)
        
        train_part = items[:train_split]
        dev_part   = items[train_split:dev_split]
        test_part  = items[dev_split:]
        
        train_data.extend(train_part)
        dev_data.extend(dev_part)
        test_data.extend(test_part)
        
        print(f"{label:<7} | {total_count:<12} | {len(train_part):<16} | {len(dev_part):<16} | {len(test_part)}")
    
    print("-" * 65)
    print(f"{'总计':<5} | {len(train_data)+len(dev_data)+len(test_data):<10} | {len(train_data):<12} | {len(dev_data):<12} | {len(test_data)}")
    if skipped_lines > 0:
        print(f"💡 提示：共跳过了 {skipped_lines} 行非正常数据 (含表头)。")
        
    print("\n🔄 正在对生成的训练集、验证集和测试集进行全局打乱...")
    
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    def write_tsv(file_path, data):
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            writer.writerows(data)
        print(f"✅ 已保存 {len(data)} 条数据至: {file_path}")

    print("\n💾 开始保存文件...")
    write_tsv(train_path, train_data)
    write_tsv(dev_path, dev_data)
    write_tsv(test_path, test_data)
    
    print("\n🎉 数据集划分全部完成！")