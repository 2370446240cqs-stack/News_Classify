import csv
import re
import os

input_path = '/root/autodl-tmp/THUCNews/Train_origin.tsv'
output_path = '/root/autodl-tmp/THUCNews/Train.tsv'

# 🚀 升级版正则表达式解析：
# 第一部分 [(\uff08].*?[)\uff09] ：匹配正常的、成对的括号及内容
# 第二部分 | ：或者
# 第三部分 [(\uff08][^)\uff09]*$ ：匹配一个左括号，且后面跟着“不是右括号”的任意字符，直到文本结尾 ($)
pattern = re.compile(r'[(\uff08].*?[)\uff09]|[(\uff08][^)\uff09]*$')

if not os.path.exists(input_path):
    print(f"❌ 找不到文件: {input_path}，请检查路径。")
else:
    print(f"🚀 正在处理文件: {input_path} ...")
    
    processed_lines = 0
    modified_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.reader(f_in, delimiter='\t', quoting=csv.QUOTE_NONE)
        writer = csv.writer(f_out, delimiter='\t', quoting=csv.QUOTE_NONE)
        
        for row in reader:
            if len(row) >= 2:
                original_text = row[1]
                
                # 替换掉完整括号内容，以及行尾未闭合的残缺括号内容
                cleaned_text = pattern.sub('', original_text)
                
                if original_text != cleaned_text:
                    modified_lines += 1
                
                row[1] = cleaned_text
            
            writer.writerow(row)
            processed_lines += 1

    print(f"✅ 处理完成！")
    print(f"📊 统计：共处理 {processed_lines} 行文本，其中清理了 {modified_lines} 行包含括号的内容。")
    print(f"💾 清理后的数据已保存至: {output_path}")