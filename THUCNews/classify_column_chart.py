import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体 (你刚刚已经学会并配置好了)
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 标签映射字典
id2label = {0: '财经', 1: '彩票', 2: '房产', 3: '股票', 4: '家居', 5: '教育', 
            6: '科技', 7: '社会', 8: '时尚', 9: '时政', 10: '体育', 11: '星座', 
            12: '游戏', 13: '娱乐'}

# 2. 读取数据 (假设第一行无表头，如果有表头请去掉 header=None)
#df = pd.read_csv("/root/autodl-tmp/THUCNews/Val.tsv", sep='\t', names=['label_id', 'text'])
df = pd.read_csv("/root/autodl-tmp/THUCNews/Val.tsv", sep='\t', names=['label_id', 'text'], header=0)
# ========== 添加这两行排查代码 ==========
print("数据前5行长这样：\n", df.head())
print("数据类型是：\n", df.dtypes)
# ========================================
# 3. 将数字ID替换为中文标签名
df['label_name'] = df['label_id'].map(id2label)

# 4. 统计并绘图
plt.figure(figsize=(12, 6))
# 按照数量降序排列
order = df['label_name'].value_counts().index
sns.countplot(data=df, x='label_name', order=order, palette='viridis')

plt.title('THUCNews 验证集类别分布', fontsize=16)
plt.xlabel('新闻类别', fontsize=14)
plt.ylabel('样本数量', fontsize=14)
plt.xticks(rotation=45, fontsize=12)

# 在柱子上标明具体数字
for p in plt.gca().patches:
    plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=10)

plt.savefig("/root/autodl-tmp/class_distribution.png", dpi=300, bbox_inches='tight')
print("分布图已保存至: /root/autodl-tmp/class_distribution.png")