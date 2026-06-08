import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据 (继续保持跳过表头的正确设置)
df = pd.read_csv("/root/autodl-tmp/THUCNews/Train.tsv", sep='\t', names=['label_id', 'text'], header=0)

# 2. 数据清洗：将可能的空值(NaN)替换为空字符串，并确保强制转换为字符串类型
df['text'] = df['text'].fillna("").astype(str)

# 3. 统计每一行文本的字符长度，并存入新的一列 'text_length'
# 由于是中文字符，.str.len() 会准确统计汉字的数量（一个汉字算1个字符）
df['text_length'] = df['text'].str.len()

# 4. 计算统计量
average_length = df['text_length'].mean()
max_length = df['text_length'].max()
min_length = df['text_length'].min()
median_length = df['text_length'].median() # 中位数通常比平均数更能反映普遍情况

print("="*30)
print(f"数据总条数: {len(df)}")
print(f"最长文本长度: {max_length} 个字符")
print(f"最短文本长度: {min_length} 个字符")
print(f"平均字符长度: {average_length:.2f} 个字符")
print(f"文本长度中位数: {median_length:.2f} 个字符")
print("="*30)

# ================= 附加：绘制文本长度分布直方图 =================
plt.figure(figsize=(10, 6))

# 绘制直方图，使用你喜欢的暖色调（这里选用了柔和的橙色）
sns.histplot(df['text_length'], bins=50, kde=True, color='#F4A261', edgecolor='white')

# 画一条红色的虚线标出平均长度的位置
plt.axvline(average_length, color='#D62828', linestyle='--', linewidth=2, 
            label=f'平均长度 ({average_length:.1f})')

# 画一条紫色的虚线标出中位数的位置
plt.axvline(median_length, color='#6A0572', linestyle='-.', linewidth=2, 
            label=f'中位数 ({median_length:.1f})')

plt.title('THUCNews 验证集文本长度分布', fontsize=16, pad=15)
plt.xlabel('文本字符数量', fontsize=14)
plt.ylabel('样本数量 (频数)', fontsize=14)
plt.legend(fontsize=12)

# 保存图片
save_path = "/root/autodl-tmp/text_length_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"文本长度分布直方图已保存至: {save_path}")