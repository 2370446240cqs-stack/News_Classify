import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 标签映射字典
id2label = {0: '财经', 1: '彩票', 2: '房产', 3: '股票', 4: '家居', 5: '教育', 
            6: '科技', 7: '社会', 8: '时尚', 9: '时政', 10: '体育', 11: '星座', 
            12: '游戏', 13: '娱乐'}

# 2. 读取数据 (跳过表头并强制转换类型)
df = pd.read_csv("/root/autodl-tmp/THUCNews/Train.tsv", sep='\t', names=['label_id', 'text'], header=0)
df['label_id'] = df['label_id'].astype(int)

# 3. 将数字ID替换为中文标签名
df['label_name'] = df['label_id'].map(id2label)

# 4. 统计各个类别的具体数量
class_counts = df['label_name'].value_counts()

# 5. 绘制饼状图
plt.figure(figsize=(10, 10))  # 饼图建议画布长宽一致，这样画出来的圆更标准
colors = sns.color_palette('YlOrRd', len(class_counts))  # 提取 viridis 颜色盘

# 绘制核心代码
plt.pie(class_counts, 
        labels=class_counts.index,     # 扇形外侧的标签名
        autopct='%1.1f%%',             # 显示百分比，保留一位小数
        startangle=140,                # 旋转起始角度，让图形分布更美观
        colors=colors,                 # 填充颜色
        textprops={'fontsize': 12},    # 设置标签和百分比的字体大小
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}) # 给扇形加上白色边框，视觉更清晰

plt.title('THUCNews 训练集类别分布', fontsize=18, pad=20)

# 6. 保存图片 (我稍微改了一下文件名，避免覆盖之前的柱状图)
save_path = "/root/autodl-tmp/class_distribution_pie.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"饼状图已保存至: {save_path}")