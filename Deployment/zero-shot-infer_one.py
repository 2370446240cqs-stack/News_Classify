import os
# 关键魔法：将 HuggingFace 的默认下载源替换为国内镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline

print("正在从国内镜像站下载或加载模型，请稍候...")
# 加载多语言零样本分类管道
classifier = pipeline("zero-shot-classification", 
                      model="joeddav/xlm-roberta-large-xnli", 
                      device=0) 

candidate_labels = ["财经", "彩票", "房产", "股票", "家居", "教育", 
                    "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"]

test_text = "上证指数今日大跌两百点，散户恐慌性抛售"

result = classifier(test_text, candidate_labels)

print(f"\n输入文本: {result['sequence']}")
print(f"预测最高类别: {result['labels'][0]}, 概率: {result['scores'][0]:.4f}")