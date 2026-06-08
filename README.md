readme_content = """# 新闻文本分类项目 (News_Classfy)

这是一个基于深度学习的新闻文本分类项目，包含多种模型的训练、推理、模型导出（ONNX）以及零样本学习（Zero-shot Learning）的实现。

## 一、项目路径说明

为了方便管理，项目核心目录结构及说明如下：

```text
News_Classfy/
├── model/               # 存放模型训练、推理代码及权重文件
│   ├── train.py         # 5个模型的训练脚本
│   └── infer.py         # 推理与模型评估脚本
├── THUCNews/            # 数据集预处理与可视化目录
│   ├── process_data.py  # 数据预处理脚本
│   └── devide.py        # 数据集划分脚本
├── architecture/        # 模型网络结构定义
│   ├── TextRNN.py       # 基础 TextRNN 网络类
│   └── TextRNN-Attn.py  # 加入注意力机制的 TextRNN-Attn 网络类
└── Deployment/          # 模型导出与部署测试
    ├── deployment.py    # lstm-attn-sougou 模型导出为 ONNX 脚本
    ├── onx_infer.py     # ONNX 模型运行与推理测试脚本
    └── zero_shot/       # 零样本模型的训练及推理代码
二、实验环境
(1)本项目在autodl上的RTX4090 24GB上运行，在Python 3.12.3上成功跑通。每个模型的训练时间大约在30分钟左右。
(2)零样本模型计算量较大，训练时间为2.2小时左右。
三、数据集下载以及sougou词向量下载
(1)THUCNews数据集网址:http://thuctc.thunlp.org/
(2)sougou词向量下载(本项目使用了其中SGNS格式的sougou新闻char词向量):https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw
四、快速开始
(1)下载THUCNews数据集，将数据集中的内容放到THUCNews文件夹中，使用process_data.py进行数据预处理。使用devide.py将处理好的训练集进行划分
(2)下载sougou词向量，将下载好的文件解压并放到任意位置，在后续训练中，你可以修改train.py中的词向量的存放路径
(3)尝试从./News_Classfy/TextRNN/train.py开始训练，按照提示进行依赖的安装。
(4)训练完成后，权重会保存在当前目录下，请新建./News_Classfy/model文件夹，将权重保存到model文件夹中。
(5)保存完毕后，运行infer.py进行模型评估