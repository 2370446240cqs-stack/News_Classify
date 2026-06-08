一、项目路径说明
(1)项目中的五个模型训练代码train.py以及推理代码infer.py均存放在./News_Classfy/model路径下。在执行推理时部分路径需要手动修改，代码中已注释了变量对应的路径
(2)./News_Classfy/THUCNews路径下存放了数据集预处理以及可视化的脚本，下载的THUCNews数据集存放在这个路径下
(3)./News_Classfy/Deployment路径下存放了将训练好的lstm-attn-sougou模型导出的代码deployment.py，onx_infer.py用于测试导出的.onnx文件是否能成功运行。同时零样本模型的训练及推理代码也存放在此路径下。
(4)./News_Classfy/architecture路径下存放了模型的两个核心类TextRNN以及加入注意力机制的TextRNN-Attn
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