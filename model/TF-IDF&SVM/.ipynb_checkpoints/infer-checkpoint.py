import joblib
import jieba
import os
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. 全局分词函数（必须保持与训练时完全一致）
# ==========================================
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# ==========================================
# 2. 类别映射字典
# ==========================================
ID_TO_CATEGORY = {
    0: '财经', 1: '彩票', 2: '房产', 3: '股票', 
    4: '家居', 5: '教育', 6: '科技', 7: '社会', 
    8: '时尚', 9: '时政', 10: '体育', 11: '星座', 
    12: '游戏', 13: '娱乐'
}

# ==========================================
# 3. 验证主函数
# ==========================================
def run_evaluation():
    # 路径配置
    model_path = '/root/autodl-tmp/model/IFDF-SVM.pkl'
    val_file_path = '/root/autodl-tmp/THUCNews/test.tsv'          # 你的验证集文件
    output_file_path = '/root/autodl-tmp/val_evaluation_result.tsv' # 详细比对结果保存路径

    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件：{model_path}")
        return
        
    print(f"正在加载模型: {model_path} ...")
    pipeline = joblib.load(model_path)
    print("✅ 模型加载成功！\n")

    # 2. 读取验证集数据
    print(f"正在读取验证集数据: {val_file_path} ...")
    y_true_ids = []
    texts = []
    
    with open(val_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            # 确保这行数据有两列：标签ID 和 文本
            if len(parts) == 2:
                try:
                    label_id = int(parts[0])
                    text = parts[1]
                    y_true_ids.append(label_id)
                    texts.append(text)
                except ValueError:
                    # 如果第一列无法转为数字（比如遇到了表头 label_id\ttext_a），则跳过这一行
                    continue
                
    print(f"✅ 数据读取完成，共检测到 {len(texts)} 条有效文本。\n")

    # 3. 批量预测
    print("开始进行模型推理，请稍候...")
    y_pred_ids = pipeline.predict(texts)
    print("✅ 推理完成！\n")

    # 4. 打印整体评估指标
    print("=" * 50)
    print("🎯 验证集整体评估报告")
    print("=" * 50)
    accuracy = accuracy_score(y_true_ids, y_pred_ids)
    # ✨ 修改：将百分比的显示也调整为 4 位小数 (.4f)
    print(f"整体准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.4f}%)\n")
    
    # 打印包含 F1-score 的详细报告，并使用中文类别名
    target_names = [ID_TO_CATEGORY.get(i, f"类{i}") for i in range(14)]
    # ✨ 修改：在此处增加了 digits=4 参数，使分类报告输出 4 位小数
    print(classification_report(y_true_ids, y_pred_ids, target_names=target_names, digits=4))

    # 5. 保存详细对比结果（方便做错误分析 Bad Case Analysis）
    print(f"\n正在将详细比对结果写入: {output_file_path} ...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 写入表头，增加 Is_Correct 列方便你用 Excel 筛选
        f.write("Is_Correct\tTrue_Label\tPredict_Label\tText\n")
        
        for true_id, pred_id, text in zip(y_true_ids, y_pred_ids, texts):
            true_cat = ID_TO_CATEGORY.get(true_id, "未知")
            pred_cat = ID_TO_CATEGORY.get(pred_id, "未知")
            # 判断是否预测正确
            is_correct = "✅" if true_id == pred_id else "❌"
            
            f.write(f"{is_correct}\t{true_cat}\t{pred_cat}\t{text}\n")

    print(f"🎉 验证完成！你可以打开 {output_file_path} 查看模型在每一条数据上的表现。")

if __name__ == '__main__':
    run_evaluation()