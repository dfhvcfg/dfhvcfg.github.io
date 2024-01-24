# 短期电力负荷预测 (STLF) 使用 CNN-BiLSTM

本项目利用卷积神经网络（CNN）和双向长短期记忆网络（BiLSTM）模型来进行短期电力负荷预测。项目包括数据预处理、模型构建、训练、预测以及性能评估等步骤，旨在提供一个高精度的电力负荷预测工具。


## 使用说明

### 数据预处理

- 数据预处理，包括季节性分解和归一化处理。

### 模型训练和预测

- 使用Jupyter笔记本`STLF_CNN_BiLSTM.ipynb`中提供的代码进行模型的训练和预测。

### 性能评估

- 查看笔记本中的评估部分，了解模型性能和预测结果的详细分析。

## 关键功能和脚本说明

- **split_dataset**: 将数据集分为训练集、验证集和测试集，重塑数据格式以适配深度学习模型的输入要求。
- **evaluate_forecasts**: 评估模型的预测性能，计算和打印RMSE得分。
- **summarize_scores**: 汇总并打印模型的整体RMSE得分及每小时的RMSE得分。
- **convert_train_val**: 重塑训练和验证数据，为深度学习模型训练准备输入和输出数据。
- **forecast**: 使用训练好的模型进行负荷预测。
- **build_model_CNNBiLSTM**: 构建CNN-BiLSTM模型，包括卷积层、双向LSTM层和全连接层。
- **evaluate_model_CNNBiLSTM**: 评估CNN-BiLSTM模型在测试集上的性能并进行负荷预测。

## 结果展示

- 分析时间序列数据中的季节性成分
![示例图片](images/example.png)

