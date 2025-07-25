基于机器学习的心理健康预测
本项目旨在通过健康和人口学数据，使用机器学习技术预测心理健康结果。主要关注通过筛查调查数据预测抑郁症、高血压和糖尿病的存在，并使用不同的机器学习模型来提高预测准确性。

项目结构
本项目包含以下几个模块：
1. 数据预处理
从CSV文件中加载健康和人口学信息数据。
使用插值和删除策略处理缺失值。
使用独热编码（One-Hot Encoding）和标签编码（Label Encoding）对数据进行编码。
2. 探索性数据分析 (EDA)
对变量（如PHQ分数）进行基本统计分析。
可视化分析包括：
PHQ分数分布的饼状图。
疾病状态（高血压、糖尿病、抑郁症）频率的柱状图。
年龄分布的密度图。
3. 数据合并
根据 PID 列将多个数据集合并。
删除不必要的列，简化数据集。
4. 特征工程
对分类特征进行独热编码（One-Hot Encoding）。
绘制特征重要性柱状图，展示关键预测变量。
5. 机器学习模型
使用的模型：
随机森林分类器（Random Forest Classifier）
支持向量机（SVM）
训练与测试：
将数据集划分为训练集和测试集（80%-20%）。
使用 GridSearchCV 进行超参数调整。
通过交叉验证和准确率评估模型性能。
6. 性能指标
随机森林模型在超参数调整后取得较高准确率。
SVM模型也进行了测试，采用线性核，表现较为稳定。
通过交叉验证评估模型的稳定性和可靠性。
绘制特征重要性图表，分析关键特征对预测的贡献。
