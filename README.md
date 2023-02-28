# Principal-component-analysis
这是一个实现主成分分析的类
## 快速开始
- pca = PCA()    &emsp;# 构建模型
- pca.fit(df)    &emsp;# df是为原始数据，无需进行标准化处理，本方法默认会对数据进行标准化处理
- pca.get_pca_desc()     &emsp;# 返回一个描述主成分的DataFrame，包括各主成分的方差、方差贡献率、累计方差贡献率、累计 方 差。
- pca.get_factor_load_matrix()    &emsp;# 这是一个获得因子载荷矩阵的方法，因子载荷矩阵，即主成分与原始因子的相关系数矩阵。
- pca.get_result()    &emsp;# 返回主成分分析的结果（基于85%原则输出的结果），结果的每一行代表一个主成分。
- score, rank = pca.get_evaluation_report()      &emsp;# 计算得分和排名
