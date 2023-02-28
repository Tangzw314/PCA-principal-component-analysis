# 作者: Tangzw314
# 开发时间：2022/11/17 17:00
import numpy as np
import pandas as pd
import pyreadstat
from copy import deepcopy


# 定义一个主成分分析的类
class PCA:
    """
    主成分分析的主要步骤如下：
    1. 求原始变量的的相关系数矩阵（原始变量标准化后的协方差矩阵）；
    2. 求相关系数矩阵的特征值和特征向量，并对特征向量进行标准化，特征值即每个主成分的方差，标准化后的特征向量即线性变换的系数；
    3. 对特征值进行降序排序，计算方差贡献率，计算累计方差贡献率，计算因子载荷量，即新的影响因子与原始因子的相关性；
    4. 确定要保留的主成分个数，一般依据85%原则，即至少能表示原来85%的信息，即累计方差贡献率第一次大于85%的主成分及之前的主成分为新的影响因子；
    5. 对分析对象进行综合评价，即计算每个个体在每个主成分上的数值，最后对所有主成分的值进行加权平均，权重为每个主成分的方差贡献率；也可单独计算
       某个主成分的值，即分别评价每个个体在每个特征上的表现；
    """

    def __init__(self, threshold=0.85):
        """
        初始化类
        :param threshold: 累计方差贡献率的阀值，即保留主成分的标准，默认是85%。
        """
        self.threshold = threshold
        self.df = None
        self.corr_ = None
        self.eigen_value = None
        self.eigen_vector = None
        self.feature_names = None
        self.all_pca_df = None

    def fit(self, df):
        """
        定义一个拟合函数，主要是对数据进行初始化处理
        :param df: 原变量数据集
        :return: None
        """
        self.df = deepcopy(df)    # 获得原始数据
        feature_names = self.df.columns.to_numpy()   # 获得原始特征的名称
        corr_ = self.df.corr()  # 计算原始变量的相关系数矩阵

        self.feature_names = feature_names      # 获得原始特征的名称
        self.corr_ = corr_    # 原始变量的相关系数矩阵
        self.eigen_value, self.eigen_vector = self._get_eig()    # 特征值和特征向量

        # 构建特征矩阵，index是特征值，columns是原始变量的名称， 每一行为每个特征值的特征向量
        eigen_df = pd.DataFrame(self.eigen_vector, columns=self.eigen_value, index=self.feature_names).T
        eigen_df.sort_index(axis=0, inplace=True, ascending=False)      # 按特征值从大到小排列特征向量
        index = [f"第{i + 1}主成分" for i in range(len(eigen_df))]      # 依据特征值大小给特征值命名
        eigen_df.index = index
        self.all_pca_df = eigen_df

    def _get_eig(self):
        """
        定义一个求特征值和特征向量的函数
        :return: 特征值和特征向量
        """
        eigen_value, eigen_vector = np.linalg.eig(self.corr_)
        # self.eigen_value, self.eigen_vector = eigen_value, eigen_vector

        return eigen_value, eigen_vector

    def get_pca_desc(self):
        """
        这是一个描述的主成分DataFrame，其中包括主成分的方差，方差贡献率，
        累计方差贡献率，累计方差。
        :return: DataFrame
        """
        index = [f"第{i + 1}主成分" for i in range(len(self.eigen_value))]
        df_pca_desc = pd.DataFrame(self.eigen_value, columns=['方差'])
        df_pca_desc.sort_values(by='方差', ascending=False, inplace=True)
        df_pca_desc.index = index
        df_pca_desc['方差贡献率'] = df_pca_desc['方差'] / df_pca_desc['方差'].sum()
        df_pca_desc['累计方差贡献率'] = df_pca_desc['方差贡献率'].cumsum()
        df_pca_desc['累计方差'] = df_pca_desc['方差'].cumsum()

        return df_pca_desc

    def get_factor_load_matrix(self):
        """
        这是一个获得因子载荷矩阵的方法，
        因子载荷矩阵，即主成分与原始因子的相关系数矩阵
        :return: DataFrame
        """
        # 构建特征矩阵，转置后的index是特征值，columns是原始变量的名称
        eigen_df = pd.DataFrame(self.eigen_vector, columns=self.eigen_value, index=self.feature_names).T
        eigen_df.sort_index(axis=0, inplace=True, ascending=False)

        # 计算因子载荷矩阵
        factor_load_matrix = eigen_df * np.sqrt(eigen_df.index.to_numpy()).reshape((-1, 1))
        index = [f"第{i + 1}主成分" for i in range(len(eigen_df))]
        factor_load_matrix.index = index

        return factor_load_matrix

    def get_result(self):
        """
        这是一个输出结果函数，返回主成分分析的结果，
        结果的每一行代表一个主成分。
        :return: DataFrame
        """
        pca_desc = self.get_pca_desc()
        index_end = pca_desc[pca_desc['累计方差贡献率'] >= self.threshold].index[0]
        pca_df = self.all_pca_df.loc[:index_end, :]

        return pca_df

    def get_evaluation_report(self):
        """
        这是一个对分析对象进行评价的函数，返回一个得分DataFrame，和一个排名DataFrame。
        单个主成分的评价步骤如下：
        1. 计算每个个体在每个主成分上的取值，令其为该个体在该主成分上的得分；
        2. 按得分对个体进行排名

        综合评价的具体步骤如下（注：这里的的主成分是经过选择后留下来的主成分）：
        1. 计算每个个体在每个主成分上的取值；
        2. 确定每个主成分的权重，权重一般取每个主成分的方差贡献率，也可使用每个主成分的方差占留下来的主成分的总方差的比例做权重，这里我们使用方差贡献率；
        3. 对每个个体在所有主成分上的取值求加权平均，令其为该个体的综合得分。
        4. 按得分对个体进行排名
        :return: DataFrame
        """
        data = self.df      # 获取原始数据
        std_data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
        score_df = pd.DataFrame()   # 准备存放得分的DataFrame
        pca_df = self.get_result()  # 获得保留的的主成分，即保留的特征值和特征向量
        pca_desc = self.get_pca_desc()
        weights = pca_desc.loc[pca_df.index, '方差贡献率'].values    # 获得权重

        # 计算得分矩阵
        score_columns = [f'“{column}”得分' for column in pca_df.index]
        for eigen_vector, score_column in zip(pca_df.values, score_columns):
            score_df[score_column] = (std_data * eigen_vector).sum(axis=1)
        score_df["综合得分"] = (score_df * weights).sum(axis=1)
        rank_columns = {score_column: score_column.replace('得分', '排名') for score_column in score_df.columns}

        # 计算排名矩阵
        rank_df = score_df.rank(method='dense', ascending=False)
        rank_df.rename(columns=rank_columns, inplace=True)

        return score_df, rank_df

