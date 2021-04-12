from numpy import *
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class PMF(object):
    def __init__(self, Lambda=0.1, feat=10, epsilon=1, alpha=0.1, momentum=0.8, epoch=100, batch=1000, batch_size=100):
        self.Lambda = Lambda  # 正则
        self.feat = feat  # 潜在特征数,即 D
        self.epsilon = epsilon
        self.alpha = alpha  # 学习率
        self.epoch = epoch  # epoch
        self.batch = batch  # 每个epoch更新次数
        self.batch_size = batch_size  # 样本数
        self.momentum = momentum  # 增量
        self.U = None  # 用户潜在特征向量
        self.V = None  # 电影潜在特征向量

        # 均方根误差
        self.rmse_train = []
        self.rmse_test = []

    # 读取数据
    def read_data(self, path='C:\\Users\\13653\\Desktop\\RCS\\1\\u.data'):
        data = []
        for i in open(path, 'r'):
            (uid, mid, rat, _) = i.split('\t')
            uid = int(uid)
            mid = int(mid)
            rat = float(rat)
            data.append([uid, mid, rat])
        return array(data)

    # 预处理数据
    def pre_handle_data(self, data, test_size=0.2):
        train_data = []
        test_data = []
        for i in data:
            rand = random.random()
            if rand > test_size:
                train_data.append(i)
            else:
                test_data.append(i)
        return array(train_data), array(test_data)

    def train(self, train_v, test_v):
        self.mean_inv = np.mean(train_v[:, 2])  # 评分的均值
        train_size, test_size = train_v.shape[0], test_v.shape[0]  # 训练集和测试集的大小

        # 求出用户和电影的总数
        num_user = int(max(np.amax(train_v[:, 0]), np.amax(test_v[:, 0]))) + 1
        num_movie = int(max(np.amax(train_v[:, 1]), np.amax(test_v[:, 1]))) + 1
        # 因为其向量分别在第0列和第1列(其id值按增序0~n)

        self.V = 0.1 * np.random.randn(num_movie, self.feat)  # 电影 M x D 正态分布矩阵
        self.U = 0.1 * np.random.randn(num_user, self.feat)  # 电影 M x D 正态分布矩阵
        # 创建电影和用户的 M x D 0矩阵，用于增量梯度下降
        self.V_inc = np.zeros((num_movie, self.feat))
        self.U_inc = np.zeros((num_user, self.feat))

        for epoch_now in range(0, self.epoch):
            # 打乱训练集元组
            shuffled_order = np.arange(train_v.shape[0])  # 生成等差数列0~n
            np.random.shuffle(shuffled_order)  # 打乱顺序

            # 梯度更新
            for batch in range(self.batch):
                # 获取此次需要学习的元素下标
                test = np.arange(self.batch_size * batch,
                                 self.batch_size * (batch + 1))
                batch_index = np.mod(
                    test, shuffled_order.shape[0])  # 本次迭代要使用的索引下标
                batch_UserID = np.array(
                    train_v[shuffled_order[batch_index], 0], dtype='int32')
                batch_MovieID = np.array(
                    train_v[shuffled_order[batch_index], 1], dtype='int32')

                # 计算目标函数
                pred_out = np.sum(np.multiply(
                    self.U[batch_UserID, :], self.V[batch_MovieID, :]), axis=1)
                # 加平均值
                rawErr = pred_out - \
                    train_v[shuffled_order[batch_index], 2] + self.mean_inv

                # 计算梯度
                Ix_U = 2 * np.multiply(rawErr[:, np.newaxis], self.V[batch_MovieID, :]) \
                    + self.Lambda * self.U[batch_UserID, :]
                Ix_V = 2 * np.multiply(rawErr[:, np.newaxis], self.U[batch_UserID, :]) \
                    + self.Lambda * (self.V[batch_MovieID, :])
                dw_V = np.zeros((num_movie, self.feat))
                dw_U = np.zeros((num_user, self.feat))

                # 计算对于同一个用户或者商品的梯度和，然后统一进行更新
                for i in range(self.batch_size):
                    dw_V[batch_MovieID[i], :] += Ix_V[i, :]
                    dw_U[batch_UserID[i], :] += Ix_U[i, :]

                # 增量更新
                self.V_inc = self.momentum * self.V_inc + self.epsilon * dw_V / self.batch_size
                self.U_inc = self.momentum * self.U_inc + self.epsilon * dw_U / self.batch_size

                self.V = self.V - self.V_inc * self.alpha
                self.U = self.U - self.U_inc * self.alpha

                # 计算训练集误差
                if batch == self.batch - 1:
                    pred_out = np.sum(np.multiply(self.U[np.array(train_v[:, 0], dtype='int32'), :],
                                                  self.V[np.array(train_v[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_v[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                        + 0.5 * \
                        self.Lambda * (np.linalg.norm(self.U)
                                        ** 2 + np.linalg.norm(self.V) ** 2)

                    self.rmse_train.append(np.sqrt(obj / train_size))

                # Compute validation error
                if batch == self.batch - 1:
                    pred_out = np.sum(np.multiply(self.U[np.array(test_v[:, 0], dtype='int32'), :],
                                                  self.V[np.array(test_v[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_v[:, 2] + self.mean_inv
                    self.rmse_test.append(np.linalg.norm(
                        rawErr) / np.sqrt(test_size))

                    # Print info
                    if batch == self.batch - 1:
                        print('Training RMSE: %f, Test RMSE %f' %
                              (self.rmse_train[-1], self.rmse_test[-1]))


if __name__ == "__main__":
    file_path = "C:\\Users\\13653\\Desktop\\RCS\\1\\u.data"
    pmf = PMF()
    # __init__(self, Lambda=0.1, feat=10, epsilon=1, alpha=0.1, momentum=0.9, epoch=20, batch=100, batch_size=1000)
    # pmf.set_params({"num_feat": 10, "epsilon": 1, "Lambda": 0.1, "momentum": 0.8, "maxepoch": 100, "batch": 100,
    #                 "batch_size": 1000})
    ratings = pmf.read_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(
        np.unique(ratings[:, 1])), pmf.feat)
    train, test = train_test_split(
        ratings, test_size=0.2)  # spilt_rating_dat(ratings)
    pmf.train(train, test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.epoch), pmf.rmse_train,
             marker='o', label='Training Data')
    plt.plot(range(pmf.epoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    # print("precision_acc,recall_acc:" + str(pmf.topK(test)))
