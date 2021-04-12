import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from pylab import *
import matplotlib
import matplotlib.pyplot as plt


class PMF(object):
    def __init__(self, Lambda=0.1, D=10, learning_rate=0.005, epoch=5):
        self.Lambda = Lambda
        self.D = D  # 潜在特征维度
        self.M = None  # 电影数
        self.N = None  # 用户数
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.train_list = None
        self.test_list = None
        self.data = None
        self.rated_movie_id_dict = None

    def read_data(self, path='C:\\Users\\13653\\Desktop\\RCS\\1\\u.data'):
        data = []
        N, M, u_idx, m_idx = 0, 0, 0, 0
        user_id_dict, rated_movie_id_dict = {}, {}
        index2movie = {}
        index2user = {}
        for i in open(path, 'r'):
            # 用户id，电影id，评分
            (uid, mid, rat, _) = i.split('\t')
            if int(uid) not in user_id_dict:
                user_id_dict[int(uid)] = u_idx
                index2user[u_idx] = int(uid)
                u_idx += 1
            if int(mid) not in rated_movie_id_dict:
                rated_movie_id_dict[int(mid)] = m_idx
                index2movie[m_idx] = int(mid)
                m_idx += 1
                # [用户id位置，电影id位置，评分]
            data.append([user_id_dict[int(uid)],
                         rated_movie_id_dict[int(mid)], float(rat)])
        # 用户数和电影数
        N = u_idx
        M = m_idx
        self.user_id_dict = user_id_dict
        print("用户数%f, 电影数%f,数据集长度%f" % (N, M, len(data)))
        return N, M, data, index2movie, index2user

    def list2mat(self, sequence, N, M):
        records_array = np.array(sequence)
        mat = np.zeros([N, M])
        row = records_array[:, 0].astype(int)
        col = records_array[:, 1].astype(int)
        values = records_array[:, 2].astype(np.float32)
        mat[row, col] = values
        return mat

    def initParameters(self):
        self.N, self.M, self.data, self.index2movie,self.index2user = self.read_data()
        self.train_list, self.test_list = train_test_split(
            self.data, test_size=0.2)

    def train(self):
        U = np.random.normal(0, 0.1, (self.N, self.D))
        V = np.random.normal(0, 0.1, (self.M, self.D))

        train_mat = self.list2mat(sequence=self.train_list, N=self.N, M=self.M)
        test_mat = self.list2mat(sequence=self.test_list, N=self.N, M=self.M)

        records = []

        for step in range(self.epoch):
            los = 0
            for data in self.train_list:
                user, movie, rate = data
                U[user], V[movie], ls = self.update(U[user], V[movie], rate=rate,
                                                    learning_rate=self.learning_rate,
                                                    Lambda=self.Lambda)
                los += ls
            pred_mat = self.prediction(U, V)
            rsme_train, rsme_test = self.evaluation(
                pred_mat, train_mat, test_mat)
            records.append(np.array([los, rsme_train, rsme_test]))

            print(' step:%d \n loss:%.4f,rmse_train:%.4f,rmse_tese:%.4f'
                  % (step, los, rsme_train, rsme_test))

        print(' end. \n loss:%.4f,rsme_train:%.4f,rmse_test:%.4f'
              % (records[-1][0], records[-1][1], records[-1][2]))
        return U, V, np.array(records)

    def update(self, u, v, rate, learning_rate, Lambda):
        error = rate - np.dot(u, v.T)
        u = u + learning_rate * (error * v - Lambda * u)
        v = v + learning_rate * (error * u-Lambda * v)
        loss = 0.5 * (error**2 + Lambda *
                      (np.square(u).sum() + np.square(v).sum()))
        return u, v, loss

    def prediction(self, U, V):
        N, D = U.shape
        M, D = V.shape
        rate_list = []

        for u in range(N):
            u_rate = np.sum(U[u, :]*V, axis=1)
            rate_list.append(u_rate)
        pred_result = np.array(rate_list)
        return pred_result

    def evaluation(self, pred_mat, train_mat, test_mat):
        y_pred_test = pred_mat[test_mat > 0]
        y_true_test = test_mat[test_mat > 0]
        rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

        y_pred_train = pred_mat[train_mat > 0]
        y_true_train = train_mat[train_mat > 0]
        rmse_train = np.sqrt(mean_squared_error(y_pred_train, y_true_train))

        return rmse_train, rmse_test


if __name__ == "__main__":
    print('begin')
    file_path = "C:\\Users\\13653\\Desktop\\RCS\\1\\u.data"
    pmf = PMF(Lambda=0.1, D=10, learning_rate=0.005, epoch=50)
    pmf.initParameters()
    U, V, records = pmf.train()
    predict = pmf.prediction(U=U, V=V)
    index2user = pmf.index2user
    index2movie = pmf.index2movie
    user_id_dict = pmf.user_id_dict

    plt.plot(range(pmf.epoch), records[:,1], marker='.', label='Training Data')
    plt.plot(range(pmf.epoch), records[:,2], marker='o', label='Test Data')
    plt.title('MovieLens')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()

    userid = 1
    while userid != -1:
        userid = int(input('请输入要预测的用户id'))
        userid_index = user_id_dict[userid]
        print(userid_index)
        # 预测评分前五
        predict_seq = predict[userid_index-1].argsort()[-5:][::-1]
        print('用户%d的推荐:'%(userid))
        for i in predict_seq:
            print('电影%d,评分%.2f'%(index2movie[i],predict[userid_index][i]))