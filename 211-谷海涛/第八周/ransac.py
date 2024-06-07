import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt


def random_sampling(n, data_nums):   #随机抽样函数
    all_idxs = np.arange(data_nums)     #取得样本所有的索引
    np.random.shuffle(all_idxs)         #随机乱序
    sample_idxs = all_idxs[:n]          #获取采样点集
    text_idx = all_idxs[n:]             #获取测试点集

    return sample_idxs, text_idx


def linear_least_square_model(data):        #使用最小二乘法计算model参数值，即计算y = ax + b中的a和b
    xi_tmp = data[:, 0:1]                   #取得第一列，即输入值
    xi = np.hstack((xi_tmp ** 0, xi_tmp ** 1))      #构造矩阵
    yi = data[:, 1:2]                               #取得第二列，即输出值
    model, *_ = la.lstsq(xi, yi)                    #调用scipy.linalg.lstsq函数计算参数值
    return model                                    #参数值，是一个2x1的向量形式


def linear_least_squares_error(data, model):    #误差函数，将测试点代入，计算误差值，用以判断是否在阈值内，是否是内群数据
    xi_tmp = data[:, 0:1]
    xi = np.hstack((xi_tmp ** 0, xi_tmp ** 1))  #同上，构造矩阵
    yi_real = data[:, 1:2]                      #获取点集的实际y值
    yi_fit = np.dot(xi, model)                  #代入模型，计算将x值输入模型后得到的y值
    error_per_point = np.sum((yi_real - yi_fit) ** 2, axis=1)   #计算所有点的误差

    return error_per_point


def ransac(data, n, k, t, d):
    """

    :param data: 数据集
    :param n: 抽样数量
    :param k: 循环次数
    :param t: 误差
    :param d: 阈值，即内群数据量满足此值后才认为成功
    :param return_all:
    :return:
    """

    iterations = 0
    best_err = np.inf   #np.inf为一个极大值，此处将best_err设置为极大值，使得后面第一次计算的误差即可对其进行替换
    best_model = None
    best_inlier_point_idxes = None
    while iterations < k:
        sample_idxes, test_idxes = random_sampling(n, data.shape[0])    #抽样
        sample_points = data[sample_idxes, :]                           #抽样点集
        test_points = data[test_idxes, :]                               #测试点集
        try_model = linear_least_square_model(sample_points)            #调用函数获取model参数
        test_errors = linear_least_squares_error(test_points, try_model)    #代入测试点
        test_points_also_inlier_idxes = test_idxes[test_errors < t]         #测试点也是内群数据时，获取此部分数据的索引值
        test_points_also_inlier = data[test_points_also_inlier_idxes, :]    #获取内群测试数据的值

        if len(test_points_also_inlier) > d:        #如果满足内群数据量大于阈值时，将内群测试数据与抽样数据合并，再次计算model参数，提高精度
            better_data = np.concatenate((sample_points, test_points_also_inlier))  #连接数据
            better_model = linear_least_square_model(better_data)                   #调用
            better_errors = linear_least_squares_error(better_data, better_model)   #获取误差值
            this_error = np.mean(better_errors)     #取误差均值
            if this_error < best_err:               #如果误差更小，以此次优化后的值作为model参数，进行下面的数据更新
                best_model = better_model
                best_err = this_error
                best_inlier_point_idxes = np.concatenate((sample_idxes, test_points_also_inlier_idxes))

        iterations += 1

    return best_model, {"inliers": best_inlier_point_idxes}


def test():
    sample_nums = 1000

    x_exact = np.linspace(-10, 10, sample_nums)        #-10~10之间均匀地生成3000个点，精确x值
    y_exact = 100 + 20 * x_exact                           #精确模型为y = 100 + 20x，精确y值

    x_noise = x_exact + np.random.normal(0, 0.5, size=sample_nums)  #加噪声
    y_noise = y_exact + np.random.normal(0, 0.5, size=sample_nums)  #加噪声

    outpoint_nums = 200     #加离群数据
    all_indexs = np.arange(x_noise.shape[0])    #获取索引，打乱，并取打乱后的前500个索引，重置为离群数据（并不一定全部是离群数据）
    np.random.shuffle(all_indexs)
    outpoint_idx = all_indexs[0:outpoint_nums]
    x_noise[outpoint_idx] = 50 * np.random.random(size=outpoint_nums)    #x加噪
    y_noise[outpoint_idx] = 50 * np.random.normal(0, 1, size=outpoint_nums)  #y加噪

    data = np.vstack((x_noise, y_noise)).T      #上面一直是行向量，此处进行将x、y连接后进行转置，变为3000x2的矩阵

    best_model, ransac_data = ransac(data, 50, 10000, 7e3, 400)        #调用ransac
    print("the best model is {},{}".format(best_model[0][0], best_model[1][0])) #打印ransac方法计算的model值

    A = np.vstack([x_noise ** 0, x_noise ** 1]).T
    sol, *_ = la.lstsq(A, y_noise)          #直接调用最小二乘法计算model的参数值
    print("linear_least_square result is {},{}".format(sol[0], sol[1]))
    y_fit = sol[0] + sol[1] * x_exact

test()
