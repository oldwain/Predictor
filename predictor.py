import numpy as np


class Predictor():
    def __init__(self, layer_dims=(), max_train_step=10000, learning_rate=0.01, convergence = 1e-7
                 , loss_method='CE', activate_method='relu', output_method='sigmoid'):

        # 激活函数方法
        self.activate_method = activate_method
        # 输出函数方法
        self.output_method = output_method
        # 损失函数方法
        self.loss_method = loss_method

        # 超参数
        self.max_train_step = max_train_step
        self.learning_rate = learning_rate
        self.convergence = convergence
        # 网络结构
        self.layer_dims = layer_dims
        # 模型参数
        self.params = {}

        # 中间结果缓存, 存放各个步骤计算出的Zi, Ai, 方便bp的时候使用
        self.cache = {}

    def init_params(self, fix_w=False):
        # 根据 layer_dims对参数进行初始化
        # fix_w: True: 用固定的w, 翻遍调试 False: 用随机数
        # reuturn yhat: 预测值

        # 可以写出通用的版本
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]
        n_layers = []
        params = {}
        n_layers[0] = n_x
        n_layers[len(self.layer_dims)] = n_y
        # W1 = np.random.random((n_1, n_x))
        # b1 = np.zeros((n_1, 1))
        # W2 = np.random.random((n_y, n_1))
        # b2 = np.zeros((n_y, 1))
        for i in self.layer_dims[1:]:
            n_layers[i] = self.layer_dims[i]
            params['W'+str(i)] =  np.random.random((n_layers[i], n_layers[i-1]))
            params['b'+str(i)] =  np.random.random((n_layers[i], 1))

        return params

    def forword(self, X):
        # input: X 训练集或测试集的输入向量
        # reuturn yhat: 预测值
        pass        # 子类实现

    def compute_cost(self, yhat, y):
        # input: y:   训练集的实际标签
        #        yhat: forword计算的预测值
        # return: cost: 代价
        m = y.shape[0]
        if self.loss_method == 'MSE':
            cost = 1 / (2 * m) * np.sum((y - yhat) ** 2)

        elif self.loss_method == 'CE':
            epsilon = 1e-20  # 加上一个小数，防止log(0)错误
            yhat_fix = np.minimum(np.maximum(yhat, epsilon), 1 - epsilon)

            cost = 1 / m * np.sum(- y * np.log(yhat_fix) - (1 - y) * np.log(1 - yhat_fix))
        else:
            raise Exception('不支持损失函数方法：{}'.format(self.loss_method))

        return cost

    def bp_grads(self, X, yhat, y):
        # input: X 训练集或测试集的输入向量
        # input: y:   训练集的实际标签
        #        yhat: forword计算的预测值
        # reuturn grads: 各个参数的梯度  dWi, dbi

        pass  # 子类实现

    def update_params(self, grads):
        # return params: 参数字典 params['W1']

        for i in self.layer_dims[1:]:
            self.params['W'+str(i)] -=  self.learning_rate * grads['W'+str(i)]

        return self.params

    def fit(self, X, y):

        # 训练过程， 这部分应该能在基类统一实现
        pass  # todo

    def predict_prob(self, X):
        return self.forword(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) > threshold).astype(int)
