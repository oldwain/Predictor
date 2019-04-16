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

        # 其它选项
        self.fix_w = False  # 初始参数的w是否随机

    def init_params(self, fix_w=False):
        # 根据 layer_dims对参数进行初始化
        # fix_w: True: 用固定的w, 翻遍调试 False: 用随机数
        # reuturn yhat: 预测值

        # 可以写出通用的版本
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]
        n_layers = []
        params = {}
        #n_layers[0] = n_x
        #n_layers[len(self.layer_dims)] = n_y
        # W1 = np.random.random((n_1, n_x))
        # b1 = np.zeros((n_1, 1))
        # W2 = np.random.random((n_y, n_1))
        # b2 = np.zeros((n_y, 1))

        for i in range(1, len(self.layer_dims)):
            #n_layers[i] = self.layer_dims[i]
            if fix_w:
                params['W' + str(i)] = np.ones((self.layer_dims[i], self.layer_dims[i - 1])) * 10
                params['b' + str(i)] = np.zeros((self.layer_dims[i], 1))
            else:
                params['W'+str(i)] =  np.random.random((self.layer_dims[i], self.layer_dims[i-1]))
                params['b'+str(i)] =  np.random.random((self.layer_dims[i], 1))

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
            self.params['W'+str(i)] -=  self.learning_rate * grads['dW'+str(i)]
            self.params['b'+str(i)] -=  self.learning_rate * grads['db'+str(i)]

        return self.params

    def fit(self, X, y, debug=False):
        self.params = self.init_params(fix_w=self.fix_w)
        if debug:
            print("初始params: {}".format(self.params))

        costlist = []
        paramslist = []
        last_L = 1e5

        for j in range(self.max_train_step):

            yhat = self.forword(X)
            # print ("yhat.shape:{}, y.shape: {}".format(yhat.shape, y.shape))
            L = self.compute_cost(yhat, y)

            if abs(last_L - L) < self.convergence:
                if debug:
                    print("已收敛")
                    print("step:{} cost:{} ".format(j, L))
                    print("params: {}".format(self.params))
                    print()
                break
            last_L = L


            costlist.append(L)
            paramslist.append(self.params)

            if debug:
                if j % 1000 == 0 or j==0 or j==1:
                    print("step:{} cost:{} ".format(j, L))
                    # print("dW1: {} dW2: {} db: {}".format(dL_dW[0,0], dL_dW[0,1], dL_db))
                    # print("params: {}".format(params))
                    print()
                if j == 1:
                    print("grads: {}".format(grads))
                    print("params: {}".format(self.params))
            grads = self.bp_grads(X, yhat, y)

            self.params = self.update_params(grads)

            # print (W.shape)
            info = (costlist, paramslist)
        else:
            if debug:
                print("已达到最大迭代次数")
                print("cost:{} ".format(L))
                print("params: {}".format(self.params))
                print()

        return (self.params, info)

    def predict_prob(self, X):
        return self.forword(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) > threshold).astype(int)
