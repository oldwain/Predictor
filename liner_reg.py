import numpy as np
from predictor import Predictor

class Liner_reg(Predictor):
    def __init__(self, layer_dims=(), max_train_step=10000, learning_rate=0.01, convergence = 1e-7):
        # 对于线性回归，没有隐层， 所以layer_dims只有两个元素： 输入维度/输出维度， 并且输出维度必须为1
        assert len(layer_dims) == 2
        assert layer_dims[1] == 1

        super(Liner_reg,self).__init__(layer_dims=layer_dims, max_train_step=max_train_step, learning_rate=learning_rate, convergence=convergence)

        self.activate_method = 'None'
        # 输出函数方法
        self.output_method = 'None'
        # 损失函数方法
        self.loss_method = 'MSE'



    def forword(self, X):
        # input: X 训练集或测试集的输入向量
        # reuturn yhat: 预测值
        W = self.params['W1']
        b = self.params['b1']
        y_hat = np.dot(X, W.T) + b
        return y_hat

    def bp_grads(self, X, yhat, y):
        # input: X 训练集或测试集的输入向量
        # input: y:   训练集的实际标签
        #        yhat: forword计算的预测值
        # reuturn grads: 各个参数的梯度  dWi, dbi
        m = y.shape[0]
        dL_dW = 1 / m * np.dot((yhat - y).T,  X)
        dL_db = 1 / m * np.sum((yhat - y), axis=0, keepdims=True)  # 结果是个标量

        grads = {}
        grads['dW1'] = dL_dW
        grads['db1'] = dL_db
        return grads



# unittest
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.array([[-0.63542758, 0.34559502],
                  [-0.20503056, -1.16046526],
                  [-0.35527073, -0.71321306],
                  [1.23871104, 0.49376572],
                  [0.20178785, 1.15609095],
                  [-0.13825707, -0.54633791],
                  [-1.21319799, -0.300875],
                  [0.21378248, 0.72615012],
                  [0.00621571, 0.25791915],
                  [0.52754902, -0.17970102],
                  [0.46443869, 0.55787273],
                  [1.67456409, 1.36540635],
                  [-0.08245173, 0.39250068],
                  [0.95945694, 0.62946369],
                  [1.40428067, 1.5256541],
                  [1.09117097, -0.14351951],
                  [1.73697247, -0.75641368],
                  [-0.33071123, 0.85576374],
                  [-0.69277707, 1.83175546],
                  [0.59478283, 0.46830967],
                  [0.15638564, 1.06072622],
                  [0.76200255, 0.25710471],
                  [0.94453919, 0.89957668],
                  [1.08837827, -0.43832633],
                  [-0.44948331, 0.72710965],
                  [0.91578971, -0.46541413],
                  [-0.05823219, 1.54157258],
                  [0.65740993, 1.33332989],
                  [-0.50253328, -0.35507938],
                  [0.05884472, 1.69382138]])

    y = np.array([-11.32807172, -16.45177202, -15.73799327, -4.46945364,
                  -5.80275635, -14.06799542, -16.1684043, -7.71856378,
                  -10.0058796, -9.70348916, -6.25039231, 0.89860586,
                  -9.66965091, -4.48870652, 0.22240431, -8.19423593,
                  -7.52320663, -7.92824654, -5.61617497, -6.87038125,
                  -5.89237478, -7.29332994, -3.84015521, -8.98821176,
                  -9.26841644, -9.47128005, -3.93217039, -1.7746131,
                  -14.65242504, -4.37513263])

    model = Liner_reg(layer_dims=(2,1), max_train_step=200, learning_rate=0.1, convergence=1e-6)
    model.fix_w = True
    (params, info) = model.fit(X,y.reshape(-1,1), debug=False)
    print (model.params)

    costlist = info[0]
    plt.plot(costlist)
    plt.xlabel('times')
    plt.ylabel('cost')
    plt.show()
