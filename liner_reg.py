import numpy as np
from predictor import Predictor

class Liner_reg(Predictor):
    def __init__(self, layer_dims=(), max_train_step=10000, learning_rate=0.01, convergence = 1e-7):

        self.activate_method = 'None'
        # 输出函数方法
        self.output_method = 'None'
        # 损失函数方法
        self.loss_method = 'MSE'


