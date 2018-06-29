import numpy as np
import math
import shelve

class BP:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def inverse_sigmoid(y):
        temp = 1/float(y)-1
        x = -math.log(temp)
        return x

    @staticmethod
    def differential_sigmoid_input_y(y):
        return np.multiply(y, 1 - y)

    @staticmethod
    def differential_sigmoid_input_x(x):
        y = BP.sigmoid(x)
        return BP.differential_sigmoid_input_y(y)

    @staticmethod
    def output_y(input_p, W, n=-1):
        if n < -1:
            raise ValueError("n仅支持-1索引")
        if not isinstance(n,int):
            raise ValueError("n必须为整数")
        if n > len(W) + 1:
            raise ValueError("n越界")
        # 输入单元使用线性激励函数
        y = input_p
        # 添加一个 -1(阈值单元)
        y = np.hstack((np.array([[-1]]), y))
        if n == 0:
            return y
        else:
            for i, w in enumerate(W):
                x = y*w
                y = BP.sigmoid(x)
                if (n == i+1 and n == len(W)) or i == len(W)-1:
                    # 输出层没有阈值单元
                    return y
                elif n == i+1:
                    y = np.hstack((np.array([[-1]]), y))
                    return y
                else:
                    y = np.hstack((np.array([[-1]]), y))

    @staticmethod
    def E(**kwargs):
        raise NotImplementedError("没有实现目标函数E")

    @staticmethod
    def train(**kwargs):
        raise NotImplementedError("没有实现训练函数train")