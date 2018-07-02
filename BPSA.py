import numpy as np
import math
from  datetime import datetime
import shelve
from BP import BP


class MyBP(BP):
    @staticmethod
    def E(samples, W):
        e = 0
        for i in samples:
            input_p = i[0]
            target = i[1]
            e += (MyBP.output_y(input_p, W) - MyBP.sigmoid(target))**2
        e *= 0.5
        return e

    @staticmethod
    def deltaW2(sample, W):
        input_p = sample[0]
        target = sample[1]
        y = MyBP.output_y(input_p, W)
        return (y - BP.sigmoid(target))*y*(1-y)

    @staticmethod
    def deltaW1(sample, W):
        input_p = sample[0]
        #阈值单元不参与计算
        y = MyBP.output_y(input_p, W, 1)[:,1:]
        df = MyBP.differential_sigmoid_input_y(y)
        dw2w1 = MyBP.deltaW2(sample, W) * W[1][1:,:].T
        return np.multiply(df, dw2w1)

    # (样本集, 权值表, 最大迭代次数, 许可误差, 学习率, 动量因子)
    @staticmethod
    def train(samples, W, iterations=3000, error=10**-6, lz=0.8, mf=0.6):
        DW2 = np.mat([10])
        DW1 = np.mat([10])
        c = 0
        old_DW = [0, 0]
        # 当 DW(ΔW) 的元素平方和的平方根 小于许可误差 或 超过最大迭代次数 或 目标函数小于许可误差 停止迭代
        while (np.sum(np.multiply(DW1, DW1)) + np.sum(np.multiply(DW2, DW2)))**0.5 > error and c <= iterations and MyBP.E(samples, W) > error:
            DW2 = 0
            DW1 = 0
            for s in samples:
                dw2 = MyBP.deltaW2(s, W)
                y1 = MyBP.output_y(s[0], W, 1)
                DW2 += (dw2.T * y1).T
                dw1 = MyBP.deltaW1(s,W)
                y0 = MyBP.output_y(s[0],W,0)
                DW1 += (dw1.T * y0).T
            DW1 = -lz*DW1+ mf*old_DW[0]
            DW2 = -lz*DW2+ mf*old_DW[1]
            old_DW[0] = DW1
            old_DW[1] = DW2
            W[0] = W[0] + DW1
            W[1] = W[1] + DW2
            c += 1
        print(c)
        return W


class BPSA:
    # 状态生成函数
    @staticmethod
    def genete(W, inv=(-1,1)):
        W_temp = []
        for w in W:
            W_temp.append(w + (np.random.rand(w.shape[0], w.shape[1]) * (inv[1] - inv[0]) + inv[0]))
        return W_temp

    @staticmethod
    def train(samples, W, path='.', BP=MyBP, t0=30000, v=0.8, error=10**-8,**kwargs):
        if 'fname' in kwargs:
            fpath = path + '/W_of_' + kwargs['fname']
        else:
            fpath = path + '/W_of_'+ str(np.random.rand())
        print('开始模拟退火训练,训练数据保存至'+ fpath)
        t = t0
        W = MyBP.train(samples, W)
        best_W = W
        while MyBP.E(samples,W) > error and t > 3*error:
            # test
            print('t:', t, '\nE:', MyBP.E(samples,W))
            for i in range(10):
                W_temp = BPSA.genete(W,(-4,4))
                dc = float(BP.E(samples,W_temp) - BP.E(samples, W))
                pr = min(1, np.exp(-dc/t))
                if pr >= np.random.rand():
                    W = W_temp
                    W = MyBP.train(samples, W, iterations=1000 + min(3*10**4, int(3/(t*10))))

                    # 保存找到的最优解,如果传入tests,则也验证tests
                    if 'tests' in kwargs:
                        flag = MyBP.E(kwargs['tests'], W) < MyBP.E(kwargs['tests'],best_W)
                    else:
                        flag = True
                    if MyBP.E(samples,W) < MyBP.E(samples,best_W) and flag:
                        best_W = W

                    # test
                    y = BP.output_y(samples[-1][0], W)
                    try:
                        print(MyBP.inverse_sigmoid(y))
                    except ValueError as er:
                        print(er)
            t = v * t

        W = MyBP.train(samples, W, iterations=3*10**4)
        s = shelve.open(fpath)
        s['0'] = W
        s['best'] = best_W
        s.close()
        return W



if __name__ == '__main__':
    samples = [
        (np.mat([5.33, 5.39, 5.29, 5.41, 5.45]), 5.50),
        (np.mat([5.39, 5.29, 5.41, 5.45, 5.50]), 5.57),
        (np.mat([5.29, 5.41, 5.45, 5.50, 5.57]), 5.58),
        (np.mat([5.41, 5.45, 5.50, 5.57, 5.58]), 5.61),
        (np.mat([5.45, 5.50, 5.57, 5.58, 5.61]), 5.69),
        (np.mat([5.50, 5.57, 5.58, 5.61, 5.69]), 5.78),
        (np.mat([5.57, 5.58, 5.61, 5.69, 5.78]), 5.78),
        (np.mat([5.58, 5.61, 5.69, 5.78, 5.78]), 5.81),
        (np.mat([5.61, 5.69, 5.78, 5.78, 5.81]), 5.86),
        (np.mat([5.69, 5.78, 5.78, 5.81, 5.86]), 5.90),
        (np.mat([5.78, 5.78, 5.81, 5.86, 5.90]), 5.97),
        (np.mat([5.78, 5.81, 5.86, 5.90, 5.97]), 6.49),
        (np.mat([5.81, 5.86, 5.90, 5.97, 6.49]), 6.60),
        (np.mat([5.86, 5.90, 5.97, 6.49, 6.60]), 6.64)
    ]
    test2 = samples[-1]
    test = np.mat([6.49,6.60,6.64,6.74,6.87])
    w1 = np.random.rand(5, 8)*6-3
    theta1 = np.random.rand(1, 8)*6-3
    W1 = np.asmatrix(np.vstack([theta1,w1]))
    w2 = np.random.rand(8, 1)*6-3
    theta2 = np.random.rand(1, 1)*6-3
    W2 = np.asmatrix(np.vstack([theta2, w2]))
    W = [W1, W2]

    BPSA.train(samples, W)







