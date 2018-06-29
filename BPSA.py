import numpy as np
import math
import shelve
from BP import BP


class MySA:
    @staticmethod
    def train(BP, W, samples,path,t0=30000, v=0.8):
        s = shelve.open(path)
        e = 10 ** -12
        t = t0
        while BP.E(samples,W) > e and t > 3*10**-4:
            for i in range(10):
                W = MyBP.train(samples, W)
                W_temp = []
                for w in W:
                    W_temp.append(w + (np.random.rand(w.shape[0],w.shape[1]) * 2 - 1))
                dc = float(BP.E(samples,W_temp) - BP.E(samples, W))
                pr = min(1, np.exp(-dc/t))
                if pr >= np.random.rand():
                    W = W_temp
                    zzz = BP.output_y(samples[-1][0], W)
                    try:
                        ddd = -math.log((1 / float(zzz)) - 1)
                    except ValueError as er:
                        print(er)
                    else:
                        print(ddd)
            s[str(t)] = W
            t = v * t
        s.close()
        return W


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
            print(MyBP.E(samples, W))
        print(c)
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

    W = MyBP.train(samples,W, iterations=5000)

    zzz = BP.output_y(test2[0], W)
    ddd = -math.log((1 / float(zzz)) - 1)
    print(ddd)







