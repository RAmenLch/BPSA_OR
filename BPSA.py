import numpy as np
import math
import shelve


class SA:
    @staticmethod
    def train(BP, W, samples,path,t0=30000, v=0.8):
        s = shelve.open(path)
        e = 10 ** -12
        t = t0
        while BP.E(samples,W) > e and t > 3*10**-4:
            for i in range(10):
                W = BP.train(samples, W)
                W_temp = []
                for w in W:
                    W_temp.append(w + (np.random.rand(w.shape[0],w.shape[1]) * 2 - 1))
                dc = float(BP.E(samples,W_temp) - BP.E(samples, W))
                pr = min(1, np.exp(-dc/t))
                if pr >= np.random.rand():
                    W = W_temp
                    zzz = BP.output(samples[-1][0], W)
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

class BP:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def output(input_p, W, n=-1):
        if n < -1:
            raise ValueError("n的值不能为负")
        if not isinstance(n,int):
            raise  ValueError("n必须为整数")
        if n > len(W) + 1:
            raise ValueError("n越界")
        y = input_p
        # 输入单元使用线性激励函数
        # 添加一个 -1(阈值单元)
        y = np.hstack((np.array([[-1]]), y))
        if n == 0:
            return y
        else:
            for i, w in enumerate(W):
                x = y*w
                y = BP.sigmoid(x)
                if (n == i+1 and n == len(W)) or i == len(W)-1:
                    return y
                elif n == i+1:
                    y = np.hstack((np.array([[-1]]), y))
                    return y
                else:
                    y = np.hstack((np.array([[-1]]), y))

    @staticmethod
    def E(samples, W):
        e = 0
        for i in samples:
            input_p = i[0]
            target = i[1]
            e += (BP.output(input_p,W) - BP.sigmoid(target))**2
        e *= 0.5
        return e

    @staticmethod
    def deltaW2(sample, W):
        input_p = sample[0]
        target = sample[1]
        y = BP.output(input_p, W)
        return (y - BP.sigmoid(target))*y*(1-y)

    @staticmethod
    def deltaW1(sample, W):
        input_p = sample[0]
        y = BP.output(input_p, W, 1)[:,1:]
        df = np.multiply(y, 1-y)
        dw2w1 = BP.deltaW2(sample, W) * W[1][1:,:].T
        return np.multiply(df, dw2w1)

    @staticmethod
    def train(samples, W, iters=3000):
        ERROR = 10**-7 # 允许误差
        LZ = 0.8 # 学习率
        MF = 0.6 # 动量因子
        Ww = [0,0]
        DW2 = np.mat([10])
        DW1 = np.mat([10])
        c = 0

        while (np.sum(np.multiply(DW1, DW1)) + np.sum(np.multiply(DW2, DW2)))**0.5 > ERROR and c <= iters and BP.E(samples, W) > ERROR:
            DW2 = 0
            DW1 = 0
            for s in samples:
                dw2 = BP.deltaW2(s, W)
                y1 = BP.output(s[0], W, 1)
                DW2 += (dw2.T * y1).T
                dw1 = BP.deltaW1(s,W)
                y0 = BP.output(s[0],W,0)
                DW1 += (dw1.T * y0).T
            DW1 = -LZ*DW1+MF*Ww[0]
            DW2 = -LZ*DW2+MF*Ww[1]
            Ww[0] = DW1
            Ww[1] = DW2
            W[0] = W[0] + DW1
            W[1] = W[1] + DW2
            c += 1
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
    W = SA.train(BP,W,samples)
    zzz = BP.output(test2[0], W)
    ddd = -math.log((1 / float(zzz)) - 1)
    print(ddd)







