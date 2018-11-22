import torch


def L2(a):
    # 一列是一个特征， 做L2归一化
    # 取出一行中最小元素, 变化到[0, ~]

    vector_len = a.shape[0]
    # m, _ = torch.min(a, 0)
    # m = m.repeat(vector_len, 1)
    # a = torch.add(a, -m)
    e = torch.mul(a, a)
    f = torch.sum(e, 0)
    # f = torch.sqrt(torch.sum(e, 0))
    g = f.repeat(vector_len, 1)
    a = torch.div(a, g)
    return a


def codeBook_L2(c, d, M):
    cnt = int(d / M)
    for i in range(M - 1):
        d = c[i * cnt:(i + 1) * cnt, :]
        r = L2(d)
        c[i * cnt:(i + 1) * cnt, :] = r

    d = c[(M - 1) * cnt:]
    r = L2(d)
    c[(M - 1) * cnt:] = r
    return c
