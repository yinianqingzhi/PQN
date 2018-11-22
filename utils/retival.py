import torch
from utils.L2_normalization import L2


def base_similarity(q, code_book, d, M, K):
    num = int(d/M)
    base_s = torch.zeros(M, K)
    for m in range(M-1):
        a = q[:, m*num:(m+1)*num].permute(1, 0)
        a = L2(a)
        a = a.permute(1, 0)
        base_s[m] = a.mm(code_book[m*num:(m+1)*num])
    a = q[:,  (M-1) * num:].permute(1, 0)
    a = L2(a)
    a = a.permute(1, 0)
    base_s[M-1] = a.mm(code_book[(M-1) * num:])
    return base_s


def code(x, code_book, d, M, K):
    num = int(d / M)
    s = torch.zeros(M, K)
    for m in range(M-1):
        x_l = x[:, m * num:(m + 1) * num].permute(1, 0)
        x_l = L2(x_l).permute(1, 0)
        s[m] = x_l.mm(code_book[m * num:(m + 1) * num])
    x_l = x[:, (M-1) * num:].permute(1, 0)
    x_l = L2(x_l).permute(1, 0)
    s[M-1] = x_l.mm(code_book[(M-1) * num:])
    bm = torch.argmax(s, 1)
    return bm



