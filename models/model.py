import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import random
from models.resnet import ResNet18
from config import opt
from torch.autograd import Variable
from utils.L2_normalization import L2


class PQN(nn.Module):
    def __init__(self, init_c, alpha, d, M, K, batch_size):
        super(PQN, self).__init__()
        # self.branches = nn.ModuleList(nn.Linear(d, 1) for _ in range(K))
        self.K = K
        self.d = d
        self.M = M
        self.alpha = alpha
        self.batch_size = batch_size
        if init_c is None:
            init_c = torch.zeros((d, K))
        self.c = nn.Parameter(torch.Tensor(init_c))

    def forward(self, x):
        # inner_product = torch.Tensor(self.batch_size, self.M, self.K).to(opt.device)
        # for i in range(0, self.K):
        #     # d = torch.diag(self.c[:, i])
        #     # e = torch.mm(x, d)
        #     # f = torch.mm(e, self.W)
        #     # self.inner_product[:, :, i] = 2 * self.alpha * f
        #     inner_product[:, :, i] = 2 * self.alpha * torch.mm(torch.mm(x, torch.diag(self.c[:, i])), self.W)
        #
        # A = F.softmax(inner_product, 2)
        # result = torch.Tensor(self.batch_size, self.d).to(opt.device)
        # for i in range(0, self.batch_size):
        #     # a = A[i]
        #     # g = torch.mm(self.W, a)
        #     # h = torch.mul(g, self.c)
        #     # result[i] = torch.sum(h, 1)
        #     result[i] = torch.sum(torch.mul(torch.mm(self.W, A[i]), self.c), 1)

        bz = x.shape[0]
        result = torch.FloatTensor(bz, self.d).to(opt.device)
        vector_len = int(self.d / self.M)
        for i in range(0, self.M-1):
            a = x[:, i*vector_len:(i+1)*vector_len].permute(1, 0)
            a = L2(a).permute(1, 0)

            # m, _ = torch.min(a, 1)
            # m = m.repeat(vector_len, 1).transpose(1, 0)
            # a = torch.add(a, -m)
            # e = torch.mul(a, a)
            # f = torch.sum(e, 1)
            # g = f.repeat(vector_len, 1).transpose(1, 0)
            # a = torch.div(a, g)

            b = self.c[i*vector_len:(i+1)*vector_len, :]
            c = 2 * self.alpha * torch.mm(a, b)
            d = F.softmax(c, 1)
            result[:, i*vector_len:(i+1)*vector_len] = torch.mm(d, b.transpose(1, 0))

        a = x[:, (self.M-1) * vector_len:].permute(1, 0)
        a = L2(a).permute(1, 0)
        # l = self.d - (self.M-1) * vector_len
        # m, _ = torch.min(a, 1)
        # m = m.repeat(l, 1).transpose(1, 0)
        # a = torch.add(a, -m)
        # e = torch.mul(a, a)
        # f = torch.sum(e, 1)
        # g = f.repeat(l, 1).transpose(1, 0)
        # a = torch.div(a, g)

        b = self.c[(self.M-1) * vector_len:, :]
        c = 2 * self.alpha * torch.mm(a, b)
        d = F.softmax(c, 1)
        result[:, (self.M-1) * vector_len:] = torch.mm(d, b.transpose(1, 0))
        return result


class CNN(nn.Module):
    def __init__(self, d, pre_model_pth):
        super(CNN, self).__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=1000)
        self.resnet.load_state_dict(torch.load(pre_model_pth))
        for i, p in enumerate(self.resnet.parameters()):
            p.requires_grad = False
        # self.fc = nn.Linear(1000, d)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.fc(x)
        return x


class Total_Model(nn.Module):
    def __init__(self, init_c, alpha, d, M, K, pre_model_pth, batch_size):
        super(Total_Model, self).__init__()
        self.resnet = ResNet18()
        self.resnet.load_state_dict(torch.load(pre_model_pth), strict=False)
        self.pqn = PQN(init_c, alpha, d, M, K, batch_size)

    def forward_x(self, x):
        x = self.resnet(x)
        return x

    def forward_pn(self, x):
        x = self.resnet(x)
        x = self.pqn(x)
        return x

    def forward(self, x, p, q, train=True):
        x = self.forward_x(x)
        p = self.forward_pn(p)
        q = self.forward_pn(q)
        return x, p, q


class AsymmetricTripleLoss(nn.Module):
    def __init__(self):
        super(AsymmetricTripleLoss, self).__init__()

    def forward(self, x, s_p, s_n):
        temp = torch.sum(x.mul(s_n)) - torch.sum(x.mul(s_p))
        loss = torch.sigmoid(temp)
        return loss


if __name__ == '__main__':
    pass






