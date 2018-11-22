import warnings
import torch
import torchvision.transforms as transforms


class DefaultConfig(object):
    env = 'default'  # visdom 环境

    vis_port = 8097  # visdom 端口
    model = 'ResNet34'

    data_root = '/home/dlc/jinyue/PQN/data/'
    # data_root = '/home/jinyue/PycharmProjects/PQN/data/'
    load_model_path = None
    download_data = True
    # save_model_root = '/home/jinyue/PycharmProjects/PQN/checkpoint/'
    save_model_root = '/home/dlc/jinyue/PQN/checkpoint/'
    fearure_net_pth = 'models/net_030.pth'

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    init_iter_num = 1000
    init_batchsize = 25

    batch_size = 32
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20

    max_epoch = 10
    lr = 0.000001  # initial learning rate
    lr_decay = 0.9  # when val_loss increase, lr = lr*lr_decay

    weight_decay = 1e-4  # 损失函数

    alpha = 8
    d = 500
    M = 8
    B = 4
    K = 32

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()









