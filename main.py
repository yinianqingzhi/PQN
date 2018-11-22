import torch
import os
from models.model import Total_Model, AsymmetricTripleLoss
from data.dataset import init_loader, train_dataloader
from config import opt
from utils.kmeans import Kmeans
import torchvision.transforms as transforms
from torchnet import meter
from utils.visualize import Visualizer
from utils.mAP import mAP
import time
from models.resnet import ResNet18
from utils.L2_normalization import codeBook_L2
from data.CIFAR10_test_base import CIFAR10_test_base
from data.CIFAR10_test_query import CIFAR10_test_query

torch.cuda.set_device(1)


def train(**kwargs):
    opt._parse(kwargs)
    # vis = Visualizer(opt.env, port=opt.vis_port)

    # data
    root = opt.data_root
    batchsize = opt.batch_size
    download_cifar10 = opt.download_data
    data_root = os.path.join(root, 'cifar-10-batches-py/')
    if (os.path.exists(data_root)) is True:
        if (os.listdir(data_root)) is not None:
            # not cifar10 dir or mnist is empyt dir
            download_cifar10 = False

    # prepare triple data for training  and the test data to calculate the accuracy
    triple_data_loader = train_dataloader(root, batchsize, 10, opt.normalize)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        opt.normalize
    ])

    testset_query = CIFAR10_test_query(root=root, train=False, transform=transform_test, query_num=100)
    base_index = testset_query.base_index
    testset_base = CIFAR10_test_base(root='./data', train=False, transform=transform_test, base_index=base_index)

    testloader_query = torch.utils.data.DataLoader(testset_query, batch_size=1, shuffle=False, num_workers=2)
    testloader_base = torch.utils.data.DataLoader(testset_base, batch_size=1, shuffle=False, num_workers=2)

    # init PQN
    feature_net = ResNet18()
    feature_net.load_state_dict(torch.load(opt.fearure_net_pth), strict=False)
    feature_net.to(opt.device)

    # prepare data for initial the code book
    init_data_loader = init_loader(root, opt.init_batchsize, DOWNLOAD_CIFAR10=download_cifar10)
    init_c = torch.zeros((opt.d, opt.K)).float()

    kmeans_data = torch.Tensor(opt.init_iter_num * opt.init_batchsize, opt.d)
    for ii, (init_data, _) in enumerate(init_data_loader):
        if ii == opt.init_iter_num:
            break
        input = init_data.to(opt.device)
        output = feature_net(input)
        kmeans_data[ii * opt.init_batchsize:(ii+1) * opt.init_batchsize] = output.data.cpu()

    # use the K-means method to initial the code book
    cnt = int(opt.d/opt.M)
    for i in range(opt.M-1):
        d = kmeans_data[:, i * cnt:(i+1)*cnt]
        r = Kmeans(d, opt.K, device=0, tol=1e-3)
        init_c[i * cnt:(i+1)*cnt] = r.permute(1, 0)

    d = kmeans_data[:, (opt.M-1) * cnt:]
    r = Kmeans(d, opt.K, device=0, tol=1e-3)
    init_c[(opt.M-1) * cnt:] = r.permute(1, 0)

    # init codebook opt.d*K
    init_c = codeBook_L2(init_c, opt.d, opt.M)

    # init model
    model = Total_Model(init_c, opt.alpha, opt.d, opt.M, opt.K, opt.fearure_net_pth, opt.batch_size)
    model.to(opt.device)
    feature_net = model.resnet

    init_c = init_c.to(opt.device)
    m, mRecall = mAP(feature_net, testloader_query, testloader_base, init_c)
    print('mAP', m, 'mRecall', mRecall)

    # init loss
    ALT_loss = AsymmetricTripleLoss()

    # set optimizer
    lr = opt.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e100
    model.train()

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        print('epoch', epoch)

        for ii, (q_input, p_input, n_input) in enumerate(triple_data_loader):

            q_input = q_input.to(opt.device)
            p_input = p_input.to(opt.device)
            n_input = n_input.to(opt.device)

            optimizer.zero_grad()
            q_output, p_output, n_output = model(q_input, p_input, n_input)
            loss = ALT_loss(q_output, p_output, n_output)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            # L2 normalize the code book
            s = model.state_dict()['pqn.c']
            s = codeBook_L2(s, opt.d, opt.M)
            model.state_dict()['pqn.c'] = s

            # print loss
            if (ii + 1) % opt.print_freq == 0:
                # vis.plot('loss', loss_meter.value()[0])
                print('iter', ii, 'loss', loss_meter.value()[0])
            else:
                pass

            if (ii + 1) % 250 == 0:
                print('calculate the mAP ... ')
                model.eval()
                feature_net = model.resnet
                code_book = s
                m, mRecall = mAP(feature_net, testloader_query, testloader_base, code_book)
                print('mAP', m, 'mRecall', mRecall)
            else:
                pass

        # save model
        now = int(time.time())
        timeStruct = time.localtime(now)
        strTime = time.strftime("%Y-%m-%d_%H:%M:%S", timeStruct)
        save_path = os.path.join(opt.save_model_root, 'model_parameter_{}'.format(epoch)+strTime+'.pth')
        torch.save(model.state_dict(), save_path)

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            print('adjust the learning rate')
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def test(total_model=None):
    if total_model is None:
        model = Total_Model(None, opt.alpha, opt.d, opt.M, opt.K)
        if opt.load_model_path:
            model.load(opt.load_model_path)
        model.to(opt.device)
    else:
        model = total_model
    model.eval()
    params = model.state_dict()
    code_book = params['pqn.c']
    feature_net = model.resnet

    # code_book = random.random(opt.d, opt.K)
    # 取参数作为code book
    root = opt.data_root
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testset_query = CIFAR10_test_query(root=root, train=False, transform=transform_test, query_num=100)
    base_index = testset_query.base_index
    testset_base = CIFAR10_test_base(root=root, train=False, transform=transform_test, base_index=base_index)

    testloader_query = torch.utils.data.DataLoader(testset_query, batch_size=1, shuffle=False, num_workers=2)
    testloader_base = torch.utils.data.DataLoader(testset_base, batch_size=1, shuffle=False, num_workers=2)

    # test_quary_loader, test_base_loader, classes = test_dataloader(root, batchsize, download_cifar10, normalize)
    MAP = mAP(feature_net, testloader_query, testloader_base, code_book)
    return MAP


if __name__ == '__main__':
    # import fire
    train()
    # fire.Fire()




































