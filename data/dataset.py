import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
from data.CIFAR10_triple import CIFAR10_triple

torch.cuda.set_device(1)


def init_loader(root, batchsize, DOWNLOAD_CIFAR10=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    # data transform
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # train data set
    train_set = tv.datasets.CIFAR10(root=root,
                                    train=True,
                                    download=DOWNLOAD_CIFAR10,
                                    transform=transform_train)

    init_data_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   pin_memory=True)

    return init_data_loader


def train_dataloader(root, batchsize, class_num, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    # data transform
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    aa = CIFAR10_triple(root, class_num, 5000, transform=transform_train)

    triple_data_loader = torch.utils.data.DataLoader(aa,
                                                     batch_size=batchsize,
                                                     shuffle=True,
                                                     num_workers=2,
                                                     pin_memory=True)
    return triple_data_loader


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        d = self.data_tensor[index]
        d = Image.fromarray(d)
        if self.transform:
            d = self.transform(d)
        return d, self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, quary_data, pos_data, neg_data, transform=None):
        self.quary_data = quary_data
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.transform = transform

    def __getitem__(self, index):
        q_img = self.quary_data[index]
        p_img = self.pos_data[index]
        n_img = self.neg_data[index]
        if self.transform:
            q_img = self.transform(q_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        return q_img, p_img, n_img

    def __len__(self):
        return self.quary_data.size(0)


if __name__ == '__main__':
    show = ToPILImage()
    root = '/home/jinyue/PycharmProjects/PQN/data/'
    batchsize = 4
    DOWNLOAD_CIFAR10 = True
    # if not (os.path.exists('./cifar10/'))is None or not os.listdir('./cifar10/') is None:
    if (os.path.exists('./cifar-10-batches-py/')) is True:
        if (os.listdir('./cifar-10-batches-py/')) is not None:
            # not cifar10 dir or mnist is empyt dir
            DOWNLOAD_CIFAR10 = False
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    init_data_tensor = init_loader(root, DOWNLOAD_CIFAR10, normalize)
    triple_data_loader = train_dataloader(root, batchsize, DOWNLOAD_CIFAR10, normalize)
    test_quary_loader, test_base_loader, classes = test_dataloader(root, batchsize, DOWNLOAD_CIFAR10, normalize)
