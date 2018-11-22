from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import random


class CIFAR10_triple(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, K, each_quary_num,
                 transform=None,
                 ):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.train_data = []
        self.train_labels = []
        self.K = K
        self.each_quary_num = each_quary_num

        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        self.train_labels = np.array(self.train_labels)
        self.train_data = np.concatenate(self.train_data)
        self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.quary_index = np.zeros((self.K*self.each_quary_num, 1), np.int)
        self.pos_index = np.zeros((self.K*self.each_quary_num, 1), np.int)
        self.neg_index = np.zeros((self.K*self.each_quary_num, 1), np.int)

        each_K_num = int(np.shape(self.train_labels)[0] / K)
        ind_quary = np.zeros((each_K_num, K), dtype=np.int)
        class_num = np.zeros((K, 1), dtype=np.int)
        for k in range(0, self.K):
            ind = np.where(self.train_labels == k)[0]
            if len(ind) > self.each_quary_num:
                ind_quary[:, k] = ind[:each_K_num]
                class_num[k] = each_K_num
            else:
                ind_quary[:len(ind), k] = ind[:]
                class_num[k] = len(ind)

        for i in range(0, self.K*self.each_quary_num):
            q_c_ind = random.randint(0, K - 1)
            q_d_ind = random.randint(0, class_num[q_c_ind] - 1)
            self.quary_index[i] = q_d_ind

            p_d_ind = random.randint(0, class_num[q_c_ind] - 1)
            while p_d_ind == q_d_ind:
                p_d_ind = random.randint(0, class_num[q_c_ind] - 1)
            self.pos_index[i] = p_d_ind

            n_c_ind = random.randint(0, K - 1)
            while n_c_ind == q_c_ind:
                n_c_ind = random.randint(0, K - 1)
            n_d_ind = random.randint(0, class_num[n_c_ind] - 1)
            self.neg_index[i] = n_d_ind

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        q_img = self.train_data[self.quary_index[index]]
        shape = q_img.shape[1:]
        q_img = np.reshape(q_img, shape)
        p_img = self.train_data[self.pos_index[index]]
        p_img = np.reshape(p_img, shape)
        n_img = self.train_data[self.neg_index[index]]
        n_img = np.reshape(n_img, shape)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        q_img = Image.fromarray(q_img)
        p_img = Image.fromarray(p_img)
        n_img = Image.fromarray(n_img)

        if self.transform is not None:
            q_img = self.transform(q_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)
        return q_img, p_img, n_img

    def __len__(self):
        return np.shape(self.quary_index)[0]


