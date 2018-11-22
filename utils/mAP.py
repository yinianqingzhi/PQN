import numpy as np
from utils.retival import base_similarity, code
from config import opt
import torchvision.transforms as transforms
import torch
import torchvision
from utils.L2_normalization import L2


def mAP(model, testloader_query, testloader_base, code_book):

    b_l = len(testloader_base)
    b = np.zeros((b_l, opt.M), dtype=np.int)

    print('calculate b ...')
    base_class = np.zeros((b_l,))
    for ii, (base_data, b_class) in enumerate(testloader_base):
        base_class[ii] = b_class.data.numpy()
        base_data = base_data.to(opt.device)
        output = model(base_data)
        bb = code(output, code_book, opt.d, opt.M, opt.K)
        b[ii] = bb.data.numpy()

    l = len(testloader_query)
    result = np.zeros((l, 1))
    target = np.zeros((l, 1))
    AP = np.zeros((l, 1))
    Recall = np.zeros((l, 1))

    print('calculate similarity ... ')
    for ii, (quary_data, quary_class) in enumerate(testloader_query):
        target[ii] = quary_class
        quary_data = quary_data.cuda()
        q = model(quary_data)
        base_s = base_similarity(q, code_book, opt.d, opt.M, opt.K)
        # 根据上面b 计算每个base_data的similarity 选最大的
        s = np.zeros((b_l,))
        for j in range(b_l):
            for m in range(opt.M):
                s[j] += base_s[m, b[j, m]].data.numpy()
        rank = np.argsort(-s)
        result[ii] = base_class[rank[0]]
        similar_threshold = s[rank[0]] * 0.95

        cnt = 0
        cnt_right = 0
        class_num = len(np.where(base_class == quary_class[0])[0])
        ind = 0
        while s[rank[ind]] >= similar_threshold:
            cnt += 1
            if base_class[rank[ind]] == quary_class[0]:
                cnt_right += 1
            ind += 1
            if ind == b_l:
                break
        print('max_similar', s[rank[0]], 'cnt_right', cnt_right, 'cnt', cnt)
        AP[ii] = cnt_right / cnt
        Recall[ii] = cnt_right / class_num


        # for ind in range(b_l):
        #     if base_class[rank[ind]] == quary_class[0]:
        #         cnt += 1
        #         precision += cnt / (ind + 1)
        # AP[ii] = precision / cnt
        # print('AP', ii, ':', AP[ii])
    m = np.sum(AP) / l
    mRecall = np.sum(Recall) / l
    return m, mRecall

