import torch
import numpy as np
from .pairwise import pairwise_distance


def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def Kmeans(X, n_clusters, device=0, tol=1e-4):
    X = X.cuda(device)

    initial_state = forgy(X, n_clusters)

    cnt = 0
    cnt2 = 0
    while True:
        cnt += 1
        cnt2 += 1
        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)
            if selected.shape[0] > 1:
                initial_state[index] = selected.mean(dim=0)
            else:
                initial_state[index] = forgy(X, 1)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        if cnt > 50:
            initial_state = forgy(X, n_clusters)
            cnt = 0

        if center_shift ** 2 < tol:
            print('iter_cnt', cnt2)
            break
    # init state K*d
    return initial_state.cpu()




