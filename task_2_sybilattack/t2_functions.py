import torch
import numpy as np


def partition_ids(ids: list[int], main_bin_num=10, train=0.8, test=0.2):
    n = len(ids)
    bin_size = n // main_bin_num
    print(f"bin_size: {bin_size}")
    train = int(bin_size * train)
    test = int(bin_size * test)
    assert train + test == bin_size
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    binned = []
    for i in range(main_bin_num):
        one_bin = ids[indexes[i:i+bin_size]]
        tvt = np.split(one_bin, [train])
        print(f"train: {train}, test: {test}")
        binned.append(tvt)

    return binned

if __name__ == '__main__':
    ids  = np.arange(100, 200)
    partitioned = partition_ids(ids)
    print(partitioned[3])