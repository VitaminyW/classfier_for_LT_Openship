import random
import numpy as np
from torch.utils.data.sampler import Sampler
import pdb


class CycleIterWithShuffle: # 用于确保每class_num次抽样，都能将所有的类别抽，

    def __init__(self, data):
        self.data_list = list(data) # [0,1,2,3,4]
        self.length = len(self.data_list) # 5
        self.i = self.length - 1 # 4

    def __iter__(self):
        return self

    def __next__(self): # 当要获取数据时，next方法会被调用
        self.i += 1 # 5
        if self.i == self.length: # 当整个列表被遍历后，打乱，将索引恢复到0
            self.i = 0
            random.shuffle(self.data_list) # [3,4,1,0,2] - > [4,2,3,0,1]
        return self.data_list[self.i]


def class_balance_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0  # 现在已经抽过多少数据了
    j = 0  # 用来一次抽某个类别的多少个数据
    while i < n:  # 是否已经抽够了max(classei_example) * class_number个数据
        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassBalanceSampler(Sampler):
    # 先决定抽取哪个类别的数据 -> 再抽取该类别的某个数据 [[1,4,5,100],[9,7]]
    def __init__(self, data_source, num_samples_cls=1):
        num_classes = len(np.unique(data_source.labels)) # 获取有几种类别
        self.class_iter = CycleIterWithShuffle(range(num_classes)) # 类别迭代器：均匀地采样类别
        cls_data_list = [[] for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [CycleIterWithShuffle(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list) # max(classei_example) * class_number
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_balance_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples