"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        # self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp', 'rich']
        # self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5, 'rich': 6}
        # self.dataset_list = ['lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp', 'rich']
        # self.dataset_dict = {'lsp-orig': 0, 'mpii': 1, 'lspet': 2, 'coco': 3, 'mpi-inf-3dhp': 4, 'rich': 5}
        # self.dataset_list = ['h36m']
        # self.dataset_dict = {'h36m': 0}
        # self.dataset_list = ['rich']
        # self.dataset_dict = {'rich': 0}
        # self.dataset_list = ['h36m', 'rich']
        # self.dataset_dict = {'h36m': 0, 'rich': 1}
        self.dataset_list = ['mpi-inf-3dhp', 'rich']
        self.dataset_dict = {'mpi-inf-3dhp': 0, 'rich': 1}
        # self.dataset_list = ['agora1', 'agora2', 'agora3', 'agora4']
        # self.dataset_dict = {'agora1': 0, 'agora2': 1, 'agora3': 2, 'agora4': 3, 'agora5':4}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets])
        # length_itw = sum([len(ds) for ds in self.datasets[1:-2]]) # only when training with 2d datasets
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        # 30% H36M - 60% ITW - 10% MPI-INF
        30% H36M - 40% ITW - 10% MPI-INF - 20% RICH
        """
        # self.partition = [0.3, .4*len(self.datasets[1])/length_itw,
        #                   .4*len(self.datasets[2])/length_itw,
        #                   .4*len(self.datasets[3])/length_itw,
        #                   .4*len(self.datasets[4])/length_itw,
        #                   0.1, 0.2]
        # self.partition = [.4*len(self.datasets[0])/length_itw,
        #                   .4*len(self.datasets[1])/length_itw,
        #                   .4*len(self.datasets[2])/length_itw,
        #                   .4*len(self.datasets[3])/length_itw,
        #                   0.2, 0.4]
        # self.partition = [1.0]
        self.partition = [0.7, 0.3]
        # self.partition = [0.3, .4*len(self.datasets[1])/length_itw,
        #                   .4*len(self.datasets[2])/length_itw,
        #                   .4*len(self.datasets[3])/length_itw,
        #                   .4*len(self.datasets[4])/length_itw,
        #                   0.1, 0.2]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        # for i in range(6):
        for i in range(2):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
