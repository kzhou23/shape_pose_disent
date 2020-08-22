from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import torch
import torch.utils.data

class AMASS_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, flag='train', transform=None):
        file_path = os.path.join(data_path, flag+'.npy')
        self.data = np.load(file_path)
        self.transform = transform

        self.subj_ind = {}
        for i, row in enumerate(self.data):
            if row['f0'] not in self.subj_ind:
                self.subj_ind[row['f0']] = [i]
            else:
                self.subj_ind[row['f0']].append(i)
        self.num_subj = len(self.subj_ind)

        self.samp_ind = {}
        self.resamp_ind()
        
        print('Dataset:', flag)
        print('Number of subjects:', self.num_subj)
        print('Number of meshes:', len(self.data))

    def resamp_ind(self):
        for idx1 in range(len(self)):
            for i in self.subj_ind.keys():
                if idx1 in self.subj_ind[i]:
                    subj_idx1 = i
                    idx2 = np.random.choice(self.subj_ind[i], 1)[0]
                    while idx2 == idx1:
                        idx2 = np.random.choice(self.subj_ind[i], 1)[0]
                    break
            subj_idx2 = np.random.choice(list(set(self.subj_ind.keys()).difference(set((subj_idx1,)))), 1)[0]
            idx3 = np.random.choice(self.subj_ind[subj_idx2], 1)[0]  
            self.samp_ind[idx1] = (idx2, idx3)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        idx1 = index
        idx2, idx3 = self.samp_ind[idx1]

        mesh1 = self.data[idx1]['f4'].reshape(6890, 3)
        mesh2 = self.data[idx2]['f4'].reshape(6890, 3)
        mesh3 = self.data[idx3]['f4'].reshape(6890, 3)

        mesh1 = mesh1 - np.mean(mesh1, axis=0, keepdims=True)
        mesh2 = mesh2 - np.mean(mesh2, axis=0, keepdims=True)
        mesh3 = mesh3 - np.mean(mesh3, axis=0, keepdims=True)

        return mesh1, mesh2, mesh3

    def __len__(self):
        return len(self.data)


class AMASS_Dataset_Eval(torch.utils.data.Dataset):
    def __init__(self, data_path, flag='vald'):
        file_path = os.path.join(data_path, '{}.npy'.format(flag))
        self.data = np.load(file_path)

        self.subj_ind = {}
        for i, row in enumerate(self.data):
            if row['f0'] not in self.subj_ind:
                self.subj_ind[row['f0']] = [i]
            else:
                self.subj_ind[row['f0']].append(i)
        self.num_subj = len(self.subj_ind)

        self.samp_ind = {}
        self.resamp_ind()

    def resamp_ind(self):
        for idx1 in range(len(self)):
            for i in self.subj_ind.keys():
                if idx1 in self.subj_ind[i]:
                    subj_idx1 = i
                    break
            subj_idx2 = np.random.choice(list(set(self.subj_ind.keys()).difference(set((subj_idx1,)))), 1)[0]
            idx2 = np.random.choice(self.subj_ind[subj_idx2], 1)[0]  
            self.samp_ind[idx1] = idx2

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        idx1 = index
        idx2 = self.samp_ind[idx1]

        mesh1 = self.data[idx1]['f4'].reshape(6890, 3)
        mesh2 = self.data[idx2]['f4'].reshape(6890, 3)

        mesh1 = mesh1 - np.mean(mesh1, axis=0, keepdims=True)
        mesh2 = mesh2 - np.mean(mesh2, axis=0, keepdims=True)

        gdr1 = self.data[idx1]['f1']
        gdr2 = self.data[idx2]['f1']
        betas1 = self.data[idx1]['f2']
        betas2 = self.data[idx2]['f2']
        pose1 = self.data[idx1]['f3']
        pose2 = self.data[idx2]['f3']

        smpl_params3 = dict(zip(['gender', 'shape', 'pose'], [gdr1, betas1, pose2]))
        smpl_params4 = dict(zip(['gender', 'shape', 'pose'], [gdr2, betas2, pose1]))

        return mesh1, mesh2, smpl_params3, smpl_params4

    def __len__(self):
        return len(self.data)


def train_collate(batch):
    meshes = torch.tensor(np.stack([data for data in batch]), dtype=torch.float32)
    return meshes


class AMASS_Eval_Collator:
    def __init__(self, smpl2mesh, data_path):
        self.smpl2mesh = smpl2mesh

    def __call__(self, batch):
        mesh1 = torch.tensor([data[0] for data in batch], dtype=torch.float32)
        mesh2 = torch.tensor([data[1] for data in batch], dtype=torch.float32)
        mesh3 = self.smpl2mesh([data[2] for data in batch])
        mesh4 = self.smpl2mesh([data[3] for data in batch])

        meshes = torch.stack([mesh1, mesh2, mesh3, mesh4], dim=1)

        return meshes
