from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, glob
import torch
import torch.utils.data
import numpy as np
import tables

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import SMPL2Mesh

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 
              'BioMotionLab_NTroje', 'EKUT', 'TCD_handMocap', 'ACCAD']
}

gdr2num = {'male':-1, 'neutral':0, 'female':1}

class AMASS_Params_Row(tables.IsDescription):
    subject = tables.Int16Col(pos=1)
    gender = tables.Int8Col(pos=2)
    shape = tables.Float32Col(16, pos=3)
    pose = tables.Float32Col(52*3, pos=4)
    dmpl = tables.Float32Col(8, pos=5)
    trans = tables.Float32Col(3, pos=6)


class AMASS_H5(torch.utils.data.Dataset):
    def __init__(self, data_path, flag='train'):
        file_path = os.path.join(data_path, '_'+flag+'.h5')
        self.f = tables.open_file(file_path, 'r')
        self.data = self.f.root.data

        subj_ind = {}
        for row in self.data.iterrows():
            if row['subject'] not in subj_ind:
                subj_ind[row['subject']] = [row.nrow]
            else:
                subj_ind[row['subject']].append(row.nrow)

        self.subj_ind = {}
        for i, v in enumerate(subj_ind.values()):
            self.subj_ind[i] = v

        self.num_subj = len(self.subj_ind)
        self.num_meshes = len(self.data)
        print('Number of subjects:', self.num_subj)
        print('Number of meshes:', self.num_meshes)

    def __getitem__(self, index):
        for i in range(len(self.subj_ind)):
            if index in self.subj_ind[i]:
                subj_idx = i
                break

        colnames = self.data.coldescrs.keys()
        params = {k: self.data[index][k] for k in colnames}
        return subj_idx, params

    def __len__(self):
        return self.num_meshes


class Collator:
    def __init__(self, smpl2mesh):
        self.smpl2mesh = smpl2mesh

    def __call__(self, batch):
        ind = torch.tensor([data[0] for (_, data) in enumerate(batch)], dtype=torch.int32)
        gdr = torch.tensor([data[1]['gender'] for (_, data) in enumerate(batch)], dtype=torch.int32)
        betas = torch.tensor([data[1]['shape'] for (_, data) in enumerate(batch)],
                             dtype=torch.float32)
        pose = torch.tensor([data[1]['pose'] for (_, data) in enumerate(batch)],
                            dtype=torch.float32)

        batch = [data[1] for data in batch]
        mesh = self.smpl2mesh(batch)

        return ind, gdr, betas, pose, mesh


def amass2h5(splits, amass_dir, out_dir, step_size=100):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    sid = 0

    for split in amass_splits:
        print('Creating {} dataset'.format(split))
        h5_path = os.path.join(out_dir, '_{}.h5'.format(split))
        with tables.open_file(h5_path, mode="w") as h5file:
            table = h5file.create_table('/', 'data', AMASS_Params_Row)

            datasets = amass_splits[split]
            for ds_name in datasets:
                print('Processing dataset {}'.format(ds_name))
                subjects = glob.glob(os.path.join(amass_dir, ds_name, '*/'))
                for s in subjects:
                    print('\tProcessing subject {}'.format(s))
                    npz_fnames = glob.glob(os.path.join(s, '*_poses.npz'))

                    for npz_fname in npz_fnames:
                        print('\t\tProcessing motion sequence {}'.format(npz_fname))
                        fdata = np.load(npz_fname)
                        N = len(fdata['poses'])
                        
                        fids = range(int(0.1*N), int(0.9*N))[::step_size]

                        if not fids:
                            continue

                        data_subject = np.array([sid for _ in fids]).astype(np.int32)
                        data_gender = np.array(gdr2num[str(fdata['gender'].astype(np.str))]).astype(np.int8)
                        data_gender = np.repeat(data_gender[np.newaxis], repeats=len(fids), axis=0)
                        data_shape = np.repeat(fdata['betas'][np.newaxis], repeats=len(fids), axis=0).astype(np.float32)
                        data_pose = fdata['poses'][fids].astype(np.float32)
                        data_dmpl = fdata['dmpls'][fids].astype(np.float32)
                        data_trans = fdata['trans'][fids].astype(np.float32)

                        table.append(list(zip(data_subject, data_gender, data_shape, data_pose, data_dmpl, data_trans)))
                        table.flush()

                    sid += 1
        

if __name__ == '__main__':
    if not os.path.isfile('./amass/processed/_train.h5'):
        amass2h5(amass_splits, './amass/datasets', './amass/processed')
    print('Processing Stage I finished')

    smpl2mesh = SMPL2Mesh('./amass/body_models/smplh', bm_type='smplh')
    collator = Collator(smpl2mesh)

    for split in ['train', 'vald', 'test']:
        data_path = os.path.join('./amass/processed', '{}.npy'.format(split))

        amass_dataset = AMASS_H5('./amass/processed', split)
        h5loader = torch.utils.data.DataLoader(amass_dataset, batch_size=2048, 
                                               num_workers=1, collate_fn=collator)

        total_meshes = len(amass_dataset)
        np_data = []
        np_dtype = np.dtype('i4, i4, (16)f4, (156)f4, (20670)f4')
        for ind, gdr, betas, pose, mesh in h5loader:
            ind = ind.numpy().astype(np.int32)
            gdr = gdr.numpy().astype(np.int32)
            betas = betas.numpy().reshape(-1, 16).astype(np.float32)
            pose = pose.numpy().reshape(-1, 156).astype(np.float32)
            mesh = mesh.numpy().reshape(-1, 6890*3).astype(np.float32)

            np_data.extend(list(zip(ind, gdr, betas, pose, mesh)))

        np_data = np.array(np_data, dtype=np_dtype)

        np.save(data_path, np_data)
                
    print('Processing Stage II finished')
