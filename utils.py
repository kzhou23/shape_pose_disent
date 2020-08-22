import numpy as np
import torch
import os
from scipy import sparse
from data.lbs import lbs

def adj2inc(adjmat):
    edges = []
    edge_signs = dict.fromkeys(range(adjmat.shape[0]), 0)
    L_upper_mask = []
    nonzero_ind = np.nonzero(adjmat)
    for i, j in zip(*nonzero_ind):
        if (j, i) not in edges:
            edges.append((i, j))
            L_upper_mask.append(True)
        else:
            edge_signs[i] += 1
            L_upper_mask.append(False)

    incmat = np.zeros((len(edges), adjmat.shape[0]), dtype=np.float32)
    for row, (i, j) in enumerate(edges):
        incmat[row][i] = 1
        incmat[row][j] = -1

    signmat = np.abs(incmat.transpose())
    for i in range(adjmat.shape[0]):
        row = signmat[i]
        ind = np.nonzero(row)[0][:edge_signs[i]]
        row[ind] = -1

    return incmat, signmat, L_upper_mask


def get_A_b(L, x, diag, fix_ind, fix_val):
    '''
    return A and b_corr
    '''
    num_b = x.shape[0]

    A_shape = (L.size()[0], L.size()[0])

    row_ind = L._indices()[0]
    col_ind = L._indices()[1]

    row_mask = np.isin(row_ind, fix_ind)
    col_mask = np.isin(col_ind, fix_ind)

    A = []
    b_corr = []
    L_ind_row = row_ind[col_mask]
    L_ind_col = col_ind[col_mask]
    fix_val_mat = np.zeros(x[0].shape, dtype=np.float32)
    for i in range(num_b):
        values = L._values().numpy().copy()

        L_ind = sparse.coo_matrix((values[col_mask], (L_ind_row, L_ind_col)), shape=A_shape)
        L_ind = L_ind.tocsc()
        fix_val_mat[fix_ind, :] = fix_val[i]
        b_corr.append(L_ind.dot(fix_val_mat))

        values[row_mask] = 0
        values[col_mask] = 0
        m = sparse.coo_matrix((-values, (row_ind, col_ind)), shape=A_shape)

        d = diag.numpy().copy()
        d[fix_ind] = 1
        m.setdiag(d)
        m.eliminate_zeros()
        A.append(m.tocsc())

    return A, b_corr


class SMPL2Mesh:
    def __init__(self, bm_path, bm_type='smplh'):
        self.models = {}
        male_npz = np.load(os.path.join(bm_path, 'male/model.npz'))
        female_npz = np.load(os.path.join(bm_path, 'female/model.npz'))
        self.models['male'] = {k: male_npz[k] for k in male_npz}
        self.models['female'] = {k: female_npz[k] for k in female_npz}
        self.bm_type = bm_type

    def __call__(self, batch):
        batch_size = len(batch)

        gdr = torch.tensor([data['gender'] for (_, data) in enumerate(batch)], dtype=torch.int32)

        gender_ind = {}
        gender_ind['male'] = [idx for (idx, data) in enumerate(batch) if data['gender']==-1]
        gender_ind['female'] = list(set(range(batch_size)).difference(set(gender_ind['male'])))

        verts = {}
        for gdr in ['male', 'female']:
            if not gender_ind[gdr]:
                continue

            gdr_betas = torch.tensor([batch[idx]['shape'] for idx in gender_ind[gdr]],
                                     dtype=torch.float32)
            gdr_pose = torch.tensor([batch[idx]['pose'] for idx in gender_ind[gdr]],
                                    dtype=torch.float32)

            v_template = np.repeat(self.models[gdr]['v_template'][np.newaxis], len(gdr_betas), axis=0)
            v_template = torch.tensor(v_template, dtype=torch.float32)

            shapedirs = torch.tensor(self.models[gdr]['shapedirs'], dtype=torch.float32)

            posedirs = self.models[gdr]['posedirs']
            posedirs = posedirs.reshape(posedirs.shape[0]*3, -1).T
            posedirs = torch.tensor(posedirs, dtype=torch.float32)

            J_regressor = torch.tensor(self.models[gdr]['J_regressor'], dtype=torch.float32)

            parents = torch.tensor(self.models[gdr]['kintree_table'][0], dtype=torch.int32).long()

            lbs_weights = torch.tensor(self.models[gdr]['weights'], dtype=torch.float32)

            v, _ = lbs(gdr_betas, gdr_pose, v_template, shapedirs, posedirs, 
                       J_regressor, parents, lbs_weights, dtype=torch.float32)

            verts[gdr] = v

        mesh = torch.zeros(len(batch), 6890, 3)

        for gdr in ['male', 'female']:
            if gdr in verts:
                mesh[gender_ind[gdr]] = verts[gdr]

        return mesh
