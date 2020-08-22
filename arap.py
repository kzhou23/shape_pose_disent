from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import multiprocessing as mp
import numpy as np
import torch
import scipy.sparse as sparse
import scipy.linalg as linalg
from sksparse.cholmod import cholesky
from utils import adj2inc, get_A_b


class ARAP_Solver:
    def __init__(self, adj, inc, sign_inc, device):
        self.L = torch.tensor(adj, dtype=torch.float32)
        self.L[self.L.nonzero(as_tuple=True)] = 1
        self.diag = torch.sum(self.L, dim=1)
        self.L = self.L.to_sparse()

        self.inc = inc
        self.sign_inc = sign_inc

        self.p = mp.Pool(32)

        fix_ind = np.random.choice(6890, size=256, replace=False)
        fix_ind.sort()
        self.fix_ind = torch.tensor(fix_ind, dtype=torch.int64, device=device)

    def __call__(self, x, y):
        fix_val = y[:, self.fix_ind]
        return self.arap_solve(x, y, self.fix_ind, fix_val, num_iter=1)

    def __del__(self):
        self.p.close()
    
    def arap_solve(self, x, y, fix_ind, fix_val, num_iter=1):
        '''
        x, y: V*3
        fix_ind: K
        fix_val: B*K*3
        '''
        num_b = x.size()[0]
        num_v = x.size()[1]
        num_e = self.inc.size()[0]

        fix_ind = fix_ind.cpu()
        fix_val = fix_val.cpu()

        A, b_corr = get_A_b(self.L, x.cpu().numpy(), self.diag, fix_ind, fix_val)
        b_corr = torch.from_numpy(np.stack(b_corr, axis=0))

        factors = []
        for i in range(num_b):
            factors.append(cholesky(A[i]))

        for j in range(num_iter):
            y_list = []

            rots = self.opt_rot(x, y)

            b = self.b_from_rot(rots, x).cpu()
            b = b + b_corr
            b[:, fix_ind] = fix_val

            for i in range(num_b):
                y_list.append(torch.tensor(factors[i](b[i]), dtype=torch.float32, device=x.device))

            y = torch.stack(y_list, dim=0)

        return y
    
    def opt_rot(self, x, y):
        num_b = x.size()[0]
        num_v = x.size()[1]
        num_e = self.inc.size()[0]

        z = torch.cat((x, y), dim=0)
        z = z.permute(1, 2, 0).contiguous().view(num_v, 6*num_b)

        edges = torch.sparse.mm(self.inc, z).view(num_e, 3, 2*num_b).permute(2, 0, 1)
        x_edges, y_edges = torch.split(edges, num_b, dim=0)

        edges_outer = torch.matmul(x_edges[:, :, :, None], y_edges[:, :, None, :])
        edges_outer = edges_outer.transpose(1, 0).contiguous().view(-1, num_b*9)

        abs_inc = self.inc.clone()
        abs_inc._values()[:] = torch.abs(abs_inc._values())
        # transposed S
        S = abs_inc.t().mm(edges_outer).view(num_v, num_b, 3, 3).permute(1, 0, 3, 2)
        S = S.cpu()
        res = self.p.map(np.linalg.svd, S.numpy())
        v = torch.stack([torch.from_numpy(i[0]) for i in res])
        u = torch.stack([torch.from_numpy(i[2].transpose(0, 2, 1)) for i in res])
        v = v.to(x.device)
        u = u.to(x.device)
        det_sign = torch.det(v) * torch.det(u)
        u[..., 2] = u[..., 2] * det_sign[..., None]
        opt_rots = torch.matmul(v, u.transpose(-2, -1))

        return opt_rots

    def b_from_rot(self, rots, x):
        '''
        L: B*V*V
        inc: E*V
        inc_sign: V*E
        rot: B*V*3*3
        x: B*V*3
        '''
        num_b = x.size()[0]
        num_v = x.size()[1]
        num_e = self.inc.size()[0]
        
        edges = self.inc.mm(x.transpose(1, 0).contiguous().view(num_v, num_b*3))
        edges = edges.view(num_e, num_b, 3).transpose(1, 0)

        abs_inc = self.inc.clone()
        abs_inc._values()[:] = torch.abs(abs_inc._values())

        edge_rots = abs_inc.mm(rots.transpose(1, 0).contiguous().view(num_v, num_b*9))
        edge_rots = edge_rots.view(num_e, num_b, 3, 3).transpose(1, 0)
        edges = torch.matmul(edge_rots, edges[..., None]).squeeze(-1)
        weighted_edges = .5 * edges

        b = self.sign_inc.mm(weighted_edges.transpose(1, 0).contiguous().view(num_e, num_b*3))
        b = b.view(num_v, num_b, 3).transpose(1, 0)

        return b