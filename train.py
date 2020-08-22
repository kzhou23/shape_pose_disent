from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import os, json, pickle

from itertools import chain
from torch import nn, optim
from sklearn.metrics.pairwise import euclidean_distances
from psbody.mesh import Mesh

from data.mesh_sampling import generate_transform_matrices
from data.data_loading import AMASS_Dataset, AMASS_Dataset_Eval, train_collate, AMASS_Eval_Collator
from model.model import SpiralEncoder, SpiralDecoder
from spiral_utils import get_adj_trigs, generate_spirals
from losses import l1_loss
from train_fn import train_model
from utils import adj2inc, SMPL2Mesh
from arap import ARAP_Solver

torch.set_printoptions(precision=6)

config_path = './config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

if config['train']['deterministic']:
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for pathname in config['path']:
    path = config['path'][pathname]
    if not os.path.isdir(path):
        os.makedirs(path)

# ----------------------------- prepare data -----------------------------
# reference SMPL template
ref_bm_path = os.path.join(config['path']['body_models'], 'neutral/model.npz')
ref_bm = np.load(ref_bm_path)
ref_mesh = Mesh(v=ref_bm['v_template'], f=ref_bm['f'])

# path to downsampling/upsampling matrices
sampling_path = config['path']['sampling_matrices']
# sampling factors between two consecutive mesh resolutions
scale_factors = config['model']['scale_factors']
if not os.path.exists(os.path.join(sampling_path, 'matrices.pkl')):
    print('Generating mesh sampling matrices')
    M, A, D, U, F = generate_transform_matrices(ref_mesh, scale_factors)

    with open(os.path.join(sampling_path, 'matrices.pkl'), 'wb') as f:
        M_vf = [(M[i].v, M[i].f) for i in range(len(M))]
        pickle.dump({'M':M_vf,'A':A,'D':D,'U':U,'F':F}, f)
else:
    print('Loading mesh sampling matrices')
    with open(os.path.join(sampling_path, 'matrices.pkl'), 'rb') as f:    
        matrices = pickle.load(f)
        M = [Mesh(v=v, f=f) for v,f in matrices['M']][:len(scale_factors)+1]
        A = matrices['A'][:len(scale_factors)+1]
        D = matrices['D'][:len(scale_factors)]
        U = matrices['U'][:len(scale_factors)]
        F = matrices['F'][:len(scale_factors)]

for i in range(len(D)):
    D[i] = D[i].todense()
    U[i] = U[i].todense()

# processed AMASS data (see data/data_extraction.py)
data_dir = config['path']['processed_data']

train_set = AMASS_Dataset(data_dir, 'train')
vald_set = AMASS_Dataset_Eval(data_dir)

smpl2mesh = SMPL2Mesh(config['path']['body_models'], bm_type='smplh')

train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['dataloader']['batch_size'],
                                           shuffle=True, pin_memory=False,
                                           num_workers=config['dataloader']['num_workers'],
                                           collate_fn=train_collate)

vald_collator = AMASS_Eval_Collator(smpl2mesh, data_dir)
vald_loader = torch.utils.data.DataLoader(vald_set, batch_size=config['dataloader']['batch_size'],
                                          shuffle=False, pin_memory=False,
                                          num_workers=config['dataloader']['num_workers'],
                                          collate_fn=vald_collator)

# ----------------------------- set up models -----------------------------
multi_gpu = False
device_config = config['train']['device']
if isinstance(device_config, list):
    device_id = device_config[0]
    if len(device_config) > 1:
        multi_gpu = True
else:
    device_id = device_config
device = torch.device('cuda:'+str(device_id) if torch.cuda.is_available() else 'cpu')

adj, trigs = get_adj_trigs(A, F, ref_mesh)

# reference vertex id when calculating spirals, check Neural3DMM for details
reference_points = [[0]]
for i in range(len(config['model']['scale_factors'])):
    dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
    reference_points.append(np.argmin(dist,axis=0).tolist())

spirals, spiral_sizes, _ = generate_spirals(config['model']['conv_hops'], 
                                            M[:-1], adj[:-1], trigs[:-1], reference_points[:-1],
                                            dilation=config['model']['dilation'])

spirals = [torch.from_numpy(s).long().to(device) for s in spirals]
D = [torch.from_numpy(s).float().to(device) for s in D]
U = [torch.from_numpy(s).float().to(device) for s in U]

# number of feature channels for each mesh resolution
shape_enc_filters = config['model']['shape_enc_filters']
pose_enc_filters = config['model']['pose_enc_filters']
dec_filters = config['model']['dec_filters']
# dimensions for latent components
shape_dim = config['model']['shape_dim']
pose_dim = config['model']['pose_dim']
# activation function
activation = config['model']['activation']

if multi_gpu:
    shape_enc = nn.DataParallel(SpiralEncoder(shape_enc_filters, spiral_sizes, spirals,
                                shape_dim, D, act=activation), device_ids=device_config)
    pose_enc = nn.DataParallel(SpiralEncoder(pose_enc_filters, spiral_sizes, spirals,
                               pose_dim, D, act=activation), device_ids=device_config)
    dec = nn.DataParallel(SpiralDecoder(dec_filters, spiral_sizes, spirals,
                          shape_dim+pose_dim, U, act=activation), device_ids=device_config)
    print('Train with GPUs', device_config)
else:
    shape_enc = SpiralEncoder(shape_enc_filters, spirals, shape_dim, D, act=activation, bn=False)
    pose_enc = SpiralEncoder(pose_enc_filters, spirals, pose_dim, D, act=activation, bn=False)
    dec = SpiralDecoder(dec_filters, spirals, shape_dim+pose_dim, U, act=activation)

shape_enc = shape_enc.to(device)
pose_enc = pose_enc.to(device)
dec = dec.to(device)

model_dict = {
    'shape_enc': shape_enc,
    'pose_enc': pose_enc,
    'dec': dec,
}
# ----------------------------- set up optimizer -----------------------------
opt_type = config['optimizer']['type']
opt_args = config['optimizer']['args']
if opt_type == 'Adam':
    opt = optim.AdamW(list(shape_enc.parameters())+list(pose_enc.parameters())+list(dec.parameters()), **opt_args)
lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, eta_min=1e-5)

opt_dict = {
    'opt': opt,
    'lr': lr_scheduler
}

loss_dict = {}
loss_dict['cross_recon_loss'] = l1_loss
adjmat = A[0].todense()
incmat_np, signmat_np, _ = adj2inc(adjmat)
incmat = torch.from_numpy(incmat_np).to_sparse().to(device)
signmat = torch.from_numpy(signmat_np).to_sparse().to(device)
face_ind = torch.tensor(ref_bm['f'].astype(np.int64)).to(device)

arap_solver = ARAP_Solver(adjmat, incmat, signmat, device)

# ----------------------------- train model -----------------------------
if config['train']['resume']:
    resume_epoch = config['train']['resume_epoch']
    # path to saved model checkpoints
    checkpoint_path = os.path.join(config['path']['checkpoints'], '{}.pth'.format(resume_epoch))
    checkpoint = torch.load(checkpoint_path)

    for k in model_dict:
        model_dict[k].load_state_dict(checkpoint['m_'+k])

    for k in opt_dict:
        opt_dict[k].load_state_dict(checkpoint['o_'+k])

    if config['train']['deterministic']:
        torch.set_rng_state(checkpoint['torch_rnd'])
        np.random.set_state(checkpoint['numpy_rnd'])

    print('Resume training from epoch', resume_epoch)

train_model(model_dict, loss_dict, opt_dict,
            train_loader=train_loader,
            vald_loader=vald_loader,
            device=device,
            config=config,
            arap_solver=arap_solver)