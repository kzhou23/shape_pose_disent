from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import numpy as np
import os, pickle, time
from visualize import output_meshes


def train_autoencoder(models, losses, optimizers, data, cur_epoch, arap_solver):
    optimizers['lr'].step(cur_epoch)
    optimizers['opt'].zero_grad()

    batch_size = data.size(0)
    
    x1, x2, x3 = data[:, 0], data[:, 1], data[:, 2]

    with torch.no_grad():
        x1n = np.random.uniform(0.7, 1.3) * x1
        x1n = x1n + 0.01*torch.rand_like(x1, device=x1.device)
        x2n = np.random.uniform(0.7, 1.3) * x2
        x2n = x2n + 0.01*torch.rand_like(x2, device=x2.device)

    zs1 = models['shape_enc'](x1)
    zs2 = models['shape_enc'](x2)
    zs3 = models['shape_enc'](x3)
    zp1 = models['pose_enc'](x1n)
    zp2 = models['pose_enc'](x2n)

    y2_1 = models['dec'](zs2, zp1)
    y3_2 = models['dec'](zs3, zp2)


    with torch.no_grad():
        y_ = arap_solver(x3, y3_2)

    zp2_ = models['pose_enc'](y_)
    y1_2 = models['dec'](zs1, zp2_)

    recon_loss = losses['cross_recon_loss'](x1, y2_1) * .5 + \
                 losses['cross_recon_loss'](x2, y1_2) * .5

    loss = recon_loss
    loss.backward()

    optimizers['opt'].step()

    return recon_loss.item()


def train_model(models, losses, optimizers, train_loader, vald_loader, device, config, arap_solver):
    num_epochs = config['train']['epochs']
    vis_step = config['train']['vis_step']
    log_step = config['train']['log_step']
    save_step = config['train']['save_step']

    if config['train']['resume']:
        start_epoch = config['train']['resume_epoch'] + 1
    else:
        start_epoch = 1

    print('Training started')

    loss_comp_dict = {}

    for epoch in range(start_epoch, start_epoch+num_epochs):
        for k in models:
            models[k].train()

        train_loss_dict = {'recon_loss': 0}

        for i, data in enumerate(train_loader):
            cur_epoch = epoch - 1 + i / len(train_loader)
            data = data.to(device)

            recon_loss = train_autoencoder(models, losses, optimizers, data, cur_epoch, arap_solver)
            train_loss_dict['recon_loss'] += recon_loss


        for k in models:
            models[k].eval()

        vald_recon_loss = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(vald_loader):
                data = data.to(device)
                
                x1, x2 = data[:, 0], data[:, 1]

                x3 = data[:, 2] - torch.mean(data[:, 2], dim=1, keepdim=True)
                x4 = data[:, 3] - torch.mean(data[:, 3], dim=1, keepdim=True)

                zs1 = models['shape_enc'](x1)
                zp1 = models['pose_enc'](x1)
                zs2 = models['shape_enc'](x2)
                zp2 = models['pose_enc'](x2)

                y1_2 = models['dec'](zs1, zp2)
                y2_1 = models['dec'](zs2, zp1)

                recon_loss = torch.mean(torch.mean(torch.sqrt(torch.sum((x3*1000 - y1_2*1000)**2, dim=2)), dim=1)) * .5 + \
                                torch.mean(torch.mean(torch.sqrt(torch.sum((x4*1000 - y2_1*1000)**2, dim=2)), dim=1)) * .5

                vald_recon_loss += recon_loss.item()

                if epoch % vis_step == 0 and batch_idx == 0:
                    x1 = x1.cpu().numpy().reshape(-1, 1, 6890, 3)
                    x2 = x2.cpu().numpy().reshape(-1, 1, 6890, 3)
                    x3 = x3.cpu().numpy().reshape(-1, 1, 6890, 3)
                    x4 = x4.cpu().numpy().reshape(-1, 1, 6890, 3)
                    y1_2 = y1_2.cpu().numpy().reshape(-1, 1, 6890, 3)
                    y2_1 = y2_1.cpu().numpy().reshape(-1, 1, 6890, 3)
                    meshes = np.concatenate([x1, x2, x3, x4, y1_2, y2_1], axis=1)
                    output_meshes(meshes, epoch, config)

        if epoch % log_step == 0:
            print('====> Epoch {}/{}: Training'.format(epoch, start_epoch+num_epochs-1))
            
            for term in train_loss_dict:
                print('\t{} {:.5f}'.format(term, train_loss_dict[term] / len(train_loader)))
            
            print('====> Epoch {}/{}: Validation loss {:.4f}'.format(epoch, start_epoch+num_epochs-1,
                                                                    vald_recon_loss / len(vald_loader)))

        if epoch % save_step == 0:
            checkpoint = dict([('m_'+t, models[t].state_dict()) for t in models])
            checkpoint.update(dict([('o_'+t, optimizers[t].state_dict()) for t in optimizers]))
            checkpoint.update({'torch_rnd': torch.get_rng_state(), 'numpy_rnd': np.random.get_state()})
            torch.save(checkpoint, os.path.join(config['path']['checkpoints'], '{}.pth'.format(epoch)))

        train_loader.dataset.resamp_ind()
