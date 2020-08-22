from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpiralConv(nn.Module):
    def __init__(self, in_dim, out_dim, spiral, act, deci_verts=None, bias=True):
        super(SpiralConv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spiral = spiral
        self.spiral_size = spiral.size(2)
        self.act = act
        self.deci_verts = deci_verts
        if deci_verts is not None:
            self.deci_verts = torch.cat([self.deci_verts, torch.tensor([-1.]).long().to(spiral.device)])

        self.conv = nn.Linear(in_dim*self.spiral_size, out_dim, bias=bias)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.02)
        elif act == 'identity':
            self.act = lambda x: x
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError

    def forward(self, x):
        batch_size, num_verts, num_feats = x.size()
        spiral = self.spiral.repeat(batch_size, 1, 1)
        if self.deci_verts is not None:
            spiral = spiral[:, self.deci_verts]
            num_verts = len(self.deci_verts) - 1
        assert num_feats == self.in_dim

        x_dummy = torch.cat([x, torch.zeros((batch_size, 1, num_feats), device=x.device)], dim=1)
        num_verts += 1
        
        batch_ind = torch.arange(batch_size, device=x.device).view(-1, 1).repeat(1, num_verts*self.spiral_size).flatten().long()
        spiral_ind = spiral.flatten()
        flat_feat = x_dummy[batch_ind, spiral_ind, :].view(-1, self.spiral_size*num_feats)

        out_feat = self.conv(flat_feat)
        out_feat = self.act(out_feat)

        out_feat = out_feat.view(batch_size, num_verts, self.out_dim)[:, :-1, :]

        return out_feat


class SpiralEncoder(nn.Module):
    def __init__(self, enc_filters, spirals, nlat, D, act='relu'):
        super(SpiralEncoder, self).__init__()
        self.enc_filters = enc_filters
        self.nlat = nlat
        self.D = D
        self.act = act

        self.enc_layers = []
        for i in range(len(enc_filters)-1):
            in_dim = enc_filters[i]
            out_dim = enc_filters[i+1]
            layer = SpiralConv(in_dim, out_dim, spirals[i], act, torch.nonzero(D[i], as_tuple=True)[1])
            self.enc_layers.append(layer)
        self.enc_layers = nn.ModuleList(self.enc_layers)

        num_verts = D[-1].size()[0]
        self.fc_enc = nn.Linear(num_verts*enc_filters[-1], nlat)

    def forward(self, x):
        batch_size = x.size()[0]

        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x)
            
        x = x.contiguous().view(batch_size, -1)
        z = self.fc_enc(x)

        return z


class SpiralDecoder(nn.Module):
    def __init__(self, dec_filters, spirals, nlat, U, act='relu'):
        super(SpiralDecoder, self).__init__()
        self.dec_filters = dec_filters
        self.nlat = nlat
        self.U = U
        self.act = act

        num_verts = U[-1].size()[1]
        self.fc_dec = nn.Linear(nlat, num_verts*dec_filters[0])

        self.dec_layers = []
        for i in range(len(dec_filters)-1):
            in_dim = dec_filters[i]
            out_dim = dec_filters[i+1]
            if i != len(dec_filters)-2:
                layer = SpiralConv(in_dim, out_dim, spirals[-1-i], act)
            else:
                layer = SpiralConv(in_dim, out_dim, spirals[-1-i], 'identity')
            self.dec_layers.append(layer)
        self.dec_layers = nn.ModuleList(self.dec_layers)

    def forward(self, zs, zp):
        batch_size = zs.size()[0]

        z = torch.cat([zs, zp], dim=1)
        x = self.fc_dec(z).view(batch_size, -1, self.dec_filters[0])

        for i in range(len(self.dec_layers)):
            x = torch.matmul(self.U[-i-1], x)
            x = self.dec_layers[i](x)

        return x
