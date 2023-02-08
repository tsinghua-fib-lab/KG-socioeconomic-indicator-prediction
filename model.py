import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import torch.nn as nn
import numpy as np
from torch_geometric.nn import RGCNConv


class MyLoss_Pretrain(torch.nn.Module):
    def __init__(self):
        super(MyLoss_Pretrain, self).__init__()
        return

    def forward(self, pred, tar):
        kg_pred_max = torch.max(pred, dim=1)[0].view(-1, 1)
        kg_pred_log_max_sum = torch.log(torch.sum(torch.exp(pred-kg_pred_max), dim=1)).view(-1, 1)
        kg_pred_log_softmax = pred - kg_pred_max - kg_pred_log_max_sum
        loss_kge = - kg_pred_log_softmax[tar==True].sum()
        return loss_kge

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)

class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, nrs, dropout, nreg):
        super(HANLayer, self).__init__()
        self.gnn_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gnn_layers.append(RGCNConv(in_size, out_size, nrs[i]))
        
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.num_meta_paths = num_meta_paths
        self.dropout=dropout
        self.nreg=nreg

    def forward(self, gs, E, ifdropout):
        semantic_embeddings = []
        for i,g in enumerate(gs):
            edge_index,eids=g[0],g[1]
            E_feat = E[eids]
            E_feat = self.gnn_layers[i](E_feat,edge_index=edge_index)
            if ifdropout:
                E_feat = F.dropout(E_feat,p=self.dropout)
            E_feat = F.relu(E_feat)
            semantic_embeddings.append(E_feat[:self.nreg])

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_attention(semantic_embeddings)

class HAN(nn.Module):
    def __init__(self, d, **kwargs):
        super(HAN, self).__init__()
        num_meta_paths=len(d.metapaths)
        self.nmp=num_meta_paths
        self.nreg=d.nreg
        ne=len(d.ent2id)
        nr=len(d.rel2id)
        nes=[len(v['ent2id']) for v in d.mp2data.values()]
        nrs=[len(v['rel2id']) for v in d.mp2data.values()]
        hidden_size=kwargs['hidden_size']
        self.R = torch.nn.Embedding(nr, kwargs['edim'])
        
        self.E=nn.Embedding(ne,kwargs['edim'])
        self.init()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, kwargs['edim'], hidden_size, nrs, kwargs['dropout'], d.nreg))

        self.predict = nn.Linear(hidden_size, kwargs['edim'])

        self.rgcn=RGCNConv(kwargs['edim'], kwargs['edim'], nr)
        self.dropout=kwargs['dropout']
        self.loss=MyLoss_Pretrain()
    
    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, gs, h_idx, r_idx, edge_index):
        # RGCN
        E = self.E.weight
        E = self.rgcn(E,edge_index=edge_index)
        E = F.dropout(E,p=self.dropout)
        E = torch.tanh(E)

        # KG completion
        h=E[h_idx] # bs*edim
        r=self.R(r_idx) # bs*edim
        x=h*r # bs*edim
        pred = torch.mm(x, E.transpose(1, 0)) # bs*ne

        # metapaths
        for gnn in self.layers:
            h = gnn.forward(gs, E, ifdropout=True)
        E_reg=self.predict(h)

        E_reg=E_reg+E[:self.nreg]

        return E_reg, pred
    
    def get_emb(self, gs, edge_index):
        # RGCN
        E = self.E.weight
        E = self.rgcn(E,edge_index=edge_index)
        E = torch.tanh(E)

        # metapaths
        for gnn in self.layers:
            h = gnn.forward(gs, E, ifdropout=False)
        E_reg=self.predict(h)

        E_kg=E[:self.nreg]

        return E_reg, E_kg

