from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
import argparse
import setproctitle
import mlflow
from mlflow.tracking import MlflowClient
import os
from tqdm import tqdm
import json
import copy
import random

from torch_geometric.data import Data as geoData
import torch_geometric.transforms as T

import torch.nn.functional as F

import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import setproctitle
setproctitle.setproctitle('RP@zzl')

device = torch.device('cuda')

def cross_entropy(pred, target):
    return torch.mean(-torch.sum(target * torch.log(pred), 1))

class Experiment:
    def __init__(self, lr, edim, batch_size):
        self.lr = lr
        self.edim = edim
        self.batch_size = batch_size
        self.batch_size_kg = params['batch_size_kg']
        self.num_iterations = args.num_iterations
        self.kwargs = params
        self.kwargs['device'] = device
        self.lamb = params['lamb']
        self.gs, self.edge_index = self.build_graph()

    def build_graph(self):
        gs=[]
        for k,v in d.mp2data.items():
            edge_index=torch.tensor([[x[0] for x in v['kg_data']], [x[2] for x in v['kg_data']]],dtype=torch.long,device=device)
            edge_type= torch.tensor([x[1] for x in v['kg_data']], dtype=torch.int, device=device)
            data=geoData(edge_index=edge_index,edge_attr=edge_type)
            trans=T.ToSparseTensor()
            trans(data)
            edge_index=data.adj_t

            eids=torch.tensor(list(v['ent2kgid'].values()),device=device)
            gs.append([edge_index,eids])

        # full kg
        edge_index=torch.tensor([[x[0] for x in d.kg_data], [x[2] for x in d.kg_data]],dtype=torch.long,device=device)
        edge_type= torch.tensor([x[1] for x in d.kg_data], dtype=torch.int, device=device)
        data=geoData(edge_index=edge_index,edge_attr=edge_type)
        trans=T.ToSparseTensor()
        trans(data)
        edge_index=data.adj_t

        return gs,edge_index

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size_kg]
        targets = torch.zeros((len(batch), len(d.ent2id)), device=device)
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return torch.tensor(batch, dtype=torch.long, device=device), targets

    def train_and_eval(self):
        print('building model....')
        model = HAN(d, **self.kwargs)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        er_vocab = self.get_er_vocab(d.kg_data)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        mob_adj=torch.tensor(d.mob_adj,device=device)

        allreg=list(range(d.nreg))
        for it in range(1, self.num_iterations + 1):
            print('\n=============== Epoch %d Starts...===============' % it)
            start_train = time.time()
            model.train()

            np.random.shuffle(er_vocab_pairs)
            np.random.shuffle(allreg)
            k=0
            losses_kg=[]
            losses_r=[]
            losses=[]
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size_kg)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                h_idx = data_batch[:, 0]
                r_idx = data_batch[:, 1]
                E_reg, predictions = model.forward(self.gs, h_idx, r_idx, self.edge_index)
                opt.zero_grad()
                # loss kg
                loss_kg = model.loss(predictions, targets)
                # loss reg
                if k+self.batch_size<=len(allreg):
                    uids=allreg[k:k+self.batch_size]
                else:
                    uids=allreg[k:]+allreg[:k+self.batch_size-len(allreg)]
                k=(k+self.batch_size)%len(allreg)
                u_idx = torch.tensor(uids, device=device)
                emb_sim=torch.mm(E_reg,E_reg.transpose(0,1))[u_idx]
                emb_sim=F.softmax(emb_sim,dim=1)
                loss_mob=cross_entropy(emb_sim,mob_adj[u_idx])
                loss_r=loss_mob
                # loss
                loss=self.lamb*loss_kg+(1-self.lamb)*loss_r

                loss.backward()
                opt.step()
                losses_kg.append(loss_kg.item())
                losses_r.append(loss_r.item())
                losses.append(loss.item())
            mlflow.log_metrics({'train_time': time.time()-start_train,
                                'loss_kg':np.mean(losses_kg),
                                'loss_r':np.mean(losses_r),
                                'loss':np.mean(losses),
                                'current_it': it}, step=it)
            print('loss:%.3f'%np.mean(losses))            

        E_reg,E_kg=model.get_emb(self.gs,self.edge_index)
        np.savez(archive_path + 'ER.npz',
                             E_reg=E_reg.detach().cpu().numpy(),E_kg=E_kg.detach().cpu().numpy())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="Batch size.")
    parser.add_argument("--batch_size_kg", type=int, default=2048, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?", help="Learning rate.")
    parser.add_argument("--edim", type=int, default=64, nargs="?", help="Entity embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, nargs="?", help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=20, nargs="?", help="random seed.")
    parser.add_argument('--hidden_size', default=128, type=int, help='')
    parser.add_argument('--lamb', default=0.5, type=float, help='lamb*loss_kg+(1-lamb)*loss_regs')

    args = parser.parse_args()
    print(args)

    metapaths=['spatial','OD','POI'] 
    data_dir = "./data/data_ny/" 
    archive_path = './output/output_ny/' 


    assert os.path.exists(data_dir)
    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    # ~~~~~~~~~~~~~~~~~~ mlflow experiment ~~~~~~~~~~~~~~~~~~~~~

    experiment_name = 'test'

    mlflow.set_tracking_uri('/data/zhouzhilun/Region_Profiling/mlflow_output/')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
        print('Initial Create!')
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id
        print('Experiment Exists, Continuing')
    with mlflow.start_run(experiment_id=EXP_ID) as current_run:
        
        # ~~~~~~~~~~~~~~~~~ reproduce setting ~~~~~~~~~~~~~~~~~~~~~
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Loading data....')
        d = Data(data_dir=data_dir, metapaths=metapaths)
        params = vars(args)
        mlflow.log_params(params)

        experiment = Experiment(batch_size=args.batch_size, lr=args.lr, edim=args.edim)
        experiment.train_and_eval()

