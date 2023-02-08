import numpy as np
import json

class Data:
    def __init__(self, data_dir, metapaths):
        self.metapaths=metapaths
        self.reg2id = self.load_region_data(data_dir)
        self.ent2id, self.rel2id, self.kg_data = self.load_full_kg(data_dir)
        self.mp2data = self.load_subkg_data(data_dir)
        self.nreg=len(self.reg2id)
        self.mob_adj=np.load(data_dir+'mob-adj.npy')[0]

        print('number of node=%d, number of edge=%d, number of relations=%d' % (len(self.ent2id), len(self.kg_data), len(self.rel2id)))
        print('sub-KGs:',metapaths)
        print('region num={}'.format(len(self.reg2id)))
        print('load finished..')

    def load_region_data(self, data_dir):      
        with open(data_dir + 'region2info.json', 'r') as f:
            region2info=json.load(f)
        regions=sorted(region2info.keys(),key=lambda x:x)
        reg2id=dict([(x,i) for i,x in enumerate(regions)])
        return reg2id
    
    def load_full_kg(self, data_dir):
        ent2id, rel2id = self.reg2id.copy(), {}
        kg_data_str = []
        with open(data_dir + 'kg.txt', 'r') as f:
            for line in f.readlines(): 
                h,r,t=line.strip().split('\t')
                kg_data_str.append((h,r,t))
        ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
        rels = sorted(list(set([x[1] for x in kg_data_str])))
        for i, x in enumerate(ents):
            try:
                ent2id[x]
            except KeyError:
                ent2id[x] = len(ent2id)
        rel2id = dict([(x, i) for i, x in enumerate(rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
        
        return ent2id, rel2id, kg_data

    def load_subkg_data(self, data_dir):
        mp2data={}
        for mp in self.metapaths:
            ent2id, rel2id = self.reg2id.copy(), {}
            kg_data_str = []
            with open(data_dir + 'kg_{}.txt'.format(mp), 'r') as f:
                for line in f.readlines():
                    h,r,t=line.strip().split('\t')
                    kg_data_str.append((h,r,t))
            ents = sorted(list(set([x[0] for x in kg_data_str] + [x[2] for x in kg_data_str])))
            rels = sorted(list(set([x[1] for x in kg_data_str])))
            for i, x in enumerate(ents):
                try:
                    ent2id[x]
                except KeyError:
                    ent2id[x] = len(ent2id)
            rel2id = dict([(x, i) for i, x in enumerate(rels)])
            kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in kg_data_str]
            ent2kgid={}
            for e in ent2id.keys():
                ent2kgid[e]=self.ent2id[e]
            mp2data[mp]={'ent2id':ent2id,'rel2id':rel2id,'kg_data':kg_data,'ent2kgid':ent2kgid}
        
        return mp2data

        