import json
import numpy as np
import itertools
import copy
import random
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

with open('./data/data_ny/region2info.json','r') as f:
    region2info=json.load(f)
print(len(region2info))

# region2id
regions=sorted(region2info.keys(),key=lambda x:x)
reg2id=dict([(x,i) for i,x in enumerate(regions)])

# load embedding
embfile="./output/output_ny/ER.npz"
data=np.load(embfile)
emb=np.concatenate((data['E_kg'],data['E_reg']),1)

fixparams={'random_state':0}
params_grid={
            'alpha':[0,1,5,10],
            }
tmp=list(params_grid.values())
tmp=list(itertools.product(*tmp))
keys=list(params_grid.keys())
params_list=[]
for p in tmp:
    params={}
    for i,key in enumerate(params_grid.keys()):
        params[key]=p[i]
    params_list.append(params)
    
def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2
    
def regression(params):
    fullparams=copy.deepcopy(params)
    for k,v in fixparams.items():
        fullparams[k]=v

    seed = 20
    np.random.seed(seed)
    random.seed(seed)
    
    reg = linear_model.Ridge(**fullparams)
    reg.fit(x_train, y_train)
    # valid
    y_pred = reg.predict(x_valid)
    mae, rmse, r2 = compute_metrics(y_pred, y_valid)
    valmetric=r2
    # test
    y_pred = reg.predict(x_test)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)

    result={'params':params}
    for m in ['mae','rmse','r2']:
        result[m]=eval(m)
    return (valmetric,result)
    
for ind_type in ['pop','edu','income','crime']:
    print('------------{}------------'.format(ind_type))
    data = []
    region_str = []
    for k,v in region2info.items():
        ind=v[ind_type]
        if ind_type in ['pop','income','crime']:
            ind=np.log(ind) if ind>1 else 0
        region_str.append(k)
        data.append([k,ind])

    data.sort(key=lambda x:x)
    np.random.seed(seed=20)
    np.random.shuffle(data)
    region_str = sorted(list(set(region_str)), key=lambda x: x)
    region2id = dict([(x, i) for i, x in enumerate(region_str)])
    data_id = [[region2id[x[0]],x[1]] for x in data]

    L = len(data_id)
    train_data, valid_data, test_data = \
        data_id[0:int(L * 0.6)], data_id[int(L * 0.6):int(L * 0.8)], data_id[int(L * 0.8)::]

    x_train=[emb[x[0],:].tolist() for x in train_data]
    y_train=[x[1] for x in train_data]
    x_valid=[emb[x[0],:].tolist() for x in valid_data]
    y_valid=[x[1] for x in valid_data]
    x_test=[emb[x[0],:].tolist() for x in test_data]
    y_test=[x[1] for x in test_data]
    
    # regression
    results=[]
    for params in params_list:
        result=regression(params)
        results.append(result)  
    results.sort(key=lambda x:x[0],reverse=True)
    for m in ['mae','rmse','r2']:
        print('%.3f'%results[0][1][m],end='\t')
    print('')
    
    