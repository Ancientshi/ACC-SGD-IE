import os, sys
from DataModule import MnistModule, NewsModule, AdultModule
import argparse
import numpy as np
import joblib
import torch
from MyNet import LogReg, DNN, NetList
from train import *
import random
import copy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 200
lr = 0.01
momentum = 0.9
num_epochs = 100


def compute_gradient(x, y, model, loss_fn):
    z = model(x)
    loss = loss_fn(z, y)
    model.zero_grad()
    loss.backward()
    u = [param.grad.data.clone() for param in model.parameters()]
    for uu in u:
        uu.requires_grad = False
    return u

def infl_true(key, model_type, seed=0, gpu=0):
    corrupted_str='_corrupted' if args.corrupted else ''
    dn = './%s%s/%s_%s_%s' % (args.datasize,corrupted_str,key, model_type, suffix)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_true%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = settings_logreg(key)
        _, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_val, y_val) = z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = settings_dnn(key)
        _, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_val, y_val) = z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # influence
    z = model(x_val)
    loss = loss_fn(z, y_val)
    infl = np.zeros(n_tr)
    for i in range(n_tr):
        m = res['counterfactual'].models[i]
        m.eval()
        zi = m(x_val)
        lossi = loss_fn(zi, y_val)
        infl[i] = lossi.item() - loss.item()
    
    # save
    joblib.dump(infl, gn, compress=9)


def infl_sgd(key, model_type, seed=0, gpu=0):
    corrupted_str='_corrupted' if args.corrupted else ''
    dn = './%s%s/%s_%s_%s' % (args.datasize,corrupted_str,key, model_type, suffix)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_sgd%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    #loss_fn = torch.nn.functional.nll_loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    
    # model list
    models = res['models'].models[:-1]
    
    # influence
    alpha = res['alpha']
    info = res['info']
    infl = np.zeros(n_tr)
    for t in range(len(models)-1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]['idx'], info[t]['lr']
        
        z = m(x_tr[idx])
        loss = loss_fn(z, y_tr[idx])
        for p in m.parameters():
            loss += 0.5 * alpha * (p * p).sum()
        m.zero_grad()
        grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)

        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            example_grad_params = torch.autograd.grad(loss, m.parameters())
            batch_grad_params = grad_params
        
            for j in range(len(batch_grad_params)):
                uu=u[j]
                example_grad_param=example_grad_params[j]
                batch_grad_param=batch_grad_params[j]
                infl[i] += (lr/ (idx.size)) * (u[j] * (example_grad_param)).sum().item()
                
        # update u
        ug = 0
        for uu, g in zip(u, grad_params):
            ug += (uu * g).sum()
        m.zero_grad()
        ug.backward()
        for j, param in enumerate(m.parameters()):
            #u[j] -= lr * param.grad.data / idx.size
            u[j] -= lr * param.grad.data
        
    # save
    joblib.dump(infl, gn, compress=9)

def infl_proposed(key, model_type, seed=0, gpu=0, simpj=1):
    if simpj:
        infl_proposed_simpj(key, model_type, seed, gpu)
    else:
        infl_proposed_nosimpj(key, model_type, seed, gpu)
        
def infl_proposed_simpj(key, model_type, seed=0, gpu=0):
    corrupted_str='_corrupted' if args.corrupted else ''
    dn = './%s%s/%s_%s_%s' % (args.datasize,corrupted_str,key, model_type, suffix)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_proposed_simpj%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)


    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn_none= torch.nn.BCEWithLogitsLoss(reduction='none')
    model.eval()

    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]


    # model list 
    models = res['models'].models[:-1]

    # influence
    alpha = res['alpha']
    info = res['info']

    infl = np.zeros(n_tr)
    for t in range(len(models)-1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]['idx'], info[t]['lr']
        
        z = m(x_tr[idx])
        loss = loss_fn(z, y_tr[idx])
        for p in m.parameters():
            loss += 0.5 * alpha * (p * p).sum()
        m.zero_grad() 
        grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
        
        z=m(x_tr[idx])
        loss_per_list=loss_fn_none(z, y_tr[idx])
        for p in m.parameters():
            loss_per_list += 0.5 * alpha * (p * p).sum()
        
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            
            example_grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
            batch_grad_params = grad_params
            
            for j in range(len(batch_grad_params)):
                uu=u[j]
                example_grad_param=example_grad_params[j]
                batch_grad_param=batch_grad_params[j]

                #in SGD-Influence it is as follows, we change this
                #infl[i] += (lr/ (idx.size)) * (u[j] * (example_grad_param)).sum().item() 
                infl[i] += (lr/ (idx.size-1)) * (u[j] * (example_grad_param-batch_grad_param)).sum().item() 

        ug = 0
        for uu, g in zip(u, grad_params):
            ug += (uu * g).sum()
        m.zero_grad()
        ug_grad=torch.autograd.grad(ug, m.parameters(), create_graph=False)
        
        #if simplify V_i^k, then every example share the same calculation
        for j in range(len(u)):
            ug_grad_param=ug_grad[j]
            u[j] = u[j]-lr * ug_grad_param

    # save
    joblib.dump(infl, gn, compress=9)
    
def infl_proposed_nosimpj(key, model_type, seed=0, gpu=0):
    corrupted_str='_corrupted' if args.corrupted else ''
    dn = './%s%s/%s_%s_%s' % (args.datasize,corrupted_str,key, model_type, suffix)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_proposed_nosimpj%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)


    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn_none= torch.nn.BCEWithLogitsLoss(reduction='none')
    model.eval()

    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    #k,u
    u_dict={}
    #u
    for i in range(n_tr):
        u_dict[i]=[uu.clone() for uu in u]

    # model list 
    models = res['models'].models[:-1]

    # influence
    alpha = res['alpha']
    info = res['info']

    infl = np.zeros(n_tr)
    for t in range(len(models)-1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]['idx'], info[t]['lr']
        
        z = m(x_tr[idx])
        loss = loss_fn(z, y_tr[idx])
        for p in m.parameters():
            loss += 0.5 * alpha * (p * p).sum()
        m.zero_grad() 
        grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
        
        z=m(x_tr[idx])
        loss_per_list=loss_fn_none(z, y_tr[idx])
        for p in m.parameters():
            loss_per_list += 0.5 * alpha * (p * p).sum()
        
        per_example_lossgap_grad_params_dict={}
        for i,loss_per_example in zip(idx,loss_per_list):
            gap_loss=loss_per_example-loss_per_list.mean()
            m.zero_grad() 
            per_example_lossgap_grad_params= torch.autograd.grad(gap_loss, m.parameters(), create_graph=True)
            per_example_lossgap_grad_params_dict[i]=per_example_lossgap_grad_params
        
        
        '''idx:
        [150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167
        168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185
        186 187 188 189 190 191 192 193 194 195 196 197 198 199]
        '''
        for i in idx:
            per_example_lossgap_grad_params = per_example_lossgap_grad_params_dict[i]
    
            for j in range(len(per_example_lossgap_grad_params)):
                uu=u_dict[i][j]
                infl[i] += (lr/ (idx.size-1)) * (u_dict[i][j] * per_example_lossgap_grad_params[j]).sum().item() 
        
        for i in range(n_tr):
            if i in idx:
                ug = 0
                for uu, g in zip(u_dict[i], grad_params):
                    ug += (uu * g).sum()
                m.zero_grad()
                ug_grad=torch.autograd.grad(ug, m.parameters(), create_graph=False,retain_graph=True)
                
                ug2=0
                for uu, g in zip(u_dict[i], per_example_lossgap_grad_params):
                    ug2 += (uu * g).sum()
                m.zero_grad()
                ug2_grad=torch.autograd.grad(ug2, m.parameters(), create_graph=False,retain_graph=True)
            
                for j in range(len(u_dict[i])):
                    ug_grad_param=ug_grad[j]
                    ug2_grad_param=ug2_grad[j]
                    u_dict[i][j] = u_dict[i][j]-lr * ug_grad_param + \
                                                lr/(idx.size)* ug2_grad_param
                
                
            else: 
                ug = 0
                for uu, g in zip(u_dict[i], grad_params):
                    ug += (uu * g).sum()
                m.zero_grad()
                ug_grad=torch.autograd.grad(ug, m.parameters(), create_graph=False,retain_graph=True)
                
                for j in range(len(u_dict[i])):
                    ug_grad_param=ug_grad[j]
                    u_dict[i][j] = u_dict[i][j]-lr * ug_grad_param
            
    # save
    joblib.dump(infl, gn, compress=9)
    
def infl_nohess(key, model_type, seed=0, gpu=0):
    corrupted_str='_corrupted' if args.corrupted else ''
    dn = './%s%s/%s_%s_%s' % (args.datasize,corrupted_str,key, model_type, suffix)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_nohess%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    
    # model list
    models = res['models'].models[:-1]
    
    # influence
    alpha = res['alpha']
    info = res['info']
    infl = np.zeros(n_tr)
    for t in range(len(models)-1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]['idx'], info[t]['lr']
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            loss.backward()
            for j, param in enumerate(m.parameters()):
                infl[i] += lr * (u[j].data * param.grad.data).sum().item() / idx.size
        
    # save
    joblib.dump(infl, gn, compress=9)
    

def infl_icml(key, model_type, seed=0, gpu=0):
    corrupted_str='_corrupted' if args.corrupted else ''
    dn = './%s%s/%s_%s_%s' % (args.datasize,corrupted_str,key, model_type, suffix)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_icml%03d.dat' % (dn, seed)
    hn = '%s/loss_icml%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        #module, (n_tr, n_val, n_test), (_, _, _, batch_size) = settings_logreg(key)
        module, (n_tr, n_val, n_test), _ = settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        #module, (n_tr, n_val, n_test), m, _, (_, _, _, batch_size) = settings_dnn(key)
        module, (n_tr, n_val, n_test), m, _, _ = settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed,args=args)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    
    # Hinv * u with SGD
    if model_type == 'logreg':
        alpha = res['alpha']
    elif model_type == 'dnn':
        alpha = 1.0
    num_steps = int(np.ceil(n_tr / batch_size))
    #v = [torch.zeros(*param.shape, requires_grad=True, device=device) for param in model.parameters()]
    v = []
    for uu in u:
        v.append(uu.clone())
        v[-1].to(device)
        v[-1].requires_grad = True
    optimizer = torch.optim.SGD(v, lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(v, lr=lr)
    loss_train = []
    for epoch in range(num_epochs):
        model.eval()
        
        # training
        np.random.seed(epoch)
        idx_list = np.array_split(np.random.permutation(n_tr), num_steps)
        for i in range(num_steps):
            idx = idx_list[i]
            z = model(x_tr[idx])
            loss = loss_fn(z, y_tr[idx])
            model.zero_grad()
            grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            
            vg = 0
            for vv, g in zip(v, grad_params):
                vg += (vv * g).sum()
            model.zero_grad()
            vgrad_params = torch.autograd.grad(vg, model.parameters(), create_graph=True)
            loss_i = 0
            for vgp, vv, uu in zip(vgrad_params, v, u):
                loss_i += 0.5 * (vgp * vv + alpha * vv * vv).sum() - (uu * vv).sum()
                
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()
            loss_train.append(loss_i.item())
            #print(loss_i.item())

    # save
    joblib.dump(np.array(loss_train), hn, compress=9)
    
    # influence
    infl = np.zeros(n_tr)
    for i in range(n_tr):
        z = model(x_tr[[i]])
        loss = loss_fn(z, y_tr[[i]])
        model.zero_grad()
        loss.backward()
        infl_i = 0
        for j, param in enumerate(model.parameters()):
            infl_i += (param.grad.data.cpu().numpy() * v[j].data.cpu().numpy()).sum()
        infl[i] = infl_i / n_tr
        
    # save
    joblib.dump(infl, gn, compress=9)
    

if __name__ == '__main__':
    if args.type == 'true':
        if args.seed >= 0:
            infl_true(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_true(args.target, args.model, seed, args.gpu)
    elif args.type == 'sgd':
        if args.seed >= 0:
            infl_sgd(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_sgd(args.target, args.model, seed, args.gpu)
    elif args.type == 'proposed':
        if args.seed >= 0:
            infl_proposed(args.target, args.model, args.seed, args.gpu, args.simpj)
        else:
            for seed in range(100):
                infl_proposed(args.target, args.model, seed, args.gpu, args.simpj)
    elif args.type == 'nohess':
        if args.seed >= 0:
            infl_nohess(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_nohess(args.target, args.model, seed, args.gpu)
    elif args.type == 'icml':
        if args.seed >= 0:
            infl_icml(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_icml(args.target, args.model, seed, args.gpu)
    