import numpy as np
import pandas as pd
import scipy.stats as stats
import joblib
from matplotlib import pyplot as plt
from train import *
from sklearn.metrics import mean_squared_error
import math

j_str='simpj' if args.simpj else 'nosimpj'
    
corrupted_str='_corrupted' if args.corrupted else ''
tau = []
jaccard = []

#for seed in range(50):
res_true = joblib.load('../shell/%s%s/%s_%s_%s/infl_true%03d.dat' % (args.datasize,corrupted_str,args.target, args.model, suffix, args.seed))
res_icml = joblib.load('../shell/%s%s/%s_%s_%s/infl_icml%03d.dat' % (args.datasize,corrupted_str,args.target, args.model, suffix, args.seed))
res_nips = joblib.load('../shell/%s%s/%s_%s_%s/infl_sgd%03d.dat' % (args.datasize,corrupted_str,args.target, args.model, suffix, args.seed))
res_proposed = joblib.load('../shell/%s%s/%s_%s_%s/infl_proposed_%s%03d.dat' % (args.datasize,corrupted_str,args.target, args.model,suffix, j_str, args.seed))

corrupted_size=int(args.datasize*0.1)
tau.append((stats.kendalltau(res_true, res_icml)[0], stats.kendalltau(res_true, res_nips)[0], stats.kendalltau(res_true, res_proposed)[0]))
idx1 = np.argsort(res_true)
idx1 = np.r_[idx1[:corrupted_size//2], idx1[-corrupted_size//2:]]
idx2 = np.argsort(res_icml)
idx2 = np.r_[idx2[:corrupted_size//2], idx2[-corrupted_size//2:]]
idx3 = np.argsort(res_nips)
idx3 = np.r_[idx3[:corrupted_size//2], idx3[-corrupted_size//2:]]
idx4 = np.argsort(res_proposed)
idx4 = np.r_[idx4[:corrupted_size//2], idx4[-corrupted_size//2:]]
jaccard.append((np.intersect1d(idx1, idx2).size / np.union1d(idx1, idx2).size, np.intersect1d(idx1, idx3).size / np.union1d(idx1, idx3).size, np.intersect1d(idx1, idx4).size / np.union1d(idx1, idx4).size))

Kendall_tau='Kendall tau: icml, nips, proposed: %s'%(np.mean(tau, axis=0))
Jaccard_index='Jaccard index: icml, nips, proposed: %s'%(np.mean(jaccard, axis=0))

print(Kendall_tau)
print(Jaccard_index)


res_true = joblib.load('../shell/%s%s/%s_%s_%s/infl_true%03d.dat' % (args.datasize,corrupted_str,args.target, args.model, suffix, args.seed))
res_icml = joblib.load('../shell/%s%s/%s_%s_%s/infl_icml%03d.dat' % (args.datasize,corrupted_str,args.target, args.model, suffix, args.seed))
res_nips = joblib.load('../shell/%s%s/%s_%s_%s/infl_sgd%03d.dat' % (args.datasize,corrupted_str,args.target, args.model, suffix, args.seed))
res_proposed = joblib.load('../shell/%s%s/%s_%s_%s/infl_proposed_%s%03d.dat' % (args.datasize,corrupted_str,args.target, args.model,suffix, j_str, args.seed))

mse_icml=mean_squared_error(res_true,res_icml)
rmse_icml=math.sqrt(mse_icml)

mse_nips=mean_squared_error(res_true,res_nips)
rmse_nips=math.sqrt(mse_nips)

mse_proposed=mean_squared_error(res_true,res_proposed)
rmse_proposed=math.sqrt(mse_proposed)


#plt.plot(res_true, res_icml, 'r', color='red',markersize=10)
plt.plot(res_true, res_nips, 'x', color='#6495ED',markersize=4)
plt.plot(res_true, res_proposed, '*', color='orange',markersize=4)
plt.plot(res_true, res_true, 'k--')
plt.tight_layout(rect=[0.05, 0.05, 1, 0.75])  
plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.95) 
plt.legend(['NIPS','Proposed'])
plt.title('icml:%.8f, nips:%.8f, proposed:%.8f \n %s \n %s'%(rmse_icml, rmse_nips, rmse_proposed, Kendall_tau, Jaccard_index))

plt.savefig('../shell/%s%s/%s_%s_%s/compare_%s.png' % (args.datasize,corrupted_str,args.target, args.model, suffix, j_str))
