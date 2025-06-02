import numpy as np
import joblib
from scipy import stats
from matplotlib import pyplot as plt
from train import *
from sklearn.metrics import mean_squared_error
import math

# choose suffix for proposed vs. nosimpj
j_str = f'simpj{args.simpj}_' if args.simpj else 'nosimpj_'
corrupted_str=f'_corrupted{args.corruption_sigma}' if args.corruption_sigma else ''
noise_str=f'_noise{args.noise_rate}' if args.noise_rate else ''

# load true, NIPS, and proposed influence scores
base_path = '../shell/{}{}{}/{}_{}_{}/'.format(
    args.datasize, corrupted_str, noise_str,
    args.target, args.model, suffix
)
res_true     = joblib.load(base_path + 'infl_true%03d.dat' % args.seed)
res_nips     = joblib.load(base_path + 'infl_sgd%03d.dat'  % args.seed)
res_proposed = joblib.load(base_path + 'infl_proposed_%s%03d.dat' % (j_str, args.seed))

# --- Kendall’s tau and Jaccard index over top/bottom corrupted samples ---
tau_vals = []
jacc_vals = {}
corrupted_size = int(args.datasize * 0.1)

# sort-and-select indices for "true", "NIPS", and "proposed"
idx_true = np.argsort(res_true)
idx_true = np.r_[idx_true[:corrupted_size//2], idx_true[-corrupted_size//2:]]
idx_nips = np.argsort(res_nips)
idx_nips = np.r_[idx_nips[:corrupted_size//2], idx_nips[-corrupted_size//2:]]
idx_prop = np.argsort(res_proposed)
idx_prop = np.r_[idx_prop[:corrupted_size//2], idx_prop[-corrupted_size//2:]]

# compute Kendall's tau
tau_vals.append((
    stats.kendalltau(res_true, res_nips)[0],
    stats.kendalltau(res_true, res_proposed)[0]
))

# calculate Jaccard values for 70%, 50%, 30%, and 10% of top/bottom corrupted samples
percentiles = [70, 50, 30, 10]
for p in percentiles:
    corrupted_size = int(args.datasize * p / 100)
    
    idx_true_p = np.argsort(res_true)
    idx_true_p = np.r_[idx_true_p[:corrupted_size//2], idx_true_p[-corrupted_size//2:]]
    
    idx_nips_p = np.argsort(res_nips)
    idx_nips_p = np.r_[idx_nips_p[:corrupted_size//2], idx_nips_p[-corrupted_size//2:]]
    
    idx_prop_p = np.argsort(res_proposed)
    idx_prop_p = np.r_[idx_prop_p[:corrupted_size//2], idx_prop_p[-corrupted_size//2:]]
    
    jacc_vals[p] = (
        np.intersect1d(idx_true_p, idx_nips_p).size / np.union1d(idx_true_p, idx_nips_p).size,
        np.intersect1d(idx_true_p, idx_prop_p).size / np.union1d(idx_true_p, idx_prop_p).size
    )

# Store Kendall's tau and Jaccard values in a readable format
Kendall_tau   = 'Kendall tau (NIPS, Proposed): %s' % np.mean(tau_vals, axis=0)
Jaccard_index = ''
for k, v in jacc_vals.items():
    Jaccard_index += 'Jaccard index (NIPS, Proposed) (%d%%): %s\n' % (k, (round(v[0], 4), round(v[1], 4)))

# --- RMSE comparison ---
rmse_nips     = math.sqrt(mean_squared_error(res_true, res_nips))
rmse_proposed = math.sqrt(mean_squared_error(res_true, res_proposed))

# --- Scatter plot ---
plt.figure(figsize=(6,6))
plt.plot(res_true, res_nips,     'x', markersize=4, label='NIPS')
plt.plot(res_true, res_proposed, '*', markersize=4, label='Proposed')
plt.plot(res_true, res_true,     'k--', linewidth=1)
plt.legend()
plt.title(
    'RMSE NIPS: %.6f  |  Proposed: %.6f\n%s\n%s'
    % (rmse_nips, rmse_proposed, Kendall_tau, Jaccard_index)
)
#x轴 Real Loss Change y轴Estimated Loss Change
plt.xlabel('Real Loss Change')
plt.ylabel('Estimated Loss Change')
#x轴和y轴都用科学计数法
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig(base_path + 'compare_%s_%03d.png' % (j_str,args.seed))

# Save results to a text file
with open(base_path + 'compare_results_%s_%03d.txt' % (j_str,args.seed), 'w') as f:
    f.write(f'RMSE NIPS: {rmse_nips}\n')
    f.write(f'RMSE Proposed: {rmse_proposed}\n')
    f.write(f'{Kendall_tau}\n')
    f.write(f'{Jaccard_index}\n')
    f.write('\nJaccard Index for different percentages:\n')
    for p in percentiles:
        f.write(f'{p}%: NIPS: {jacc_vals[p][0]}, Proposed: {jacc_vals[p][1]}\n')
