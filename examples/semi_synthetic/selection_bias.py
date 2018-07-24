from config import alpha_dir, data_dir
from config import n_hashtag

from os import path
from sys import stdout

import utils

import math
import numpy as np
import pickle

utils.create_dir(alpha_dir)

def collect_alpha(alpha, betas, dataset, recom_list, risk):
  d_betas, sd_betas = betas
  n_users, n_items, cmpl_rates, cmpl_cnt = dataset
  risk_name, risk = risk

  out_file = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, alpha))
  if path.isfile(out_file):
    print('%s exists' % (path.basename(out_file)))
    # return

  if alpha <= 0.0:
    alpha += 1e-2

  k = utils.solve_k(alpha, n_users, n_items, cmpl_cnt)
  print('alpha=%.2f k=%.4f' % (alpha, k))
  print('\n' + '#'*n_hashtag + '\n')

  cmpl_props = utils.complete_prop(alpha, k, cmpl_rates)
  # print('#cmpl_prop=%d dtype=%s' % (len(cmpl_props), cmpl_props.dtype))

  n_risk_cum, p_risk_cum, sp_risk_cum = 0.0, 0.0, 0.0
  d_risk_cum, sd_risk_cum = 0.0, 0.0
  d_risk_cums = np.zeros(len(d_betas))
  sd_risk_cums = np.zeros(len(sd_betas))

  for recom in recom_list:
    recom_name, pred_rates = recom
    # if recom_name != 'coarsened':
    #   continue

    t_risk = utils.compute_t(pred_rates, cmpl_rates, risk)
    print('%s %s t=%.4f' % (risk_name, recom_name, t_risk))
    dataset = n_users, n_items, cmpl_rates, cmpl_cnt, t_risk
    res = utils.evaluate_est(recom, dataset, cmpl_props, betas, risk)
    n_risk_mse, p_risk_mse, sp_risk_mse, d_risk_mses, sd_risk_mses = res
    print('\n' + '#'*n_hashtag + '\n')

    n_risk_cum += n_risk_mse
    p_risk_cum += p_risk_mse
    sp_risk_cum += sp_risk_mse

    d_risk_cum += min(d_risk_mses)
    d_risk_cums += np.asarray(d_risk_mses)

    if len(sd_risk_mses) > 0:
      sd_risk_cum += min(sd_risk_mses)
      sd_risk_cums += np.asarray(sd_risk_mses)

  n_recoms = len(recom_list)
  n_risk_rmse = math.sqrt(n_risk_cum / n_recoms)
  p_risk_rmse = math.sqrt(p_risk_cum / n_recoms)
  sp_risk_rmse = math.sqrt(sp_risk_cum / n_recoms)
  d_risk_rmse = math.sqrt(d_risk_cum / n_recoms)
  sd_risk_rmse = math.sqrt(sd_risk_cum / n_recoms)
  d_risk_rmses = np.sqrt(d_risk_cums / n_recoms)
  sd_risk_rmses = np.sqrt(sd_risk_cums / n_recoms)

  print('%s rmse n=%.4f p=%.4f sp=%.4f' % 
      (risk_name, n_risk_rmse, p_risk_rmse, sp_risk_rmse))
  for d_beta, d_mae_rmse in zip(d_betas, d_risk_rmses):
    print('  d_beta=%.1f d=%.4f' % (d_beta, d_mae_rmse))
  for sd_beta, sd_mae_rmse in zip(sd_betas, sd_risk_rmses):
    print('  sd_beta=%.1f sd=%.4f' % (sd_beta, sd_mae_rmse))
  print('%s rmse d=%.4f sd=%.4f' % (risk_name, d_risk_rmse, sd_risk_rmse))
  print('\n' + '#'*n_hashtag + '\n')

  if path.isfile(out_file):
    print('do not override %s' % (path.basename(out_file)))
    return
  data = {
    'n': n_risk_rmse,
    'p': p_risk_rmse, 'sp': sp_risk_rmse,
    'd': d_risk_rmse, 'sd': sd_risk_rmse,
  }
  pickle.dump(data, open(out_file, 'wb'))

n_users, n_items, indexes = utils.read_data()
print('#user=%d #item=%d' % (n_users, n_items))
# [stdout.write('%d ' % (idx)) for idx in indexes]
# stdout.write('\n')

cmpl_rates = utils.complete_rate(indexes)
# print('#cmpl_rate=%d dtype=%s' % (len(cmpl_rates), cmpl_rates.dtype))

cmpl_cnt = utils.count_index(indexes)
cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
# [stdout.write('%.4f ' % (p)) for p in cmpl_dist]
# stdout.write('\n')

alphas = np.arange(0.10, 1.05, 0.10)
d_betas = np.arange(0.00, 0.05, 0.10)
# sd_betas = np.arange(0.10, 0.85, 0.10)
# d_betas = np.arange(0.00, 2.05, 0.10)
sd_betas = np.arange(0.10, 0.05, 0.10)
betas = d_betas, sd_betas
dataset = (n_users, n_items, cmpl_rates, cmpl_cnt)
recom_list = utils.provide_recom(indexes, cmpl_rates)

for alpha in alphas:
  risk = 'mae', np.absolute
  collect_alpha(alpha, betas, dataset, recom_list, risk)
  risk = 'mse', np.square
  collect_alpha(alpha, betas, dataset, recom_list, risk)
  stdout.flush()
