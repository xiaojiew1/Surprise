from config import song_file, omega_dir
from config import f_alpha, n_hashtag

from os import path
from sys import stdout

import config

import math
import numpy as np
import pickle

utils.create_dir(beta_dir)

def vary_omega(alpha, omegas, dataset, recom_list, risk):
  n_users, n_items, n_rates, indexes, cmpl_rates= dataset
  risk_name, risk = risk
  cmpl_cnt = config.count_index(indexes)
  cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
  k = config.solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
  cmpl_props = config.complete_prop(alpha, k, indexes)

  n_risk_cum, p_risk_cum, sp_risk_cum = 0.0, 0.0, 0.0
  d_risk_cum, sd_risk_cum = 0.0, 0.0
  d_risk_cums = np.zeros(len(d_betas))
  sd_risk_cums = np.zeros(len(sd_betas))

  n_rmse, p_rmse, s_rmse, d_rmse = 0.0, 0.0, 0.0, 0.0
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

  beta_rmse = {
    'n': n_risk_rmse,
    'p': p_risk_rmse, 'sp': sp_risk_rmse,
    'betas': d_betas,'rmses': d_risk_rmses,
  }
  out_file = path.join(beta_dir, '%s_%.1f.p' % (risk_name, alpha))
  pickle.dump(beta_rmse, open(out_file, 'wb'))

#### load data
n_users, n_items, n_rates, indexes = config.read_data(song_file)
cmpl_rates = config.complete_rate(indexes)
dataset = n_users, n_items, n_rates, indexes, cmpl_rates
recom_list = config.provide_recom(indexes, cmpl_rates)

alpha = f_alpha

risk = 'mae', np.absolute
omegas = np.arange(0.00, 3.25, 0.10)
vary_omega(alpha, omegas, dataset, recom_list, risk)
risk = 'mse', np.square
omegas = np.arange(0.00, 4.85, 0.10)
vary_omega(alpha, omegas, dataset, recom_list, risk)

