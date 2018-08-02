from config import song_file, beta_dir
from config import n_hashtag, f_alpha, v_beta

from os import path
from sys import stdout

import config

import math
import numpy as np
import pickle

def given_beta(alpha, beta, dataset, recom_list, risk):
  n_users, n_items, n_rates, indexes, cmpl_rates= dataset
  risk_name, risk = risk

  outfile = path.join(beta_dir, '%s_%.1f.p' % (risk_name, beta))
  if path.isfile(outfile):
    print('%s exists' % (path.basename(outfile)))
    # return

  cmpl_cnt = config.count_index(indexes)
  cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
  k = config.solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
  cmpl_props = config.complete_prop(alpha, k, indexes)

  n_rmse, p_rmse, s_rmse, d_rmse = 0.0, 0.0, 0.0, 0.0
  for recom in recom_list:
    recom_name, pred_rates = recom
    t_risk = config.compute_t(pred_rates, cmpl_rates, risk)
    dataset = n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk

    max_try = 1
    n_mses, p_mses, s_mses, d_mses = [], [], [], []
    for i in range(max_try):
      res = config.eval_wo_omega(recom, dataset, cmpl_props, (risk_name, risk), beta=beta)
      n_mse, p_mse, s_mse, d_mse, rerun = res
      # print('rerun%d %s %s p=%.8f s=%.8f d=%.8f' % (i+1, risk_name, recom_name, p_mse, s_mse, d_mse))
      n_mses.append(n_mse)
      p_mses.append(p_mse)
      s_mses.append(s_mse)
      d_mses.append(d_mse)

    d_minus_s, min_idx = d_mses[0] - s_mses[0], 0
    for i in range(1, max_try):
      if d_mses[i] - s_mses[i] < d_minus_s:
        d_minus_s, min_idx = d_mses[i] - s_mses[i], i
    i = min_idx
    n_mse, p_mse, s_mse, d_mse = n_mses[i], p_mses[i], s_mses[i], d_mses[i]
    print('select %s %s p=%.8f s=%.8f d=%.8f' % (risk_name, recom_name, p_mse, s_mse, d_mse))

    n_rmse += n_mse
    p_rmse += p_mse
    s_rmse += s_mse
    d_rmse += d_mse
  n_recoms = len(recom_list)
  n_rmse = math.sqrt(n_rmse / n_recoms)
  p_rmse = math.sqrt(p_rmse / n_recoms)
  s_rmse = math.sqrt(s_rmse / n_recoms)
  d_rmse = math.sqrt(d_rmse / n_recoms)

  print('%s alpha=%.1f k=%.4f beta=%.1f' % (risk_name, alpha, k, beta))
  print('  n=%.4f p=%.4f s=%.4f d=%.4f' % (n_rmse, p_rmse, s_rmse, d_rmse))
  print('\n' + '#'*n_hashtag + '\n')

  config.make_file_dir(outfile)
  data = {
    'a': alpha,
    'k': k,
    'b': beta,
    'n': n_rmse,
    'p': p_rmse,
    's': s_rmse,
    'd': d_rmse,
  }
  pickle.dump(data, open(outfile, 'wb'))

#### load data
n_users, n_items, n_rates, indexes = config.read_data(song_file)
cmpl_rates = config.complete_rate(indexes)
dataset = n_users, n_items, n_rates, indexes, cmpl_rates
recom_list = config.provide_recom(indexes, cmpl_rates)

alpha = f_alpha
betas = v_beta
# betas = np.flip(v_beta)

for beta in betas:
  risk = 'mae', np.absolute
  given_beta(alpha, beta, dataset, recom_list, risk)
  risk = 'mse', np.square
  given_beta(alpha, beta, dataset, recom_list, risk)
  stdout.flush()


