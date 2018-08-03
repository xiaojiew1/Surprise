from config import song_file, gamma_dir
from config import n_hashtag, f_alpha, mae_v_gamma, mse_v_gamma

from os import path
from sys import stdout

import config

import math
import numpy as np
import pickle

def vary_gamma(alpha, gammas, dataset, recom_list, risk):
  n_users, n_items, n_rates, indexes, cmpl_rates= dataset
  risk_name, risk = risk
  cmpl_cnt = config.count_index(indexes)
  cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
  k = config.solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
  cmpl_props = config.complete_prop(alpha, k, indexes)

  outfile = path.join(gamma_dir, '%s_%.1f.p' % (risk_name, alpha))
  # if path.isfile(outfile):
  #   print('%s exists' % (path.basename(outfile)))
  #   return

  n_rmse, p_rmse, s_rmse = 0.0, 0.0, 0.0
  d_rmses = np.zeros(len(gammas))
  for recom in recom_list:
    recom_name, pred_rates = recom
    t_risk = config.compute_t(pred_rates, cmpl_rates, risk)
    dataset = n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk

    res = config.eval_wt_gamma(recom, dataset, cmpl_props, (risk_name, risk), gammas)
    n_mse, p_mse, s_mse, d_mses = res

    n_rmse += n_mse
    p_rmse += p_mse
    s_rmse += s_mse
    d_rmses += d_mses
  n_recoms = len(recom_list)
  n_rmse = math.sqrt(n_rmse / n_recoms)
  p_rmse = math.sqrt(p_rmse / n_recoms)
  s_rmse = math.sqrt(s_rmse / n_recoms)
  d_rmses = np.sqrt(d_rmses / n_recoms)

  print('%s alpha=%.1f k=%.4f' % (risk_name, alpha, k))
  print('  n=%.4f p=%.4f s=%.4f' % (n_rmse, p_rmse, s_rmse))
  for gamma, d_rmse in zip(gammas, d_rmses):
    print('  gamma=%.1f d=%.4f' % (gamma, d_rmse))
  print('\n' + '#'*n_hashtag + '\n')

  return
  config.make_file_dir(outfile)
  data = {
    'a': alpha,
    'k': k,
    'n': n_rmse,
    'p': p_rmse,
    's': s_rmse,
    'd': d_rmses,
  }
  pickle.dump(data, open(outfile, 'wb'))

#### load data
n_users, n_items, n_rates, indexes = config.read_data(song_file)
cmpl_rates = config.complete_rate(indexes)
dataset = n_users, n_items, n_rates, indexes, cmpl_rates
recom_list = config.provide_recom(indexes, cmpl_rates)

alpha = f_alpha

risk = 'mae', np.absolute
gammas = mae_v_gamma
# gammas = np.arange(-2.00, -1.25, 0.50)
gammas = np.arange(4.00, 4.25, 0.50)
vary_gamma(alpha, gammas, dataset, recom_list, risk)
risk = 'mse', np.square
gammas = mse_v_gamma
# gammas = np.arange(-2.00, -1.25, 0.50)
gammas = np.arange(4.00, 4.25, 0.50)
vary_gamma(alpha, gammas, dataset, recom_list, risk)



