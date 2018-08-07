from config import song_file, alpha_dir
from config import n_hashtag, v_alpha

from os import path
from sys import stdout

import config

import math
import numpy as np
import pickle

def given_alpha(alpha, dataset, recom_list, risk):
  n_users, n_items, n_rates, indexes, cmpl_rates= dataset
  risk_name, risk = risk

  outfile = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, alpha))
  # if path.isfile(outfile):
  #   print('%s exists' % (path.basename(outfile)))
  #   return

  cmpl_cnt = config.count_index(indexes)
  cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
  k = config.solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
  cmpl_props = config.complete_prop(alpha, k, indexes)

  n_rmse, p_rmse, s_rmse, d_rmse = 0.0, 0.0, 0.0, 0.0
  for recom in recom_list:
    recom_name, pred_rates = recom
    t_risk = config.compute_t(pred_rates, cmpl_rates, risk)
    dataset = n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk

    while True:
      res = config.eval_wo_error(recom, dataset, cmpl_props, (risk_name, risk))
      n_mse, p_mse, s_mse, d_mse, rerun = res
      if not rerun:
        break
      else:
        print('rerun %s %s' % (risk_name, recom_name))

    n_rmse += n_mse
    p_rmse += p_mse
    s_rmse += s_mse
    d_rmse += d_mse
  n_recoms = len(recom_list)
  n_rmse = math.sqrt(n_rmse / n_recoms)
  p_rmse = math.sqrt(p_rmse / n_recoms)
  s_rmse = math.sqrt(s_rmse / n_recoms)
  d_rmse = math.sqrt(d_rmse / n_recoms)

  print('%s alpha=%.1f k=%.4f' % (risk_name, alpha, k))
  print('  n=%.4f p=%.4f s=%.4f d=%.4f' % (n_rmse, p_rmse, s_rmse, d_rmse))
  print('\n' + '#'*n_hashtag + '\n')

  return
  config.make_file_dir(outfile)
  data = {
    'a': alpha,
    'k': k,
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

from config import f_alpha
omega = 1.0
risk = 'mae', np.absolute
alpha = f_alpha
n_users, n_items, n_rates, indexes, cmpl_rates= dataset
risk_name, risk = risk
cmpl_cnt = config.count_index(indexes)
cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
k = config.solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
cmpl_props = config.complete_prop(alpha, k, indexes)
betas = [0.0, 1.0,]

e_rmse = 0.0
d_rmses = np.zeros(len(betas))
for recom in recom_list:
  recom_name, pred_rates = recom
  t_risk = config.compute_t(pred_rates, cmpl_rates, risk)
  dataset = n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk

  res = config.eval_wt_beta(recom, dataset, cmpl_props, (risk_name, risk), omega, betas)
  e_mse, d_mses = res

  e_rmse += e_mse
  d_rmses += d_mses
n_recoms = len(recom_list)
e_rmse = math.sqrt(e_rmse / n_recoms)
d_rmses = np.sqrt(d_rmses / n_recoms)
print('%s alpha=%.1f k=%.4f' % (risk_name, alpha, k))
print('  e=%.4f' % (e_rmse))
for beta, d_rmse in zip(betas, d_rmses):
  print('  beta=%.1f d=%.4f' % (beta, d_rmse))
print('\n' + '#'*n_hashtag + '\n')

exit()
alphas = v_alpha
print('\n' + '#'*n_hashtag + '\n')
for alpha in alphas:
  risk = 'mae', np.absolute
  given_alpha(alpha, dataset, recom_list, risk)
  risk = 'mse', np.square
  given_alpha(alpha, dataset, recom_list, risk)
  stdout.flush()
