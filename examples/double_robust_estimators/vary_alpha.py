from config import song_file, alpha_dir, error_dir
from config import n_hashtag, v_alpha
from config import min_rate, max_rate

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
n_mcar = 50
risk = 'mae', np.absolute
alpha = f_alpha
n_users, n_items, n_rates, indexes, cmpl_rates= dataset
risk_name, risk = risk
cmpl_cnt = config.count_index(indexes)
cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
k = config.solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)

cmpl_props = config.complete_prop(alpha, k, indexes)
# [stdout.write('%.4f ' % p) for p in set(cmpl_props)]
# stdout.write('\n')

p_o = n_rates / (n_users * n_items)
# print('p_o: %.4f' % p_o)
p_r = np.copy(cmpl_dist)
stdout.write('p_r:')
[stdout.write(' %.4f' % p) for p in p_r]
stdout.write('\n')
p_o_r = config.compute_prop(alpha, k)
stdout.write('p_o_r:')
[stdout.write(' %.4f' % p) for p in p_o_r]
stdout.write('\n')
p_r_o = p_o_r * p_r / p_o
# stdout.write('p_r_o:')
# [stdout.write(' %.4f' % p) for p in p_r_o]
# stdout.write('\n')
np.random.seed(0)
while True:
  mcar_rates = np.random.choice(max_rate-min_rate+1, n_mcar, p=list(p_r))
  p_r = np.zeros(max_rate-min_rate+1)
  for rid in mcar_rates:
    p_r[rid] += 1
  p_r /= p_r.sum()
  success = True
  if p_r.min() == 0.0:
    success = False
  if p_r[0] <= p_r[1]:
    success = False
  if p_r[1] <= p_r[2]:
    success = False
  if p_r[2] <= p_r[3]:
    success = False
  if p_r[3] <= p_r[4]:
    success = False
  if success:
    break
stdout.write('p_r:')
[stdout.write(' %.4f' % p) for p in p_r]
stdout.write('\n')
p_o_r = p_r_o * p_o / p_r
stdout.write('p_o_r:')
[stdout.write(' %.4f' % p) for p in p_o_r]
stdout.write('\n')

rate_props = config.complete_prop(alpha, k, indexes, rate_props=p_o_r)
[stdout.write('%.4f ' % p) for p in set(rate_props)]
stdout.write('\n')

e_rmses, d_rmses, omegas = [], [], []
for omega in np.arange(0.0, 1.05, 0.1):
  e_rmse = 0.0
  d_rmse = 0.0
  for recom in recom_list:
    recom_name, pred_rates = recom
    t_risk = config.compute_t(pred_rates, cmpl_rates, risk)
    dataset = n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk

    res = config.eval_wt_mcar(recom, dataset, cmpl_props, rate_props, (risk_name, risk), omega)
    e_mse, d_mse = res

    e_rmse += e_mse
    d_rmse += d_mse
  n_recoms = len(recom_list)
  e_rmse = math.sqrt(e_rmse / n_recoms)
  d_rmse = math.sqrt(d_rmse / n_recoms)
  print('%s alpha=%.1f k=%.4f' % (risk_name, alpha, k))
  print('  e=%.4f d=%.4f' % (e_rmse, d_rmse))
  print('\n' + '#'*n_hashtag + '\n')
  e_rmses.append(e_rmse)
  d_rmses.append(d_rmse)
  omegas.append(omega)
  break
data = {
  'e': e_rmses,
  'd': d_rmses,
  'o': omegas,
}
outfile = path.join(error_dir, '%s_%03d.p' % (risk_name, n_mcar))
config.make_file_dir(outfile)
pickle.dump(data, open(outfile, 'wb'))
exit()
alphas = v_alpha
print('\n' + '#'*n_hashtag + '\n')
for alpha in alphas:
  risk = 'mae', np.absolute
  given_alpha(alpha, dataset, recom_list, risk)
  risk = 'mse', np.square
  given_alpha(alpha, dataset, recom_list, risk)
  stdout.flush()
