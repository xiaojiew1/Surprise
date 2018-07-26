from os import path
from sklearn import metrics
from sys import stdout

import math
import numpy as np
import os

def make_dir(dirpath):
  if not path.exists(dirpath):
    os.makedirs(dirpath)

def make_file_dir(filepath):
  dirpath = path.dirname(filepath)
  make_dir(dirpath)

def read_data(infile):
  with open(infile) as fin:
    line = fin.readline()
    n_users, n_items, n_rates = [int(f) for f in line.split()]

    line = fin.readline()
    indexes = [int(f) for f in line.split()]
  return n_users, n_items, n_rates, indexes

def complete_rate(indexes):
  cmpl_rates = []
  for rate in range(min_rate, min_rate+max_rate):
    n_rates = indexes[rate] - indexes[rate-1]
    cmpl_rates.append(np.ones(n_rates, dtype=int) * rate)
  cmpl_rates = np.concatenate(cmpl_rates)
  return cmpl_rates

def provide_recones(indexes, cmpl_rates):
  recones = np.copy(cmpl_rates)
  ones_idx = list(range(indexes[0], indexes[1]))
  num_fives = indexes[5] - indexes[4]
  rnd_idx = np.random.choice(ones_idx, num_fives, replace=False)
  recones[rnd_idx] = 5
  return recones

def provide_recfours(indexes, cmpl_rates):
  recfours = np.copy(cmpl_rates)
  fours_idx = list(range(indexes[3], indexes[4]))
  num_fives = indexes[5] - indexes[4]
  rnd_idx = np.random.choice(fours_idx, num_fives, replace=False)
  recfours[rnd_idx] = 5
  return recfours

def provide_rotate(cmpl_rates):
  rotate = np.copy(cmpl_rates) - 1
  rotate[rotate==1-1] = 5
  return rotate

def provide_skewed(cmpl_rates):
  sigma = (6.0 - cmpl_rates) / 2.0
  skewed = np.random.normal(cmpl_rates, sigma)
  skewed = np.clip(skewed, 0.0, 6.0)
  return skewed

def provide_coarsened(cmpl_rates):
  coarsened = np.copy(cmpl_rates)
  coarsened[coarsened<2.5] = 1
  coarsened[coarsened>2.5] = 5
  #### icml2016
  # coarsened[coarsened<3.5] = 3
  # coarsened[coarsened>3.5] = 4
  return coarsened

def provide_recom(indexes, cmpl_rates):
  recom_list = [
    ('recones', provide_recones(indexes, cmpl_rates)),
    ('recfours', provide_recfours(indexes, cmpl_rates)),
    ('rotate', provide_rotate(cmpl_rates)),
    ('skewed', provide_skewed(cmpl_rates)),
    ('coarsened', provide_coarsened(cmpl_rates)),
  ]
  return recom_list

def count_index(indexes):
  rate_cnt = np.zeros(max_rate, dtype=int)
  for rate in range(min_rate, min_rate+max_rate):
    rate_cnt[rate-1] = indexes[rate] - indexes[rate-1]
  return rate_cnt

def compute_decay(rate, alpha):
  #### ml100k
  # decay = pow(alpha, max(0, 4-rate))
  # return decay

  if rate == max_rate:
    return 1.0
  decay = pow(alpha, min(4, 6-rate))
  return decay

def solve_k(alpha, n_users, n_items, n_rates, rate_cnt):
  sparsity = n_rates / (n_users * n_items)
  numerator = sparsity * n_users * n_items
  denominator = 0.0
  for rate in range(min_rate, min_rate+max_rate):
    decay = compute_decay(rate, alpha)
    denominator += rate_cnt[rate-1] * decay
  k = numerator / denominator
  return k

def cmpt_propensity(rate, alpha, k):
  decay = compute_decay(rate, alpha)
  return k * decay

def complete_prop(alpha, k, indexes):
  cmpl_props = []
  for rate in range(min_rate, min_rate+max_rate):
    n_rates = indexes[rate] - indexes[rate-1]
    prop = cmpt_propensity(rate, alpha, k)
    cmpl_props.append(np.ones(n_rates, dtype=int) * prop)
  cmpl_props = np.concatenate(cmpl_props)
  return cmpl_props

def compute_t(pred_rates, true_rates, risk):
  n_rates = len(true_rates)
  true_errors = risk(pred_rates - true_rates)
  return true_errors.sum() / n_rates

def sample_train(cmpl_props):
  train_obs = np.random.binomial(1, cmpl_props)
  return train_obs

def estimate_n(cmpl_rates, pred_rates, train_obs, risk):
  true_errors = risk(pred_rates - cmpl_rates)
  true_errors = np.multiply(true_errors, train_obs)
  return true_errors.sum() / train_obs.sum()

def p(cmpl_rates, pred_rates, train_obs, propensities, risk):
  true_errors = risk(pred_rates - cmpl_rates)
  true_errors = np.divide(true_errors, propensities)
  true_errors = np.multiply(true_errors, train_obs)
  return true_errors

def estimate_p(cmpl_rates, pred_rates, train_obs, propensities, risk):
  true_errors = p(cmpl_rates, pred_rates, train_obs, propensities, risk)
  return true_errors.sum() / len(cmpl_rates)

def estimate_s(cmpl_rates, pred_rates, train_obs, propensities, risk):
  true_errors = p(cmpl_rates, pred_rates, train_obs, propensities, risk)
  normalizer = np.multiply(1.0 / propensities, train_obs).sum()
  return true_errors.sum() / normalizer

def d(cmpl_rates, pred_rates, train_obs, propensities, beta, risk):
  true_errors = risk(pred_rates - cmpl_rates)
  true_errors = np.multiply(train_obs, true_errors)
  true_errors = np.divide(true_errors, propensities)

  #### true error
  # pred_errors = beta * true_errors
  #### mean error
  # pred_errors = beta * np.mean(true_errors) * np.ones_like(true_errors)
  #### mean rate
  pred_errors = risk(pred_rates - np.mean(cmpl_rates))
  # beta = true_errors.sum() / risk(pred_rates - np.mean(cmpl_rates)).sum()
  pred_errors *= beta
  # print('pred_beta=%.4f' % (pred_beta))

  pred_errors = np.multiply(propensities-train_obs, pred_errors)
  pred_errors = np.divide(pred_errors, propensities)
  tot_errors = true_errors + pred_errors
  return tot_errors

def estimate_d(cmpl_rates, pred_rates, train_obs, propensities, beta, risk):
  tot_errors = d(cmpl_rates, pred_rates, train_obs, propensities, beta, risk)
  return tot_errors.sum() / len(cmpl_rates)

def format_float(f):
  if -1.0 < f < 1.0:
    f = '%.4f' % f
    f = f.replace('0', '', 1)
  else:
    f = ('%.3f' % f)
  return f

def evaluate_est(recom, dataset, cmpl_props, risk, betas=None, gamma=0.0):
  recom_name, pred_rates = recom
  n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk = dataset
  t_rates = n_users * n_items
  # print('#user=%d #item=%d #rating=%d' % (n_users, n_items, n_rates))

  n_risks = np.zeros(n_trials)
  p_risks = np.zeros(n_trials)
  s_risks = np.zeros(n_trials)
  d_risks = np.zeros(n_trials)

  cmpl_mean = np.mean(cmpl_rates)
  beta = risk(pred_rates-cmpl_rates).sum() / risk(pred_rates-cmpl_mean).sum()
  for trial in range(n_trials):
    train_obs = sample_train(cmpl_props)
    propensities = (gamma * train_obs.sum() / t_rates) * np.ones(t_rates)
    propensities += (1 - gamma) * np.copy(cmpl_props)

    n_risk = estimate_n(cmpl_rates, pred_rates, train_obs, risk)
    n_risks[trial] = n_risk

    p_risk = estimate_p(cmpl_rates, pred_rates, train_obs, propensities, risk)
    p_risks[trial] = p_risk

    sp_risk = estimate_s(cmpl_rates, pred_rates, train_obs, propensities, risk)
    s_risks[trial] = sp_risk

    d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, propensities, beta, risk)
    d_risks[trial] = d_risk

  n_mean = abs(np.mean(n_risks) - t_risk)
  n_std = np.std(n_risks)
  p_mean = abs(np.mean(p_risks) - t_risk)
  p_std = np.std(p_risks)
  s_mean = abs(np.mean(s_risks) - t_risk)
  s_std = np.std(s_risks)
  d_mean = abs(np.mean(d_risks) - t_risk)
  d_std = np.std(d_risks)

  # stdout.write('%s t=%s\n' % (recom_name, format_float(t_risk)))
  # stdout.write(' n=%s+%s' % (format_float(n_risk), format_float(n_std)))
  stdout.write('%s & %.3f' % (recom_name.upper(), t_risk))
  stdout.write(' & %s$\\pm$%s' % (format_float(p_mean), format_float(p_std)))
  stdout.write(' & %s$\\pm$%s' % (format_float(s_mean), format_float(s_std)))
  stdout.write(' & %s$\\pm$%s' % (format_float(d_mean), format_float(d_std)))
  stdout.write('\n')

  t_risks = np.ones(n_trials) * t_risk
  n_risk_mse = metrics.mean_squared_error(t_risks, n_risks)
  p_risk_mse = metrics.mean_squared_error(t_risks, p_risks)
  sp_risk_mse = metrics.mean_squared_error(t_risks, s_risks)
  d_risk_mse = metrics.mean_squared_error(t_risks, d_risks)
  return n_risk_mse, p_risk_mse, sp_risk_mse, d_risk_mse

def cmpt_bias(alpha, dataset, recom_list, risk):
  n_users, n_items, n_rates, indexes, cmpl_rates= dataset
  risk_name, risk = risk

  cmpl_cnt = count_index(indexes)
  # stdout.write('cnt:')
  # [stdout.write(' %07d' % (c)) for c in cmpl_cnt]
  # stdout.write('\n')

  k = solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
  print('alpha=%.2f k=%.4f' % (alpha, k))
  # mnar_dist = np.zeros(max_rate)
  # for rate in range(min_rate, min_rate+max_rate):
  #   decay = compute_decay(rate, alpha)
  #   mnar_dist[rate-1] = decay * cmpl_cnt[rate-1]
  # mnar_dist /= mnar_dist.sum()
  # stdout.write('obs:')
  # [stdout.write(' %.4f' % p) for p in mnar_dist]
  # stdout.write('\n')

  cmpl_props = complete_prop(alpha, k, indexes)
  # rp_set = set()
  # for rate, prop in zip(cmpl_rates, cmpl_props):
  #   rp_set.add('%d %.8f' % (rate, prop))
  # props = np.zeros(max_rate)
  # n_mnar = 0
  # for rp in rp_set:
  #   fields = rp.split()
  #   rate = int(fields[0])
  #   prop = float(fields[1])
  #   n_mnar += prop * cmpl_cnt[rate-1]
  # print('#mnar=%d' % n_mnar)

  for recom in recom_list:
    recom_name, pred_rates = recom
    # if recom_name != 'coarsened':
    #   continue

    t_risk = compute_t(pred_rates, cmpl_rates, risk)
    dataset = n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk
    res = evaluate_est(recom, dataset, cmpl_props, risk)

min_rate = 1
max_rate = 5
n_trials = 50
data_dir = 'data'
song_file = path.join(data_dir, 'song.txt')

if __name__ == '__main__':
  n_users, n_items, n_rates, indexes = read_data(song_file)
  # print('#user=%d #item=%d #rating=%d' % (n_users, n_items, n_rates))
  # stdout.write('cum:')
  # [stdout.write(' %07d' % index) for index in indexes]
  # stdout.write('\n')

  cmpl_rates = complete_rate(indexes)

  dataset = n_users, n_items, n_rates, indexes, cmpl_rates
  recom_list = provide_recom(indexes, cmpl_rates)

  alpha = 0.50
  #### ml100k
  # alpha = 0.25

  risk = 'mae', np.absolute
  cmpt_bias(alpha, dataset, recom_list, risk)

  risk = 'mse', np.square
  cmpt_bias(alpha, dataset, recom_list, risk)


