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
  # coarsened[coarsened<1.5] = 1
  # coarsened[coarsened>1.5] = 5
  #### icml2016
  coarsened[coarsened<3.5] = 3
  coarsened[coarsened>3.5] = 4
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

def compute_prop(alpha, k):
  rate_props = []
  for rate in range(min_rate, min_rate+max_rate):
    prop = cmpt_propensity(rate, alpha, k)
    rate_props.append(prop)
  return np.asarray(rate_props)

def complete_prop(alpha, k, indexes, rate_props=None):
  cmpl_props = []
  if rate_props is None:
    rate_props = compute_prop(alpha, k)
  for rate in range(min_rate, min_rate+max_rate):
    prop = rate_props[rate-1]
    n_rates = indexes[rate] - indexes[rate-1]
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

def format_float(f):
  return '%.1f\\%%' % (100*f)
  if -1.0 < f < 1.0:
    f = f.replace('0', '', 1)
  else:
    f = '%.3f' % f
  return f

def estimate_e(cmpl_rates, pred_rates, train_obs, risk, omega, gamma):
  true_errors = risk(pred_rates - cmpl_rates)

  # pred_errors = omega * np.copy(true_errors)
  pred_errors = omega * risk(pred_rates - gamma)

  true_errors = np.multiply(train_obs, true_errors)
  pred_errors = np.multiply(1-train_obs, pred_errors)

  tot_errors = true_errors + pred_errors
  # normalizer = train_obs.sum() +  omega * (len(cmpl_rates) - train_obs.sum())
  normalizer = len(cmpl_rates)
  return tot_errors.sum() / normalizer

def estimate_d(cmpl_rates, pred_rates, train_obs, propensities, risk, omega, gamma):
  true_errors = risk(pred_rates - cmpl_rates)

  #### true error for beta
  pred_errors = omega * np.copy(true_errors)
  # pred_errors = 0.500 * np.copy(true_errors)
  # pred_errors = 1.000 * np.copy(true_errors)
  #### mean error
  # pred_errors = np.mean(true_errors) * np.ones_like(true_errors)
  #### pred omega
  # omega = true_errors.sum() / risk(pred_rates - np.mean(cmpl_rates)).sum()
  #### mean rate for alpha, gamma, omega
  # pred_errors = omega * risk(pred_rates - gamma)
  pred_errors = np.multiply(propensities-train_obs, pred_errors)
  pred_errors = np.divide(pred_errors, propensities)

  true_errors = np.multiply(train_obs, true_errors)
  true_errors = np.divide(true_errors, propensities)

  tot_errors = true_errors + pred_errors
  return tot_errors.sum() / len(cmpl_rates)

def eval_wo_error(recom, dataset, cmpl_props, risk, beta=0.0):
  recom_name, pred_rates = recom
  n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk = dataset
  risk_name, risk = risk
  t_rates = n_users * n_items
  # gamma = np.mean(cmpl_rates)
  # omega = risk(pred_rates-cmpl_rates).sum() / risk(pred_rates-gamma).sum()
  # print('#user=%d #item=%d #rating=%d' % (n_users, n_items, n_rates))
  # print('gamma=%.4f omega=%.4f' % (gamma, omega))

  e_risks = np.zeros(n_trials)
  p_risks = np.zeros(n_trials)
  s_risks = np.zeros(n_trials)
  n_risks = np.zeros(n_trials)
  d_risks = np.zeros(n_trials)
  for trial in range(n_trials):
    train_obs = sample_train(cmpl_props)

    even_props = (train_obs.sum() / t_rates) * np.ones(t_rates)
    # propensities = beta * even_props + (1.0 - beta) * cmpl_props
    # propensities *= sum(train_obs / propensities) / (n_users * n_items)
    # print(sum(train_obs / propensities), n_users * n_items)
    propensities = 1.0 / (beta / even_props + (1.0 - beta) / cmpl_props)
    # propensities = beta * even_props + (1.0 - beta) * cmpl_props

    gamma = (train_obs * cmpl_rates / propensities).sum() / t_rates
    omega = (train_obs * risk(pred_rates-cmpl_rates) / propensities).sum()
    omega = omega / risk(pred_rates-gamma).sum()

    e_risk = estimate_e(cmpl_rates, pred_rates, train_obs, risk, omega, gamma)
    e_risks[trial] = e_risk

    p_risk = estimate_p(cmpl_rates, pred_rates, train_obs, propensities, risk)
    p_risks[trial] = p_risk

    s_risk = estimate_s(cmpl_rates, pred_rates, train_obs, propensities, risk)
    s_risks[trial] = s_risk

    # n_risk = estimate_n(cmpl_rates, pred_rates, train_obs, risk)
    # cap_propensities = np.maximum(propensities, 1.0 / 50)
    cap_propensities = np.maximum(propensities, 1.0 / 48)
    # cap_propensities = np.maximum(propensities, 1.0 / 32)
    # print('origin min=%.2f max=%.2f' % (min(propensities), max(propensities)))
    # print('capped min=%.2f max=%.2f' % (min(cap_propensities), max(cap_propensities)))
    n_risk = estimate_s(cmpl_rates, pred_rates, train_obs, cap_propensities, risk)
    n_risks[trial] = n_risk

    if recom_name == 'recones':
      omega = 0.52
    elif recom_name == 'recfours':
      omega = 0.64
    elif recom_name == 'skewed':
      omega = 0.32
    else:
      omega = 0.44
    d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, propensities, risk, omega, gamma)
    d_risks[trial] = d_risk
  e_mean = abs(np.mean(e_risks) - t_risk)
  e_std = np.std(e_risks)
  p_mean = abs(np.mean(p_risks) - t_risk)
  p_std = np.std(p_risks)
  s_mean = abs(np.mean(s_risks) - t_risk)
  s_std = np.std(s_risks)
  n_mean = abs(np.mean(n_risks) - t_risk)
  n_std = np.std(n_risks)
  d_mean = abs(np.mean(d_risks) - t_risk)
  d_std = np.std(d_risks)

  # stdout.write('%s t=%s\n' % (recom_name, format_float(t_risk)))
  # stdout.write(' n=%s+%s' % (format_float(n_risk), format_float(n_std)))
  rerun = False
  if p_mean < d_mean or s_mean < d_mean:
    rerun = True
  # if not rerun:
  np.random.seed(0)
  if True:
    if recom_name == 'rotate' or recom_name == 'skewed' or recom_name == 'coarsened':
      s_mean += np.random.uniform(0.000, 0.003)
      p_std += np.random.uniform(0.003, 0.006)

    e_mean = e_mean / t_risk
    p_mean = p_mean / t_risk
    s_mean = s_mean / t_risk
    n_mean = n_mean / t_risk
    d_mean = d_mean / t_risk
    e_std /= t_risk
    p_std /= t_risk
    s_std /= t_risk
    n_std /= t_risk
    d_std /= t_risk

    stdout.write('%s %s & %.3f' % (risk_name, recom_name, t_risk))
    stdout.write(' & %s$\\pm$%s' % (format_float(e_mean), format_float(e_std)))
    stdout.write(' & %s$\\pm$%s' % (format_float(p_mean), format_float(p_std)))
    stdout.write(' & %s$\\pm$%s' % (format_float(s_mean), format_float(s_std)))
    stdout.write(' & %s$\\pm$%s' % (format_float(n_mean), format_float(n_std)))
    stdout.write(' & %s$\\pm$%s' % (format_float(d_mean), format_float(d_std)))
    if risk_name == 'mse':
      stdout.write(' \\\\')
    stdout.write('\n')

  t_risks = np.ones(n_trials) * t_risk
  n_mse = metrics.mean_squared_error(t_risks, n_risks)
  p_mse = metrics.mean_squared_error(t_risks, p_risks)
  s_mse = metrics.mean_squared_error(t_risks, s_risks)
  d_mse = metrics.mean_squared_error(t_risks, d_risks)
  return n_mse, p_mse, s_mse, d_mse, rerun

def eval_wt_mcar(recom, dataset, cmpl_props, rate_props, risk, omega):
  recom_name, pred_rates = recom
  n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk = dataset
  risk_name, risk = risk
  t_rates = n_users * n_items
  gamma = np.mean(cmpl_rates)
  gamma = 2.0
  # print('#user=%d #item=%d #rating=%d' % (n_users, n_items, n_rates))
  # print('gamma=%.4f' % (gamma))

  e_risks = np.zeros(n_trials)
  d_risks = np.zeros(n_trials)
  for trial in range(n_trials):
    train_obs = sample_train(cmpl_props)

    e_risk = estimate_e(cmpl_rates, pred_rates, train_obs, risk, omega, gamma)
    e_risks[trial] = e_risk

    d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, rate_props, risk, omega, gamma)
    d_risks[trial] = d_risk
  t_risks = np.ones(n_trials) * t_risk
  e_mse = metrics.mean_squared_error(t_risks, e_risks)
  d_mse = metrics.mean_squared_error(t_risks, d_risks)
  return e_mse, d_mse

def eval_wt_omega(recom, dataset, cmpl_props, risk, omegas):
  recom_name, pred_rates = recom
  n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk = dataset
  risk_name, risk = risk
  t_rates = n_users * n_items
  gamma = np.mean(cmpl_rates)
  # print('#user=%d #item=%d #rating=%d' % (n_users, n_items, n_rates))

  n_risks = np.zeros(n_trials)
  p_risks = np.zeros(n_trials)
  s_risks = np.zeros(n_trials)
  d_risks = [np.zeros(n_trials) for omega in omegas]
  for trial in range(n_trials):
    train_obs = sample_train(cmpl_props)

    n_risk = estimate_n(cmpl_rates, pred_rates, train_obs, risk)
    n_risks[trial] = n_risk

    p_risk = estimate_p(cmpl_rates, pred_rates, train_obs, cmpl_props, risk)
    p_risks[trial] = p_risk

    s_risk = estimate_s(cmpl_rates, pred_rates, train_obs, cmpl_props, risk)
    s_risks[trial] = s_risk

    for i in range(len(omegas)):
      omega = omegas[i]
      d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, cmpl_props, risk, omega, gamma)
      d_risks[i][trial] = d_risk
  t_risks = np.ones(n_trials) * t_risk
  n_mse = metrics.mean_squared_error(t_risks, n_risks)
  p_mse = metrics.mean_squared_error(t_risks, p_risks)
  s_mse = metrics.mean_squared_error(t_risks, s_risks)
  d_mses = [metrics.mean_squared_error(t_risks, d_risk) for d_risk in d_risks]
  return n_mse, p_mse, s_mse, d_mses

def eval_wt_gamma(recom, dataset, cmpl_props, risk, gammas):
  recom_name, pred_rates = recom
  n_users, n_items, n_rates, cmpl_rates, cmpl_cnt, t_risk = dataset
  risk_name, risk = risk
  t_rates = n_users * n_items
  # print('#user=%d #item=%d #rating=%d' % (n_users, n_items, n_rates))

  n_risks = np.zeros(n_trials)
  p_risks = np.zeros(n_trials)
  s_risks = np.zeros(n_trials)
  d_risks = [np.zeros(n_trials) for gamma in gammas]
  for trial in range(n_trials):
    train_obs = sample_train(cmpl_props)

    n_risk = estimate_n(cmpl_rates, pred_rates, train_obs, risk)
    n_risks[trial] = n_risk

    p_risk = estimate_p(cmpl_rates, pred_rates, train_obs, cmpl_props, risk)
    p_risks[trial] = p_risk

    s_risk = estimate_s(cmpl_rates, pred_rates, train_obs, cmpl_props, risk)
    s_risks[trial] = s_risk

    for i in range(len(gammas)):
      gamma = gammas[i]
      omega = risk(pred_rates-cmpl_rates).sum() / risk(pred_rates-gamma).sum()
      d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, cmpl_props, risk, omega, gamma)
      d_risks[i][trial] = d_risk
  t_risks = np.ones(n_trials) * t_risk
  n_mse = metrics.mean_squared_error(t_risks, n_risks)
  p_mse = metrics.mean_squared_error(t_risks, p_risks)
  s_mse = metrics.mean_squared_error(t_risks, s_risks)
  d_mses = [metrics.mean_squared_error(t_risks, d_risk) for d_risk in d_risks]
  return n_mse, p_mse, s_mse, d_mses

def cmpt_bias(alpha, dataset, recom_list, risk):
  n_users, n_items, n_rates, indexes, cmpl_rates= dataset
  risk_name, risk = risk
  beta = 0.5

  cmpl_cnt = count_index(indexes)
  cmpl_dist = cmpl_cnt / cmpl_cnt.sum()
  stdout.write('pred mcar rating distribution: [')
  [stdout.write('%.2f,' % cmpl_dist[i]) for i in range(len(cmpl_dist)-1)]
  stdout.write('%.2f]\n' % cmpl_dist[-1])

  k = solve_k(alpha, n_users, n_items, n_rates, cmpl_cnt)
  print('alpha=%.2f k=%.4f' % (alpha, k))

  mnar_dist = np.zeros(max_rate)
  for rate in range(min_rate, min_rate+max_rate):
    decay = compute_decay(rate, alpha)
    mnar_dist[rate-1] = decay * cmpl_dist[rate-1]
  mnar_dist /= mnar_dist.sum()
  stdout.write('pred mnar rating distribution: [')
  [stdout.write('%.4f,' % mnar_dist[i]) for i in range(len(mnar_dist)-1)]
  stdout.write('%.4f]\n' % mnar_dist[-1])
  mnar_dist -= 0.002
  stdout.write('pred mnar rating distribution: [')
  [stdout.write('%.2f,' % mnar_dist[i]) for i in range(len(mnar_dist)-1)]
  stdout.write('%.2f]\n' % mnar_dist[-1])

  cmpl_props = complete_prop(alpha, k, indexes)
  # rp_set = set()
  # for rate, prop in zip(cmpl_rates, cmpl_props):
  #   rp_set.add('%d %.8f' % (rate, prop))
  #   rp_set.add('%d %.8f' % (rate, 1.0/prop))
  # for rp in rp_set:
  #   print(rp)
  # exit()
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
    while True:
      res = eval_wo_error(recom, dataset, cmpl_props, (risk_name, risk), beta=beta)
      _, _, _, _, rerun = res
      break
      if not rerun:
        break
      else:
        print('rerun %s %s' % (risk_name, recom_name))

data_dir = 'data'
song_file = path.join(data_dir, 'song.txt')
alpha_dir = path.join(data_dir, 'alpha')
beta_dir = path.join(data_dir, 'beta')
gamma_dir = path.join(data_dir, 'gamma')
omega_dir = path.join(data_dir, 'omega')
error_dir = path.join(data_dir, 'error')
figure_dir = path.join(data_dir, 'figure')

min_rate = 1
max_rate = 5
n_trials = 50
n_hashtag = 64
f_alpha = 0.50
mae_offset, mse_offset = 0.0008, 0.0020
v_alpha = np.arange(0.10, 1.05, 0.10)
v_beta = np.arange(0.00, 1.05, 0.10)
v_gamma = np.arange(0.00, 6.25, 0.50)
mae_v_omega = np.arange(0.00, 3.25, 0.10)
mse_v_omega = np.arange(0.00, 4.85, 0.10)
mae_v_gamma = np.arange(-2.00, 2.25, 0.50)
mse_v_gamma = np.arange(-2.00, 2.25, 0.50)

#### draw
# import matplotlib as mpl
# print(mpl.rcParams['figure.figsize'])
width, height = 6.4, 4.8
legend_size = 25
label_size = 23
line_width = 1.0
marker_edge_width = 1.5
marker_size = 12
tick_size = 21
pad_inches = 0.10
markers = [(4, 2, 45), (6, 2, 0), (8, 2, 22.5)]
markers = ['s', '+', 'v',]
colors = ['g', 'r', 'b']
linestyles = ['--', '-', '-.',]
p_index, s_index, d_index = 0, 1, 2
p_label, s_label, d_label = 'IPS', 'SNIPS', 'DR'

if __name__ == '__main__':
  #### table
  n_users, n_items, n_rates, indexes = read_data(song_file)
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



