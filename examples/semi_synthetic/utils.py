from config import data_file, music_train, music_test
from config import min_rate, max_rate, sparsity, n_trials

from os import path
from scipy.spatial import distance
from sklearn import metrics
from sys import stdout

import numpy as np
import os

def create_dir(out_dir):
  if not path.exists(out_dir):
    os.makedirs(out_dir)

def read_data():
  with open(data_file) as fin:
    line = fin.readline()
    fields = line.strip().split()
    n_users, n_items = int(fields[0]), int(fields[1])

    line = fin.readline()
    indexes = [int(idx) for idx in line.strip().split()]
  return n_users, n_items, indexes

def complete_rate(indexes):
  cmpl_rates = []
  for rate in range(min_rate, min_rate+max_rate):
    n_rates = indexes[rate] - indexes[rate-1]
    cmpl_rates.append(np.ones(n_rates, dtype=int) * rate)
  cmpl_rates = np.concatenate(cmpl_rates)
  return cmpl_rates

def count_index(indexes):
  rate_cnt = np.zeros(max_rate, dtype=int)
  for rate in range(min_rate, min_rate+max_rate):
    rate_cnt[rate-1] = indexes[rate] - indexes[rate-1]
  return rate_cnt

def compute_decay(rate, alpha):
  decay = pow(alpha, max(0, 4-rate))
  return decay

def solve_k(alpha, n_users, n_items, rate_cnt):
  numerator = sparsity * n_users * n_items
  denominator = 0.0
  for rate in range(min_rate, min_rate+max_rate):
    denominator += rate_cnt[rate-1] * compute_decay(rate, alpha)
  k = numerator / denominator
  return k

def complete_prop(alpha, k, cmpl_rates):
  cmpl_props = np.maximum(0, 4-np.copy(cmpl_rates))
  cmpl_props = k * np.power(alpha, cmpl_props)
  return cmpl_props

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

def estimate_sp(cmpl_rates, pred_rates, train_obs, propensities, risk):
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

def estimate_sd(cmpl_rates, pred_rates, train_obs, propensities, beta, risk):
  tot_errors = d(cmpl_rates, pred_rates, train_obs, propensities, beta, risk)
  normalizer = np.multiply(1.0 / propensities, train_obs).sum()
  return tot_errors.sum() / normalizer

def evaluate_est(recom, dataset, cmpl_props, betas, risk, gamma=0.0):
  recom_name, pred_rates = recom
  d_betas, sd_betas = betas
  n_users, n_items, cmpl_rates, cmpl_cnt, t_risk = dataset
  n_rates = len(cmpl_rates)

  n_risks = np.zeros(n_trials)
  p_risks = np.zeros(n_trials)
  sp_risks = np.zeros(n_trials)
  d_riskl = [np.zeros(n_trials) for d_beta in d_betas]
  sd_riskl = [np.zeros(n_trials) for sd_beta in sd_betas]

  cmpl_mean = np.mean(cmpl_rates)
  beta = risk(pred_rates - cmpl_rates).sum() / risk(pred_rates - cmpl_mean).sum()

  for trial in range(n_trials):
    train_obs = sample_train(cmpl_props)
    propensities = (gamma * train_obs.sum() / n_rates) * np.ones(n_rates)
    propensities += (1 - gamma) * np.copy(cmpl_props)

    n_risk = estimate_n(cmpl_rates, pred_rates, train_obs, risk)
    n_risks[trial] = n_risk

    p_risk = estimate_p(cmpl_rates, pred_rates, train_obs, propensities, risk)
    p_risks[trial] = p_risk

    sp_risk = estimate_sp(cmpl_rates, pred_rates, train_obs, propensities, risk)
    sp_risks[trial] = sp_risk

    for i in range(max(len(d_betas), len(sd_betas))):
      d_beta = d_betas[i]
      # d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, propensities, d_beta, risk)
      d_risk = estimate_d(cmpl_rates, pred_rates, train_obs, propensities, beta, risk)
      d_riskl[i][trial] = d_risk
      if i < len(sd_betas):
        sd_beta = sd_betas[i]
        sd_risk = estimate_sd(cmpl_rates, pred_rates, train_obs, propensities, sd_beta, risk)
        sd_riskl[i][trial] = sd_risk

  n_mean = np.mean(n_risks)
  n_std = np.std(n_risks)
  p_mean = np.mean(p_risks)
  p_std = np.std(p_risks)
  sp_mean = np.mean(sp_risks)
  sp_std = np.std(sp_risks)
  
  d_means = [np.mean(d_maes) for d_maes in d_riskl]
  d_stds = [np.std(d_maes) for d_maes in d_riskl]
  sd_means = [np.mean(sd_maes) for sd_maes in sd_riskl]
  sd_stds = [np.std(sd_maes) for sd_maes in sd_riskl]

  print('%s n=%.4f+%.4f(%.4f) p=%.4f+%.4f(%.4f) sp=%.4f+%.4f(%.4f)' % (
      recom_name, n_risk, n_std, abs(t_risk-n_risk), 
      p_mean, p_std, abs(t_risk-p_mean),
      sp_mean, sp_std, abs(t_risk-sp_mean)))
  for d_beta, d_mean, d_std in zip(d_betas, d_means, d_stds):
    d_diff = abs(t_risk - d_mean)
    print('  d_beta=%.1f d=%.4f+%.4f(%.4f)' % (d_beta, d_mean, d_std, d_diff))
  for sd_beta, sd_mean, sd_std in zip(sd_betas, sd_means, sd_stds):
    sd_diff = abs(t_risk-sd_mean)
    print('  sd_beta=%.1f sd=%.4f+%.4f(%.4f)' % (sd_beta, sd_mean, sd_std, sd_diff))

  t_risks = np.ones(n_trials) * t_risk
  n_risk_mse = metrics.mean_squared_error(t_risks, n_risks)
  p_risk_mse = metrics.mean_squared_error(t_risks, p_risks)
  sp_risk_mse = metrics.mean_squared_error(t_risks, sp_risks)
  d_risk_mses = [metrics.mean_squared_error(t_risks, d_maes) for d_maes in d_riskl]
  sd_risk_mses = [metrics.mean_squared_error(t_risks, sd_maes) for sd_maes in sd_riskl]
  return n_risk_mse, p_risk_mse, sp_risk_mse, d_risk_mses, sd_risk_mses

def marginalize(rate_file):
  rate_dist = np.zeros(max_rate)
  with open(rate_file) as fin:
    for line in fin.readlines():
      fields = line.strip().split()
      rate = int(fields[2]) - min_rate
      rate_dist[rate] += 1.0
  rate_dist /= rate_dist.sum()
  return rate_dist

def observe_dist(alpha, test_dist):
  obs_dist = np.ones_like(test_dist)
  for rate in range(min_rate, min_rate+max_rate):
    # factor = max(max_rate-rate, 0)
    # factor = max(4-rate, 0)
    # factor = max(rate-2, 0)
    factor = min(6-rate, 4) if rate < max_rate else 1.0
    obs_dist[rate-1] = test_dist[rate-1] * pow(alpha, factor)
  obs_dist = obs_dist / obs_dist.sum()
  return obs_dist

if __name__ == '__main__':
  # read_data()

  train_dist = marginalize(music_train)

  test_dist = marginalize(music_test)
  stdout.write('%5s:' % ('test'))
  [stdout.write(' %.4f' % (p*5400*1000)) for p in test_dist]
  stdout.write('\n')

  stdout.write('%5s:' % 'ratio')
  for train_p, test_p in zip(train_dist, test_dist):
    stdout.write(' %.4f' % (train_p/test_p))
  stdout.write('\n')

  bst_alpha, min_error = np.inf, np.inf
  for alpha in np.arange(0.10, 0.90, 0.005):
    obs_dist = observe_dist(alpha, test_dist)
    error = distance.euclidean(obs_dist, train_dist)
    if error < min_error:
      bst_alpha = alpha
      min_error = error
    # stdout.write('%5s:' % ('train'))
    # [stdout.write(' %.4f' % (p)) for p in train_dist]
    # stdout.write('\n')
    # stdout.write('%5s:' % ('obs'))
    # [stdout.write(' %.4f' % (p)) for p in obs_dist]
    # stdout.write('\n')
  bst_alpha = 0.40
  stdout.write('%5s:' % ('train'))
  [stdout.write(' %.4f' % (p)) for p in train_dist]
  stdout.write('\n')
  stdout.write('%5s: [' % ('train'))
  [stdout.write('%.2f,' % (p)) for p in train_dist]
  stdout.write(']\n')
  obs_dist = observe_dist(bst_alpha, test_dist)
  stdout.write('%.3f:' % (bst_alpha))
  [stdout.write(' %.4f' % (p)) for p in obs_dist]
  stdout.write('\n')
  stdout.write('%.3f: [' % (bst_alpha))
  [stdout.write('%.2f,' % (p)) for p in obs_dist]
  stdout.write(']\n')

