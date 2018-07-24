from os import path
from sklearn import metrics
from surprise import accuracy
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.builtin_datasets import BUILTIN_DATASETS
from sys import stdout

import math
import os
import random

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = '../data'
if not path.exists(data_dir):
  os.makedirs(data_dir)

name = 'ml-100k'
cv = 5
default_param = {
  'n_factors':[100],
  'n_epochs': [20],
  'biased': [True],
  'lr_all': [0.005],
  'reg_all': [0.02],
}
def search():
  data = Dataset.load_builtin(name)
  param_grid = {
    'n_factors':[50, 100, 200],
    'n_epochs': [10, 20, 50],
    'biased': [True, False],
    'lr_all': [0.0005, 0.005, 0.05],
    'reg_all': [0.002, 0.02, 0.2],
  }
  # param_grid = default_param
  gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=cv)
  gs.fit(data)
  print(gs.best_score['rmse'])
  print(gs.best_params['rmse'])
  results_df = pd.DataFrame.from_dict(gs.cv_results)
  results_csv = path.join(data_dir, 'semi_synthetic.csv')
  results_df.to_csv(results_csv)

# search()

max_rate = 5
min_rate = 1
def marginalize(rate_file):
  rate_dist = np.zeros(max_rate)
  with open(rate_file) as fin:
    for line in fin.readlines():
      fields = line.strip().split()
      rate = int(fields[2]) - 1
      rate_dist[rate] += 1.0
  rate_dist /= rate_dist.sum()
  # fig, ax = plt.subplots(1, 1)
  # ax.bar(range(1, 1 + max_rate), rate_dist)
  # eps_file = path.join(data_dir, path.basename(rate_file) + '.eps')
  # fig.savefig(eps_file, format='eps')
  return rate_dist

def evaluate(ui_rates):
  dataset = BUILTIN_DATASETS[name]
  ml_rates = []
  with open(dataset.path) as fin:
    for line in fin.readlines():
      fields = line.strip().split()
      uid, iid = fields[0], fields[1]
      rate = float(fields[2])
      ml_rates.append((uid, iid, rate))
  ml_rates = sorted(ml_rates, key=lambda uir: (uir[0], uir[1]))
  ui_rates = sorted(ui_rates, key=lambda uir: (uir[0], uir[1]))
  print('#ml_rates=%d #ui_rates=%d' % (len(ml_rates), len(ui_rates)))

  rmse, cnt = 0.0, 0
  ml_i, ui_i = 0, 0
  while True:
    if ml_i >= len(ml_rates) or ui_i >= len(ui_rates):
      break
    ml_uid, ml_iid, ml_rate = ml_rates[ml_i]
    ui_uid, ui_iid, ui_rate = ui_rates[ui_i]
    if ml_uid == ui_uid and ml_iid == ui_iid:
      rmse += pow(ml_rate - ui_rate, 2.0)
      cnt += 1
      ml_i += 1
      ui_i += 1
    else:
      ui_i += 1
  rmse /= cnt
  rmse = math.sqrt(rmse)
  print('RMSE: %.8f' % (rmse))

music_dir = path.expanduser('~/Projects/drrec/data/music/')
train_file = 'ydata-ymusic-rating-study-v1_0-train.txt'
test_file = 'ydata-ymusic-rating-study-v1_0-test.txt'
train_file = path.join(music_dir, train_file)
test_file = path.join(music_dir, test_file)
complete_file = path.join(data_dir, 'complete.npy')
def complete():
  data = Dataset.load_builtin(name)
  best_param = {
    'n_factors': 50,
    'n_epochs': 20,
    'biased': True,
    'lr_all': 0.005,
    'reg_all': 0.02,
  }
  trainset = data.build_full_trainset()
  algo = SVD(**best_param)
  algo.fit(trainset)
  testset = trainset.build_testset()
  predictions = algo.test(testset)
  accuracy.rmse(predictions, verbose=True)

  n_users = trainset.n_users
  n_items = trainset.n_items
  print('#user=%d #item=%d' % (n_users, n_items))

  # train_dist = marginalize(train_file)
  test_dist = marginalize(test_file)
  test_cum = np.zeros(max_rate+1)
  for i in range(max_rate):
    test_cum[i+1] = test_dist[i]
  for i in range(max_rate):
    test_cum[i+1] = test_cum[i+1] + test_cum[i]

  ui_rates = []
  for u in trainset.all_users():
    uid = trainset.to_raw_uid(u)
    for i in trainset.all_items():
      iid = trainset.to_raw_iid(i)
      pred = algo.predict(uid, iid, clip=False)
      rate = pred.est
      ui_rates.append((uid, iid, rate))
  evaluate(ui_rates)

  ui_rates = sorted(ui_rates, key=lambda uir: uir[-1])

  n_data = n_users * n_items
  for i in range(max_rate):
    sidx = int(n_data * test_cum[i])
    eidx = int(n_data * test_cum[i+1])
    rate = int(i + 1)
    print('rate=%d [%d, %d)' % (rate, sidx, eidx))
    for j in range(sidx, eidx):
      ui_rates[j] = (ui_rates[j][0], ui_rates[j][1], rate)
  [print('%4s %3s %d'% ui_rates[i*n_items*int(n_users/10)]) for i in range(10)]

  # ui_rates = sorted(ui_rates, key=lambda uir: (uir[0], uir[1]))
  # [print(ui_rate) for ui_rate in ui_rates[:10]]

  r_matrix = np.zeros((n_users, n_items), dtype=int)
  for uid, iid, rate in ui_rates:
    u = trainset.to_inner_uid(uid)
    i = trainset.to_inner_iid(iid)
    r_matrix[u, i] = rate
  print(0 in r_matrix)
  np.save(complete_file, r_matrix)

# complete()

def count_rate(rates):
  rate_cnt = np.zeros(max_rate, dtype=int)
  for rate in rates:
    rate_cnt[rate-1] += 1
  # for rate in range(1, max_rate+1):
  #   stdout.write('#%d=%d ' % (rate, rate_cnt[rate-1]))
  # stdout.write('\n')
  return rate_cnt

def comp_decay(rate, alpha):
  decay = pow(alpha, max(0, 4 - rate))
  return decay

n_users = 943
n_items = 1682
sparsity = 0.05 # 0.0001
def solve_k(rate_cnt, alpha):
  numerator = sparsity * n_users * n_items
  denominator = 0.0
  for rate in range(1, max_rate+1):
    cnt = rate_cnt[rate-1]
    denominator += cnt * comp_decay(rate, alpha)
  k = numerator / denominator
  return k

r_matrix = np.load(complete_file)
# print('max=%.1f min=%.1f' % (r_matrix.max(), r_matrix.min()))
true_rates = r_matrix.flatten()
true_rates = np.sort(true_rates)
aux_idx = [0]
for i in range(len(true_rates)-1):
  if true_rates[i+1] != true_rates[i]:
    aux_idx.append(i+1)
aux_idx.append(len(true_rates))
[stdout.write('%d ' % (i)) for i in aux_idx]
true_cnt = count_rate(true_rates)
true_dist = true_cnt / true_cnt.sum()
[stdout.write('%.4f ' % (p)) for p in true_dist]
stdout.write('\n')
exit()

def mae_true_risk(pred_rates):
  n_rates = len(true_rates)
  # diff = np.absolute(pred_rates - true_rates)
  diff = np.square(pred_rates - true_rates)
  mae = diff.sum() / n_rates
  return mae

def gen_recones(aux_idx):
  recones = np.copy(true_rates)
  ones_idx = list(range(aux_idx[0], aux_idx[1]))
  num_fives = aux_idx[5] - aux_idx[4]
  rnd_idx = np.random.choice(ones_idx, num_fives, replace=False)
  recones[rnd_idx] = 5
  # count_rate(true_rates)
  # count_rate(recones)
  return recones

def gen_recfours(aux_idx):
  recfours = np.copy(true_rates)
  fours_idx = list(range(aux_idx[3], aux_idx[4]))
  num_fives = aux_idx[5] - aux_idx[4]
  rnd_idx = np.random.choice(fours_idx, num_fives, replace=False)
  recfours[rnd_idx] = 5
  return recfours

def gen_rotate():
  rotate = np.copy(true_rates) - 1
  rotate[rotate==1-1] = 5
  return rotate

def gen_skewed():
  sigma = (6.0 - true_rates) / 2.0
  skewed = np.random.normal(true_rates, sigma)
  skewed = np.clip(skewed, 0.0, 6.0)
  return skewed

def gen_coarsened():
  coarsened = np.copy(true_rates)
  coarsened[coarsened<3.5] = 3
  coarsened[coarsened>3.5] = 4
  return coarsened

# alpha = 0.25
alpha = 1.00
def model_observation():
  k = solve_k(true_cnt, alpha)
  obs_dist = np.zeros(max_rate)
  for rate in range(1, 1+max_rate):
    decay = comp_decay(rate, alpha)
    obs_dist[rate-1] = true_cnt[rate-1] * k * decay
  obs_dist /= obs_dist.sum()
  # [stdout.write('%.2f ' % (p + 0.0008)) for p in obs_dist]
  # stdout.write('\n')
  true_propensities = np.maximum(0, 4 - np.copy(true_rates))
  true_propensities = k * np.power(alpha, true_propensities)
  # stdout.write('true propensities: ')
  # [stdout.write('%.4f ' % p) for p in sorted(set(true_propensities))]
  # stdout.write('\n')
  return true_propensities

true_propensities = model_observation()

recones = gen_recones(aux_idx)
recones_mae = mae_true_risk(recones)

recfours = gen_recfours(aux_idx)
recfours_mae = mae_true_risk(recfours)

rotate = gen_rotate()
rotate_mae = mae_true_risk(rotate)

skewed = gen_skewed()
skewed_mae = mae_true_risk(skewed)

coarsened = gen_coarsened()
coarsened_mae = mae_true_risk(coarsened)

def sample_obs():
  obs = np.random.binomial(1, true_propensities)
  return obs

def estimate_n_mae(true_obs, pred_obs):
  n_obs = len(true_obs)
  # true_error = np.absolute(true_obs - pred_obs)
  true_error = np.square(true_obs - pred_obs)
  mae = true_error.sum() / n_obs
  return mae

def estimate_p_mae(true_obs, prop_obs, pred_obs):
  # true_error = np.absolute(true_obs - pred_obs)
  true_error = np.square(true_obs - pred_obs)
  true_error = np.divide(true_error, prop_obs)
  mae = true_error.sum() / (n_users * n_items)
  return mae

def estimate_s_mae(true_obs, prop_obs, pred_obs):
  # true_error = np.absolute(true_obs - pred_obs)
  true_error = np.square(true_obs - pred_obs)
  true_error = np.divide(true_error, prop_obs)
  mae = true_error.sum() / (1.0 / prop_obs).sum()
  return mae

def estimate_d_mae(propensities, pred_rates, train_obs, impute):
  # true_error = np.absolute(true_rates - pred_rates)
  true_error = np.square(true_rates - pred_rates)
  est_error = impute * true_error
  true_mae = np.divide(np.multiply(train_obs, true_error), propensities)
  est_mae = np.divide(np.multiply(propensities-train_obs, est_error), propensities)
  mae = (true_mae + est_mae).sum() / (n_users * n_items)
  return mae

def sample_mcar_rates(n_mcar):
  a = list(range(1, 1+max_rate))
  mcar_rates = np.random.choice(a, n_mcar, p=true_dist)
  return mcar_rates

def est_propensities(true_obs, mcar_rates):
  p_o = len(true_obs) / (n_users * n_items)
  true_obs_cnt = count_rate(true_obs)
  p_r_o = true_obs_cnt / true_obs_cnt.sum()
  # print('p(o=1)=%.4f' % (p_o))
  # [stdout.write('p(r=%d|o=1)=%.4f\n' % (i+1, p)) for i,p in enumerate(p_r_o)]

  #### how to deal with zero count in mcar?
  # mcar_rate_cnt = count_rate(mcar_rates)
  mcar_rate_cnt = count_rate(mcar_rates) + 1e-8
  p_r = mcar_rate_cnt / mcar_rate_cnt.sum()
  # [stdout.write('p(r=%d)=%.4f\n' % (i+1, p)) for i,p in enumerate(p_r)]

  p_o_r = np.divide(p_r_o*p_o, p_r)
  # stdout.write('est propensities: ')
  # [stdout.write('%.4f ' % (p)) for p in p_o_r]
  # stdout.write('\n')

  propensities = np.copy(true_propensities)
  for rate in range(min_rate, min_rate+max_rate):
    propensities[true_rates==rate] = p_o_r[rate-1]

  return propensities

# imputes = [0.0+i*0.1 for i in range(21)]
imputes = [0.8+i*0.2 for i in range(3)]
def evaluate_once(pred_name, pred_rates, true_mae, n_mcar):
  n_trials = 50
  n_maes = np.zeros(n_trials)
  p_maes = np.zeros(n_trials)
  s_maes = np.zeros(n_trials)
  d_mael = [np.zeros(n_trials) for impute in imputes]

  if n_mcar < 0:
    propensities = true_propensities
  for trial in range(n_trials):
    train_obs = sample_obs()
    obs_mask = train_obs == 1

    true_obs = true_rates[obs_mask]
    pred_obs = pred_rates[obs_mask]

    if n_mcar > -1:
      mcar_rates = sample_mcar_rates(n_mcar)
      propensities = est_propensities(true_obs, mcar_rates)

    prop_obs = propensities[obs_mask]

    n_mae = estimate_n_mae(true_obs, pred_obs)
    n_maes[trial] = n_mae

    p_mae = estimate_p_mae(true_obs, prop_obs, pred_obs)
    p_maes[trial] = p_mae

    s_mae = estimate_s_mae(true_obs, prop_obs, pred_obs)
    s_maes[trial] = s_mae

    for i in range(len(imputes)):
      impute = imputes[i]
      d_mae = estimate_d_mae(propensities, pred_rates, train_obs, impute)
      d_mael[i][trial] = d_mae

  n_mean = np.mean(n_maes)
  n_std = np.std(n_maes)
  p_mean = np.mean(p_maes)
  p_std = np.std(p_maes)
  s_mean = np.mean(s_maes)
  s_std = np.std(s_maes)
  
  d_means = [np.mean(d_maes) for d_maes in d_mael]
  d_stds = [np.std(d_maes) for d_maes in d_mael]

  print('%s n=%.4f+%.4f(%.4f) p=%.4f+%.4f(%.4f) s=%.4f+%.4f(%.4f)' % (
      pred_name, 
      n_mae, n_std, abs(true_mae-n_mae), 
      p_mean, p_std, abs(true_mae-p_mean),
      s_mean, s_std, abs(true_mae-s_mean)))
  for impute, d_mean, d_std in zip(imputes, d_means, d_stds):
    stdout.write('  %.1f d=%.4f+%.4f(%.4f)\n' % (
       impute, d_mean, d_std, abs(true_mae-d_mean)))

  true_maes = np.ones(n_trials) * true_mae
  n_mae_mse = metrics.mean_squared_error(true_maes, n_maes)
  p_mae_mse = metrics.mean_squared_error(true_maes, p_maes)
  s_mae_mse = metrics.mean_squared_error(true_maes, s_maes)
  d_mae_mses = [metrics.mean_squared_error(true_maes, d_maes) for d_maes in d_mael]
  n_mae_rmse = math.sqrt(n_mae_mse)
  p_mae_rmse = math.sqrt(p_mae_mse)
  s_mae_rmse = math.sqrt(s_mae_mse)
  d_mae_rmses = np.sqrt(d_mae_mses)
  return n_mae_mse, p_mae_mse, s_mae_mse, d_mae_mses
  # return n_mae_rmse, p_mae_rmse, s_mae_rmse, d_mae_rmses

n_hashtag = 64
def evaluate_five(n_mcar=-1):
  n_cum, p_cum, s_cum = 0.0, 0.0, 0.0
  d_cums = np.zeros(len(imputes))

  print('%s mae=%.3f' % ('recones', recones_mae))
  n, p, s, ds = evaluate_once('recones', recones, recones_mae, n_mcar)
  n_cum += n
  p_cum += p
  s_cum += s
  d_cums += np.asarray(ds)
  print('\n' + '#'*n_hashtag + '\n')

  print('%s mae=%.3f' % ('recfours', recfours_mae))
  n, p, s, ds = evaluate_once('recfours', recfours, recfours_mae, n_mcar)
  n_cum += n
  p_cum += p
  s_cum += s
  d_cums += np.asarray(ds)
  print('\n' + '#'*n_hashtag + '\n')

  print('%s mae=%.3f' % ('rotate', rotate_mae))
  n, p, s, ds = evaluate_once('rotate', rotate, rotate_mae, n_mcar)
  n_cum += n
  p_cum += p
  s_cum += s
  d_cums += np.asarray(ds)
  print('\n' + '#'*n_hashtag + '\n')

  print('%s mae=%.3f' % ('skewed', skewed_mae))
  n, p, s, ds = evaluate_once('skewed', skewed, skewed_mae, n_mcar)
  n_cum += n
  p_cum += p
  s_cum += s
  d_cums += np.asarray(ds)
  print('\n' + '#'*n_hashtag + '\n')

  print('%s mae=%.3f' % ('coarsened', coarsened_mae))
  n, p, s, ds = evaluate_once('coarsened', coarsened, coarsened_mae, n_mcar)
  n_cum += n
  p_cum += p
  s_cum += s
  d_cums += np.asarray(ds)
  print('\n' + '#'*n_hashtag + '\n')

  n_preds = 5
  n_mae_rmse = math.sqrt(n_cum / n_preds)
  p_mae_rmse = math.sqrt(p_cum / n_preds)
  s_mae_rmse = math.sqrt(s_cum / n_preds)
  d_mae_rmses = np.sqrt(d_cums / n_preds)
  # n_mae_rmse = n_cum / n_preds
  # p_mae_rmse = p_cum / n_preds
  # s_mae_rmse = s_cum / n_preds
  # d_mae_rmses = d_cums / n_preds
  print('rmse n=%.4f p=%.4f s=%.4f' % (n_mae_rmse, p_mae_rmse, s_mae_rmse))
  for impute, d_mae_rmse in zip(imputes, d_mae_rmses):
    stdout.write('  %.1f d=%.4f\n' % (impute, d_mae_rmse))

#### when the propensity model is accurate
evaluate_five()
exit()

#### when the propensity model is estimated
n_mcar = 0
print('%s mae=%.3f' % ('recones', recones_mae))
evaluate_once('recones', recones, recones_mae, n_mcar=n_mcar)
print('\n' + '#'*n_hashtag + '\n')
print('%s mae=%.3f' % ('recfours', recfours_mae))
evaluate_once('recfours', recfours, recfours_mae, n_mcar=n_mcar)
print('\n' + '#'*n_hashtag + '\n')
print('%s mae=%.3f' % ('rotate', rotate_mae))
evaluate_once('rotate', rotate, rotate_mae, n_mcar=n_mcar)
print('\n' + '#'*n_hashtag + '\n')
print('%s mae=%.3f' % ('skewed', skewed_mae))
evaluate_once('skewed', skewed, skewed_mae, n_mcar=n_mcar)
print('\n' + '#'*n_hashtag + '\n')
print('%s mae=%.3f' % ('coarsened', coarsened_mae))
evaluate_once('coarsened', coarsened, coarsened_mae,n_mcar=n_mcar)
print('\n' + '#'*n_hashtag + '\n')
exit()

