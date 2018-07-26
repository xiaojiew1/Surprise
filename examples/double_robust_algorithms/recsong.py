from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import MFIPS
from surprise import SVD

from os import path
from sys import stdout

import itertools
import io
import numpy as np
import requests
import shutil
import time
import zipfile

dnld_dir = path.expanduser('~/Downloads')
data_dir = path.join(dnld_dir, 'Webscope_R3')
if not path.exists(data_dir):
  raise Exception('the song dataset does not exist')
train_rawfile = path.join(data_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
train_file = path.join(data_dir, 'train.txt')
test_rawfile = path.join(data_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
test_file = path.join(data_dir, 'test.txt')
propensity_file = path.join(data_dir, 'propensities.txt')
rating_scale = (1, 5)
min_rate, max_rate = rating_scale

if not path.isfile(train_file):
  shutil.copy(train_rawfile, train_file)

if not path.isfile(test_file) or not path.isfile(propensity_file):
  shutil.copy(train_rawfile, train_file)
  test_ratings = []
  with open(test_rawfile) as fin:
    for line in fin.readlines():
      u, i, r = [int(f) for f in line.split()]
      test_ratings.append((u, i, r))
  np.random.shuffle(test_ratings)
  n_test = len(test_ratings)
  n_est_prop = int(0.05 * n_test)
  prop_ratings = test_ratings[:n_est_prop]
  test_ratings = test_ratings[n_est_prop:]
  with open(test_file, 'w') as fout:
    for u, i, r in test_ratings:
      fout.write('%d\t%d\t%d\n' % (u, i, r))
  propensities = np.zeros(max_rate)
  for u, i, r in prop_ratings:
    propensities[r-1] += 1
  with open(propensity_file, 'w') as fout:
    for r in range(max_rate):
      fout.write('%d' % propensities[r])
      if r < max_rate - 1:
        fout.write(' ')
      else:
        fout.write('\n')
  propensities /= propensities.sum()
  stdout.write('propensities')
  [stdout.write(' %.4f' % p) for p in propensities]
  stdout.write('\n')

reader = Reader(line_format='user item rating', sep='\t')
data = Dataset(reader=reader, rating_scale=rating_scale)
raw_trainset = data.read_ratings(train_file)
raw_testset = data.read_ratings(test_file)
trainset = data.construct_trainset(raw_trainset)
testset = data.construct_testset(raw_testset)

#### default
n_factors_opt = [100,]
n_epochs_opt = [20,]
biased_opt = [True,]
reg_all_opt = [0.02,]
# n_factors_opt = [10, 20, 50, 100, 200,]
# n_epochs_opt = [10, 20, 50, 100, 200,]
# biased_opt = [True, False,]
# reg_all_opt = [0.001, 0.005, 0.01, 0.05, 0.1]
n_factors_opt = [200,]
n_epochs_opt = [200,]
biased_opt = [False,]
reg_all_opt = [0.1,]
#### develop
n_factors_opt = [50,]
n_epochs_opt = [200,]
biased_opt = [False,]
reg_all_opt = [0.1,]
var_all_opt = [0.0001]

mae_bst, mse_bst, kwargs_bst = np.inf, np.inf, None
st_time = time.time()
for n_factors, n_epochs, biased, reg_all, var_all in itertools.product(
    n_factors_opt, n_epochs_opt, biased_opt, reg_all_opt, var_all_opt):
  algo_kwargs = {
    'n_factors': n_factors,
    'n_epochs': n_epochs,
    'biased': biased,
    'reg_all': reg_all,
    # 'var_all': var_all,
    'verbose': True,
  }
  algo = SVD(**algo_kwargs)
  algo.fit(trainset)
  # algo = SVD(**algo_kwargs)
  # algo.fit(trainset, weights)
  predictions = algo.test(testset)

  eval_kwargs = {'verbose':False}
  mae = accuracy.mae(predictions, **eval_kwargs)
  mse = pow(accuracy.rmse(predictions, **eval_kwargs), 2.0)
  print('var_all=%.6f mae=%.4f mse=%.4f' % (var_all, mae, mse))

  if mse < mse_bst:
    mae_bst = min(mae, mae_bst)
    mse_bst = min(mse, mse_bst)
    kwargs_bst = algo_kwargs

print('best mae=%.4f mse=%.4f' % (mae_bst, mse_bst))
[print('%10s: %s' % (k, v)) for k, v in kwargs_bst.items()]
e_time = time.time()
print('%.2fs' % (e_time - st_time))









