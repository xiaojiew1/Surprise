from config import rec_song_file
from util_song import trainset, testset
from surprise import accuracy
from surprise import MFREC

from os import path
from sys import stdout

import config

import itertools
import numpy as np
import operator
import time

#### default
n_factors_opt = [100,]
n_epochs_opt = [20,]
biased_opt = [True,]
reg_all_opt = [0.02,]
lr_all_opt = [0.005,]

#### gsearch
n_factors_opt = [16, 32, 64, 128, 256]
n_epochs_opt = [16, 32, 64, 128, 256]
biased_opt = [True, False]
reg_all_opt = [0.005, 0.01, 0.05, 0.1, 0.5]
lr_all_opt = [0.0005, 0.001, 0.005, 0.01, 0.05]

#### develop
# n_factors_opt = [100,]
# n_epochs_opt = [20,]
# biased_opt = [True, False,]
# reg_all_opt = [0.02,]
# lr_all_opt = [0.005,]

gsearch_file = rec_song_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)

s_time = time.time()
for n_factors, n_epochs, biased, reg_all, lr_all in itertools.product(
    n_factors_opt, n_epochs_opt, biased_opt, reg_all_opt, lr_all_opt):
  algo_kwargs = {
    'n_factors': n_factors,
    'n_epochs': n_epochs,
    'biased': biased,
    'reg_all': reg_all,
    'lr_all': lr_all,
  }
  kwargs_str = config.stringify(algo_kwargs)
  if kwargs_str in kwargs_set:
    continue

  algo = MFREC(**algo_kwargs)
  algo.fit(trainset)

  predictions = algo.test(testset)
  eval_kwargs = {'verbose':False}
  mae = accuracy.mae(predictions, **eval_kwargs)
  mse = pow(accuracy.rmse(predictions, **eval_kwargs), 2.0)
  print('%.4f %.4f %s' % (mae, mse, kwargs_str))
  stdout.flush()
  err_kwargs.append((mae, mse, kwargs_str))

e_time = time.time()
print('%.2fs' % (e_time - s_time))
err_kwargs = sorted(err_kwargs, key=operator.itemgetter(1,0))
config.write_gsearch(err_kwargs, gsearch_file)








