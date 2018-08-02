from config import tmp_dir
from util_coat import trainset, testset
from surprise import accuracy
from surprise import MFREC

from os import path

import config

import itertools
import numpy as np
import time

#### default
n_factors_opt = [100,]
n_epochs_opt = [20,]
biased_opt = [True,]
reg_all_opt = [0.02,]
lr_all_opt = [0.005,]

n_factors_opt = [100,]
n_epochs_opt = [20,]
biased_opt = [True, False]
reg_all_opt = [0.02,]
lr_all_opt = [0.005,]

mae_bst, mse_bst, kwargs_bst = np.inf, np.inf, {}
st_time = time.time()
for n_factors, n_epochs, biased, reg_all, lr_all in itertools.product(
    n_factors_opt, n_epochs_opt, biased_opt, reg_all_opt, lr_all_opt):
  algo_kwargs = {
    'n_factors': n_factors,
    'n_epochs': n_epochs,
    'biased': biased,
    'reg_all': reg_all,
    'lr_all': lr_all,
    # 'verbose': True,
  }

  algo = MFREC(**algo_kwargs)
  algo.fit(trainset)

  predictions = algo.test(testset)

  eval_kwargs = {'verbose':False}
  mae = accuracy.mae(predictions, **eval_kwargs)
  mse = pow(accuracy.rmse(predictions, **eval_kwargs), 2.0)
  kwargs_str = config.stringify(algo_kwargs)
  print('%.4f %.4f %s' % (mae, mse, kwargs_str))

  if mse < mse_bst:
    mae_bst = min(mae, mae_bst)
    mse_bst = min(mse, mse_bst)
    kwargs_bst = algo_kwargs

kwargs_bst = config.stringify(kwargs_bst)
print('%.4f %.4f %s' % (mae_bst, mse_bst, kwargs_bst))

e_time = time.time()
# print('%.2fs' % (e_time - st_time))









