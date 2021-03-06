from config import tune_coat_file
from config import coat_n_epochs

from util_coat import trainset, testset
from surprise import accuracy
from surprise import MFREC

from os import path
from sys import stdout

import config

import itertools
import numpy as np
import operator
import time

def reg_coat(alg_kwargs, err_kwargs):
  kwargs_file = config.get_coat_file(alg_kwargs)
  if path.isfile(kwargs_file):
    return
  config.make_file_dir(kwargs_file)

  alg_kwargs = config.min_kwargs(alg_kwargs, err_kwargs)
  alg_kwargs['eval_space'] = trainset.n_ratings
  alg_kwargs['kwargs_file'] = kwargs_file
  alg_kwargs['n_epochs'] = n_epochs

  algo = MFREC(**alg_kwargs)
  algo.fit(trainset, testset)
  predictions = algo.test(testset)
  mae = accuracy.mae(predictions, **{'verbose':False})
  mse = pow(accuracy.rmse(predictions, **{'verbose':False}), 2.0)
  print('%.4f %.4f %s' % (mae, mse, path.basename(kwargs_file)))
  stdout.flush()

n_epochs = coat_n_epochs
gsearch_file = tune_coat_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
lr_all_opt, reg_all_opt = set(), set()
for kwargs_str in kwargs_set:
  alg_kwargs = config.dictify(kwargs_str)
  lr_all_opt.add(alg_kwargs['lr_all'])
  reg_all_opt.add(alg_kwargs['reg_all'])

for lr_all in lr_all_opt:
  alg_kwargs = {'lr_all':lr_all,}
  reg_coat(alg_kwargs, err_kwargs)

for reg_all in reg_all_opt:
  alg_kwargs = {'reg_all':reg_all,}
  reg_coat(alg_kwargs, err_kwargs)

for lr_all, reg_all in itertools.product(lr_all_opt, reg_all_opt):
  alg_kwargs = {'lr_all':lr_all, 'reg_all':reg_all,}
  reg_coat(alg_kwargs, err_kwargs)





