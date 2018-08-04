from config import tune_song_file
from config import song_n_points, song_lr_count

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

def reg_song(alg_kwargs, err_kwargs):
  kwargs_file = config.get_song_file(alg_kwargs)
  if path.isfile(kwargs_file):
    return
  config.make_file_dir(kwargs_file)

  alg_kwargs = config.min_kwargs(alg_kwargs, err_kwargs)
  n_epochs = alg_kwargs['n_epochs']
  alg_kwargs['eval_space'] = int(trainset.n_ratings / (n_points / n_epochs))
  alg_kwargs['kwargs_file'] = kwargs_file
  # print(alg_kwargs)

  algo = MFREC(**alg_kwargs)
  algo.fit(trainset, testset)
  predictions = algo.test(testset)
  mae = accuracy.mae(predictions, **{'verbose':False})
  mse = pow(accuracy.rmse(predictions, **{'verbose':False}), 2.0)
  print('%.4f %.4f %s' % (mae, mse, path.basename(kwargs_file)))
  stdout.flush()

n_points = song_n_points
gsearch_file = tune_song_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
lr_all_opt, reg_all_opt = set(), set()
for kwargs_str in kwargs_set:
  alg_kwargs = config.dictify(kwargs_str)
  lr_all_opt.add(alg_kwargs['lr_all'])
  reg_all_opt.add(alg_kwargs['reg_all'])

lr_all_opt = sorted(lr_all_opt)[:song_lr_count]
for lr_all, reg_all in itertools.product(lr_all_opt, reg_all_opt):
  alg_kwargs = {'lr_all':lr_all, 'reg_all':reg_all,}
  reg_song(alg_kwargs, err_kwargs)










