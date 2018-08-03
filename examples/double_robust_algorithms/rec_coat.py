from config import data_dir, curve_dir, figure_dir, rec_coat_file
from config import width, height, pad_inches
from config import line_width, marker_size, legend_size, tick_size, label_size

from util_coat import trainset, testset
from surprise import accuracy
from surprise import MFREC


from os import path
from sys import stdout

import config

import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

def rec_coat(alg_kwargs):
  alg_kwargs['n_epochs'] = n_epochs
  kwargs_str = config.stringify(alg_kwargs)
  kwargs_file = path.join(curve_dir, 'COAT_%s.p' % kwargs_str)
  config.make_file_dir(kwargs_file)
  alg_kwargs['eval_space'] = eval_space
  alg_kwargs['kwargs_file'] = kwargs_file

  algo = MFREC(**alg_kwargs)
  algo.fit(trainset, testset)
  predictions = algo.test(testset)
  mae = accuracy.mae(predictions, **{'verbose':False})
  mse = pow(accuracy.rmse(predictions, **{'verbose':False}), 2.0)
  print('%.4f %.4f %s' % (mae, mse, path.basename(kwargs_file)))
  stdout.flush()

def load_mae(kwargs_file):
  maes = []
  with open(kwargs_file) as fin:
    for line in fin.readlines():
      fields = line.split()
      mae = float(fields[1])
      maes.append(mae)
  return np.asarray(maes)

mae_index = 0
arg_index = 2
n_epochs = 400
eval_space = trainset.n_ratings

gsearch_file = rec_coat_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
err_kwargs = sorted(err_kwargs, key=operator.itemgetter(mae_index))
if len(err_kwargs) == 0:
  raise Exception('first tune coat')
bt_kwargs = err_kwargs[0][arg_index]
# print(bt_kwargs)
# exit()

lr_all_opt = [0.05,]
reg_all_opt = [0.005,]
lr_all_opt = [0.01, 0.03, 0.05,]
reg_all_opt = [0.001, 0.003, 0.005,]
lr_reg_all_opt = [
  (0.05, 0.1),
  # (0.004, 0.2),
]
# for lr_all, reg_all in itertools.product(lr_all_opt, reg_all_opt):
for lr_all, reg_all in lr_reg_all_opt:
  alg_kwargs = config.dictify(bt_kwargs)
  alg_kwargs['lr_all'] = lr_all
  alg_kwargs['reg_all'] = reg_all
  rec_coat(alg_kwargs)
exit()

reg_all_opt = [pow(10.0, i) for i in range(-5, -0)]
reg_all_opt += [5*pow(10.0, i) for i in range(-5, -0)]
for reg_all in reg_all_opt:
  alg_kwargs = config.dictify(bt_kwargs)
  alg_kwargs['reg_all'] = reg_all
  rec_coat(alg_kwargs)
exit()

lr_all_opt = [pow(10.0, i) for i in range(-6, -1)]
lr_all_opt += [5*pow(10.0, i) for i in range(-6, -1)]
for lr_all in lr_all_opt:
  alg_kwargs = config.dictify(bt_kwargs)
  alg_kwargs['lr_all'] = lr_all
  rec_coat(alg_kwargs)


