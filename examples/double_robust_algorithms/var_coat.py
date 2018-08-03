from config import data_dir, curve_dir, figure_dir, tune_coat_file
from config import width, height, pad_inches
from config import line_width, marker_size, legend_size, tick_size, label_size

from util_coat import trainset, testset
from surprise import accuracy
from surprise import VARREC


from os import path
from sys import stdout

import config

import matplotlib.pyplot as plt
import numpy as np
import operator
import time

def var_coat(alg_kwargs, var_all):
  alg_kwargs = config.dictify(alg_kwargs)
  alg_kwargs['n_epochs'] = n_epochs
  alg_kwargs['var_all'] = var_all
  kwargs_str = config.stringify(alg_kwargs)
  kwargs_file = path.join(curve_dir, 'COAT_%s.p' % kwargs_str)
  alg_kwargs['eval_space'] = eval_space
  alg_kwargs['kwargs_file'] = kwargs_file
  algo = VARREC(**alg_kwargs)
  algo.fit(trainset, testset)
  predictions = algo.test(testset)
  mae = accuracy.mae(predictions, **{'verbose':False})
  mse = pow(accuracy.rmse(predictions, **{'verbose':False}), 2.0)
  print('%.4f %.4f %s' % (mae, mse, kwargs_str))
  stdout.flush()

mae_index = 0
arg_index = 2
n_epochs = 400
eval_space = int(trainset.n_ratings * n_epochs / n_epochs)

gsearch_file = tune_coat_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
err_kwargs = sorted(err_kwargs, key=operator.itemgetter(mae_index))
if len(err_kwargs) == 0:
  raise Exception('first tune coat')
bt_kwargs = err_kwargs[0][arg_index]

s_time = time.time()
var_all_opt = [pow(10.0, i) for i in range(-10, 3)]
for var_all in var_all_opt:
  bt_file = var_coat(bt_kwargs, var_all)
e_time = time.time()
print('%.2fs' % (e_time - s_time))





