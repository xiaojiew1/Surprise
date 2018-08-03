from config import tune_song_file
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

gsearch_file = tune_song_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
err_kwargs = sorted(err_kwargs, key=operator.itemgetter(1,0))
if len(err_kwargs) == 0:
  raise Exception('first tune song')

kwargs_str = err_kwargs[0][2]
algo_kwargs = config.dictify(kwargs_str)

s_time = time.time()

algo = MFREC(**algo_kwargs)
algo.fit(trainset)
predictions = algo.test(testset)
eval_kwargs = {'verbose':False}
mae = accuracy.mae(predictions, **eval_kwargs)
mse = pow(accuracy.rmse(predictions, **eval_kwargs), 2.0)
print('%.4f %.4f %s' % (mae, mse, kwargs_str))
stdout.flush()

e_time = time.time()
print('%.2fs' % (e_time - s_time))









