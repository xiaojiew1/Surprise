from config import dnld_dir
from surprise import Dataset
from surprise import Reader

from os import path
import config

import io
import numpy as np
import requests
import shutil
import zipfile

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

reader = Reader(line_format='user item rating', sep='\t')
data = Dataset(reader=reader, rating_scale=rating_scale)
raw_trainset = data.read_ratings(train_file)
raw_testset = data.read_ratings(test_file)
trainset = data.construct_trainset(raw_trainset)
testset = data.construct_testset(raw_testset)



