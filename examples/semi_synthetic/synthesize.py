from config import data_file
from config import min_rate, max_rate

from os import path
from surprise import accuracy
from surprise import Dataset
from surprise import SVD
from surprise.builtin_datasets import BUILTIN_DATASETS
from sys import stdout

import utils

import math
import numpy as np

def evaluate(ui_rates, data_file):
  ml_rates = []
  with open(data_file) as fin:
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

data_name = 'ml-100k'
data = Dataset.load_builtin(data_name)
params = {'n_factors': 50, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}

train_set = data.build_full_trainset()
algo = SVD(**params)
algo.fit(train_set)
test_set = train_set.build_testset()
predictions = algo.test(test_set)
accuracy.rmse(predictions, verbose=True)

n_users = train_set.n_users
n_items = train_set.n_items
print('#user=%d #item=%d' % (n_users, n_items))

music_dir = path.expanduser('~/Projects/drrec/data/music/')
music_file = 'ydata-ymusic-rating-study-v1_0-test.txt'
music_file = path.join(music_dir, music_file)

music_dist = utils.marginalize(music_file)
[stdout.write('%.4f, ' % p) for p in music_dist]
stdout.write('\n')
[stdout.write('%.2f, ' % p) for p in music_dist]
stdout.write('\n')
exit()
music_cum = np.zeros(max_rate+1)
for rate in range(min_rate, min_rate+max_rate):
  music_cum[rate] = music_dist[rate-1]
for rate in range(min_rate, min_rate+max_rate):
  music_cum[rate] = music_cum[rate] + music_cum[rate-1]

ui_rates = []
for inner_uid in train_set.all_users():
  raw_uid = train_set.to_raw_uid(inner_uid)
  for inner_iid in train_set.all_items():
    raw_iid = train_set.to_raw_iid(inner_iid)
    pred = algo.predict(raw_uid, raw_iid)
    rate = pred.est
    ui_rates.append((raw_uid, raw_iid, rate))
# rates = [rate for _, _, rate in ui_rates]
# print('min=%.4f max=%.4f' % (min(rates), max(rates)))
dataset = BUILTIN_DATASETS[data_name]
evaluate(ui_rates, dataset.path)

ui_rates = sorted(ui_rates, key=lambda uir: uir[-1])

fout = open(data_file, 'w')
fout.write('%d %d\n' % (n_users, n_items))
n_rates = n_users * n_items
for rate in range(min_rate, min_rate+max_rate):
  sidx = int(n_rates * music_cum[rate-1])
  eidx = int(n_rates * music_cum[rate])
  print('rate=%d [%d, %d)' % (rate, sidx, eidx))
  for i in range(sidx, eidx):
    ui_rates[i] = (ui_rates[i][0], ui_rates[i][1], rate)
  fout.write('%d ' % (sidx))
# [print('%4s %3s %d'% ui_rates[i*n_items*int(n_users/10)]) for i in range(10)]
fout.write('%d\n' % (eidx))
fout.close()
