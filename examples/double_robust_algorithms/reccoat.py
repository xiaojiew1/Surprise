from config import tmp_dir
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import MFIPS

from os import path

import config

import itertools
import io
import numpy as np
import requests
import time
import zipfile

def dense_to_sparse(rawfile, outfile):
  true_ratings = np.loadtxt(rawfile, dtype=int)
  u_indexes, i_indexes = np.nonzero(true_ratings)
  with open(outfile, 'w') as fout:
    for uid, iid in zip(u_indexes, i_indexes):
      rate = true_ratings[uid, iid]
      fout.write('%d %d %d\n' % (uid, iid, rate))
  n_rates = np.count_nonzero(true_ratings)
  print('%10s: %04d' % (path.basename(outfile), n_rates))

dnld_dir = path.expanduser('~/Downloads')
data_dir = path.join(dnld_dir, 'coat')
data_url = 'http://www.cs.cornell.edu/~schnabts/mnar/coat.zip'
train_rawfile = path.join(data_dir, 'train.ascii')
train_file = path.join(data_dir, 'train.txt')
test_rawfile = path.join(data_dir, 'test.ascii')
test_file = path.join(data_dir, 'test.txt')
propensity_rawfile = path.join(data_dir, 'propensities.ascii')
rating_scale = (1, 5)

if not path.isdir(data_dir):
  response = requests.get(data_url, stream=True)
  zip_file = zipfile.ZipFile(io.BytesIO(response.content))
  zip_file.extractall(path=dnld_dir)
if not path.isfile(train_file):
  dense_to_sparse(train_rawfile, train_file)
if not path.isfile(test_file):
  dense_to_sparse(test_rawfile, test_file)

reader = Reader(line_format='user item rating', sep=' ')
data = Dataset(reader=reader, rating_scale=rating_scale)
raw_trainset = data.read_ratings(train_file)
raw_testset = data.read_ratings(test_file)
trainset = data.construct_trainset(raw_trainset)
testset = data.construct_testset(raw_testset)

n_users, n_items = trainset.n_users, trainset.n_items
rpropensities = np.loadtxt(propensity_rawfile)
propensities = np.zeros_like(rpropensities)
for ruid in range(n_users):
  uid = trainset.to_inner_uid(str(ruid))
  for riid in range(n_items):
    iid = trainset.to_inner_iid(str(riid))
    # print('%d, %d -> %d, %d' % (ruid, riid, uid, iid))
    propensities[uid, iid] = rpropensities[ruid, riid]
# weights = np.minimum(10.0, 1.0 / propensities)
# weights = np.ones_like(propensities)
org_sum, tgt_sum = 0.0, 0.0
for uid, iid, r in trainset.all_ratings():
  org_sum += 1.0 / propensities[uid, iid]
  tgt_sum += 1.0
# print('org_sum=%.4f tgt_sum=%.4f' % (org_sum, tgt_sum))
weights = (tgt_sum / org_sum) / propensities
# print('min=%.4f max=%.4f' % (weights.min(), weights.max()))

#### default
n_factors_opt = [100,]
n_epochs_opt = [20,]
biased_opt = [True,]
reg_all_opt = [0.02,]
n_factors_opt = [10, 20, 50, 100, 200,]
n_epochs_opt = [10, 20, 50, 100, 200,]
biased_opt = [True, False,]
reg_all_opt = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
# n_factors_opt = [200,]
# n_epochs_opt = [200,]
# biased_opt = [False,]
# reg_all_opt = [0.1,]

#### develop
n_factors_opt = [50,]
n_epochs_opt = [200,]
biased_opt = [False,]
reg_all_opt = [0.1,]
var_all_opt = [0.0001]
# var_all_opt = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

mae_bst, mse_bst, kwargs_bst = np.inf, np.inf, {}
st_time = time.time()
for n_factors, n_epochs, biased, reg_all, var_all in itertools.product(
    n_factors_opt, n_epochs_opt, biased_opt, reg_all_opt, var_all_opt):
  algo_kwargs = {
    'n_factors': n_factors,
    'n_epochs': n_epochs,
    'biased': biased,
    'reg_all': reg_all,
    'var_all': var_all,
    # 'verbose': True,
  }
  kwargs_str = config.stringify(algo_kwargs)
  outfile = path.join(tmp_dir, '%s.log' % kwargs_str)
  fit_kwargs = {
    'testset': testset,
    'outfile': outfile,
  }

  algo = MFIPS(**algo_kwargs)
  algo.fit(trainset, weights, **fit_kwargs)
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









