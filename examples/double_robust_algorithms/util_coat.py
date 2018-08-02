from config import dnld_dir
from surprise import Dataset
from surprise import Reader

from os import path

import io
import numpy as np
import requests
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

data_dir = path.join(dnld_dir, 'coat')
data_url = 'http://www.cs.cornell.edu/~schnabts/mnar/coat.zip'
train_rawfile = path.join(data_dir, 'train.ascii')
train_file = path.join(data_dir, 'train.txt')
test_rawfile = path.join(data_dir, 'test.ascii')
test_file = path.join(data_dir, 'test.txt')
propensity_rawfile = path.join(data_dir, 'propensities.ascii')
rating_scale = (1, 5)

if not path.isdir(data_dir):
  print('download coat dataset...')
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

