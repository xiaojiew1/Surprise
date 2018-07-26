from utils import min_rate, max_rate, song_file

from os import path
from sys import stdout

import utils

import numpy as np

dnld_dir = path.expanduser('~/Downloads')
song_dir = path.join(dnld_dir, 'Webscope_R3')
if not path.isdir(song_dir):
  raise Exception('the song dataset does not exist')
train_file = path.join(song_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
test_file = path.join(song_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
u_set, i_set = set(), set()
rate_dist = np.zeros(max_rate-min_rate+1)
with open(test_file) as fin:
  for line in fin.readlines():
    user, item, rate = [int(f) for f in line.split()]
    u_set.add(user)
    i_set.add(item)
    rate_dist[rate-1] += 1
n_users, n_items = len(u_set), len(i_set)
rate_dist /= rate_dist.sum()
stdout.write('dist:')
[stdout.write(' %.4f' % p) for p in rate_dist]
stdout.write('\n')

r_cum = np.zeros(max_rate-min_rate+2)
for rate in range(min_rate, min_rate+max_rate):
  r_cum[rate] = rate_dist[rate-1]
for rate in range(min_rate, min_rate+max_rate):
  r_cum[rate] = r_cum[rate] + r_cum[rate-1]
stdout.write('cum:')
[stdout.write(' %.4f' % p) for p in r_cum]
stdout.write('\n')

n_rates = 0
with open(train_file) as fin:
  for line in fin.readlines():
    user, item, rate = [int(f) for f in line.split()]
    if user not in u_set:
      continue
    n_rates += 1
    
#### ml100k
# n_users, n_items = 943, 1683
# n_rates = int(n_users * n_items * 0.05)

utils.make_file_dir(song_file)
fout = open(song_file, 'w')
fout.write('%d %d %d\n' % (n_users, n_items, n_rates))
indexes = np.zeros(max_rate-min_rate+2)
n_entries = n_users * n_items
for rate in range(min_rate, min_rate+max_rate):
  indexes[rate] = int(r_cum[rate] * n_entries)
for rate in range(max_rate-min_rate+2):
  fout.write('%d' % indexes[rate])
  if rate < max_rate-min_rate+1:
    fout.write(' ')
  else:
    fout.write('\n')
fout.close()

sparsity = n_rates / (n_users * n_items)
print('#user=%d #item=%d sparsity=%.4f' % (n_users, n_items, sparsity))



