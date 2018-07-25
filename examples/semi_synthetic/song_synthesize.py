from config import min_rate, max_rate

from os import path
from sys import stdout

import numpy as np

dnld_dir = path.expanduser('~/Downloads')
song_dir = path.join(dnld_dir, 'Webscope_R3')
if not path.isdir(song_dir):
  raise Exception('the song dataset does not exist')
train_file = path.join(song_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
test_file = path.join(song_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
u_set, i_set = set(), set()
r_dist = np.zeros(max_rate - min_rate + 1)
with open(test_file) as fin:
  for line in fin.readlines():
    u, i, r = [int(f) for f in line.split()]
    u_set.add(u)
    i_set.add(i)
    r_dist[r - 1] += 1
n_users, n_items = len(u_set), len(i_set)
print('#user=%d #item=%d' % (n_users, n_items))
r_dist /= r_dist.sum()
stdout.write('test:')
[stdout.write(' %.4f' % p) for p in r_dist]
stdout.write('\n')
exit()

music_dist = utils.marginalize(music_file)
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
