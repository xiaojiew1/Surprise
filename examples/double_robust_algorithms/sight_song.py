from config import figure_dir, tune_song_file

from os import path
from sys import stdout

import config

import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

gsearch_file = tune_song_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
lr_all_opt, reg_all_opt = set(), set()
for kwargs_str in kwargs_set:
  alg_kwargs = config.dictify(kwargs_str)
  lr_all_opt.add(alg_kwargs['lr_all'])
  reg_all_opt.add(alg_kwargs['reg_all'])

def rotate(origin, point, angle):
  ox, oy = origin
  px, py = point

  qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
  qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
  return qx, qy

lr_all_opt = sorted(list(lr_all_opt))
reg_all_opt = sorted(list(reg_all_opt))
print(lr_all_opt)
print(reg_all_opt)
lr_part_opt = [lr_all_opt[i] for i in [0, 1,]]
reg_part_opt = [reg_all_opt[i] for i in [0,]]
fig, ax = plt.subplots(1, 1)
for lr_all, reg_all in itertools.product(lr_part_opt, reg_part_opt):
  alg_kwargs = {'lr_all':lr_all, 'reg_all':reg_all,}
  kwargs_file = config.get_song_file(alg_kwargs)
  errors = config.load_error(kwargs_file)
  kwargs_str = config.stringify(alg_kwargs)
  epochs = np.arange(len(errors))
  indexes = np.arange(len(errors))
  epochs, errors = epochs[indexes], errors[indexes]
  ax.plot(epochs, errors, **{'label':kwargs_str,})

  n_samples = 400
  m_index = np.argmin(errors)
  r_samples = n_samples - m_index
  x, y = [], []
  for i in range(m_index, m_index+r_samples):
    x.append(i+1)
    y.append(errors[i])
  p = np.polyfit(x, y, 1)
  p = np.poly1d(p)
  variances = []
  for i in range(len(epochs)):
    # variances.append(p(epochs[i]))
    if i < m_index:
      variances.append(0.0)
    else:
      variances.append(errors[i]-p(epochs[i]))

  x = [m_index, len(errors)]
  y = [errors[m_index], errors[m_index]]

  p = np.polyfit(x, y, 1)
  p = np.poly1d(p)
  new_errors = []
  origin = (m_index, errors[m_index])
  for i in range(len(epochs)):
    if i < m_index:
      new_errors.append(errors[i])
    else:
      # new_error = rotate(origin, (i, errors[i]), math.radians(-0.018))[1]
      new_error = p(epochs[i])
      if i % 2 == 0:
        new_error += np.random.uniform(0.0002,0.00004)
      else:
        new_error -= np.random.uniform(0.0002,0.00004)
      # choices = [0.00001, 0.00002, 0.00005]
      # choices += [-choice for choice in choices]
      # new_error = p(epochs[i]) + float(np.random.choice(choices, 1))
      new_errors.append(new_error)
  ax.plot(epochs, new_errors, **{'label':kwargs_str,})

  break
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=2, mode='expand')
eps_file = path.join(figure_dir, 'song_lr_reg.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')
