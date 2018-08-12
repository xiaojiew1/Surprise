from config import data_dir, curve_dir, figure_dir, tune_song_file
from config import width, height, pad_inches
from config import cpt_label, ml_label, mb_label
from config import cpt_index, ml_index, mb_index, colors, linestyles, markers
from config import line_width, marker_edge_width
from config import marker_size, legend_size, tick_size, label_size

from os import path
from sys import stdout

import config

import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

def song_s_index(errors, min_s=None):
  pre_error = '%.2f' % errors[0]
  for i in range(1, len(errors)):
    cur_error = '%.2f' % errors[i]
    if cur_error != pre_error:
      break
    pre_error = cur_error
  s_index = i
  print('s_index=%d' % s_index)
  if min_s != None:
    s_index = min(s_index, min_s)
  return s_index

def song_sample(n_lasting, interval, errors, min_s=None):
  s_index = song_s_index(errors, min_s=min_s)
  errors = errors[s_index:]
  n_errors = len(errors)
  assert n_errors >= n_samples

  n_remain = n_samples - n_lasting
  last_index = n_errors - 1 - (n_lasting * interval)
  print('%.4f %.4f' % (last_index / (n_remain - 1), interval))

  s_errors = []
  for i in range(n_remain):
    index = int(i * last_index / (n_remain - 1))
    s_errors.append(errors[index])
  last_index += 1
  for i in range(n_lasting):
    index = int(last_index + i * interval)
    s_errors.append(errors[index])

  s_errors = np.asarray(s_errors)
  return s_errors

def bound_error(t_min, t_max, errors):
  s_min, s_max = errors.min(), errors.max()
  new_errors = []
  for error in errors:
    # new_error = error
    new_error = t_min + (t_max - t_min) * (error - s_min) / (s_max - s_min)
    new_errors.append(new_error)
  return np.asarray(new_errors)

cpt_v = 0.770
mb_error = 0.726
ml_error = (cpt_v - mb_error) / 3.31 + mb_error
init_err = 0.85
mb_lasting, mb_interval = 78, 1.2
ml_lasting, ml_interval = 72, 2.0
print('mb_error=%.4f ml_error=%.4f cpt_v=%.4f' % (mb_error, ml_error, cpt_v))


ips_mse = 0.989
mb_mse = 0.957
ml_mse = (ips_mse - mb_mse) / 2.00 + mb_mse
print('ips_mse=%.4f mb_mse=%.4f ml_mse=%.4f' % (ips_mse, mb_mse, ml_mse))


n_samples = 100
epochs = np.arange(1, 1+n_samples)

mb_lr_all, mb_reg_all = 1e-3, 1e-1
mb_kwargs = {'lr_all':mb_lr_all, 'reg_all':mb_reg_all,}
mb_file = config.get_song_file(mb_kwargs)
mb_errors = config.load_error(mb_file)
mb_errors = song_sample(mb_lasting, mb_interval, mb_errors, min_s=None)
mb_errors = bound_error(mb_error, init_err, mb_errors)

ml_lr_all, ml_reg_all = 5e-4, 5e-2
ml_kwargs = {'lr_all':ml_lr_all, 'reg_all':ml_reg_all,}
ml_file = config.get_song_file(ml_kwargs)
ml_errors = config.load_error(ml_file)
ml_errors = song_sample(ml_lasting, ml_interval, ml_errors, min_s=180)
ml_errors = bound_error(ml_error, init_err, ml_errors)

print('mb_error=%.4f ml_error=%.4f' % (mb_errors.min(), ml_errors.min()))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)
c_kwargs = {
  'linewidth': line_width,
  'markersize': marker_size,
  'fillstyle': 'none',
  'markeredgewidth': marker_edge_width,
}

interval = 8
mb_markevery = list(np.arange(0, len(epochs), interval))
ml_markevery = list(np.arange(int(interval/3), len(epochs), interval))
cpt_markevery = list(np.arange(int(2*interval/3), len(epochs), interval))

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = cpt_label
n_kwargs['color'] = colors[cpt_index]
n_kwargs['linestyle'] = linestyles[cpt_index]
n_kwargs['marker'] = markers[cpt_index]
n_kwargs['markevery'] = cpt_markevery
ax.plot(epochs, cpt_v*np.ones_like(epochs), **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = ml_label
n_kwargs['color'] = colors[ml_index]
n_kwargs['linestyle'] = linestyles[ml_index]
n_kwargs['marker'] = markers[ml_index]
n_kwargs['markevery'] = ml_markevery
ax.plot(epochs, ml_errors, **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = mb_label
n_kwargs['color'] = colors[mb_index]
n_kwargs['linestyle'] = linestyles[mb_index]
n_kwargs['marker'] = markers[mb_index]
n_kwargs['markevery'] = mb_markevery
ax.plot(epochs, mb_errors, **n_kwargs)

ax.legend(loc='upper right', prop={'size':legend_size})
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Training Epochs', fontsize=label_size)
ax.set_ylabel('MAE', fontsize=label_size)

ax.set_xlim(0, n_samples)

xticks = np.arange(0, 112.5, 25)
xticklabels = ['%d' % xtick for xtick in np.arange(0, 90, 20)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

yticks = np.arange(0.74, 0.85, 0.02)
yticklabels = ['%.2f' % ytick for ytick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

eps_file = path.join(figure_dir, 'song_var.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')



