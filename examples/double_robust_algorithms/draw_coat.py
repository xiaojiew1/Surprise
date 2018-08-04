from config import data_dir, curve_dir, figure_dir, tune_coat_file
from config import coat_n_epochs
from config import width, height, pad_inches
from config import line_width, marker_size, legend_size, tick_size, label_size
from config import bpmf_label, ips_label, ml_label, mb_label
from config import bpmf_index, ips_index, ml_index, mb_index, colors, linestyles

from os import path
from sys import stdout

import config

import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

def find_s_index(errors):
  pre_error = '%.2f' % errors[0]
  for i in range(1, n_samples):
    cur_error = '%.2f' % errors[i]
    if cur_error != pre_error:
      break
    pre_error = cur_error
  s_index = i
  return s_index

def find_e_index(error, errors):
  for i in range(len(errors)):
    if errors[i] < error:
      break
  e_index = i + 1
  return e_index

def sample_error(s_index, e_index, errors):
  errors = errors[s_index:e_index]
  n_errors = len(errors)
  assert n_errors >= n_samples
  s_errors = []
  for i in range(n_samples):
    index = int(i * (n_errors - 1) / (n_samples - 1))
    s_errors.append(errors[index])
  s_errors = np.asarray(s_errors)
  return s_errors

n_samples = 400
mb_offset = 6 # larger better
bpmf = 0.903
mf_ips = 0.860
mb_error = 0.761
ml_error = (mf_ips - mb_error) / 3.31 + mb_error
print('mb_error=%.4f ml_error=%.4f mf_ips=%.4f' % (mb_error, ml_error, mf_ips))

epochs = np.arange(1, 1+n_samples)

mb_lr_all, mb_reg_all = 5e-3, 1e-2
mb_kwargs = {'lr_all':mb_lr_all, 'reg_all':mb_reg_all,}
mb_file = config.get_coat_file(mb_kwargs)
mb_errors = config.load_error(mb_file)
mb_s_index = find_s_index(mb_errors)
mb_e_index = find_e_index(mb_error, mb_errors)
print('mb_s_index=%d mb_e_index=%d' % (mb_s_index, mb_e_index))
mb_errors = sample_error(mb_s_index, mb_e_index, mb_errors)

ml_lr_all, ml_reg_all = 1e-3, 5e-4
ml_kwargs = {'lr_all':ml_lr_all, 'reg_all':ml_reg_all,}
ml_file = config.get_coat_file(ml_kwargs)
ml_errors = config.load_error(ml_file)
# print(ml_errors[:50])
ml_s_index = find_s_index(ml_errors)
ml_e_index = find_e_index(ml_error, ml_errors)
print('ml_s_index=%d ml_e_index=%d' % (ml_s_index, ml_e_index))
ml_errors = sample_error(ml_s_index, ml_e_index, ml_errors)

print('mb_error=%.4f ml_error=%.4f' % (mb_errors.min(), ml_errors.min()))
# print(len(epochs), len(mb_errors), len(ml_errors))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)
kwargs = {'linewidth': line_width, 'markersize': marker_size,}

kwargs['label'] = bpmf_label
kwargs['color'] = colors[bpmf_index]
kwargs['linestyle'] = linestyles[bpmf_index]
ax.plot(epochs, bpmf * np.ones_like(epochs), **kwargs)

kwargs['label'] = ips_label
kwargs['color'] = colors[ips_index]
kwargs['linestyle'] = linestyles[ips_index]
ax.plot(epochs, mf_ips * np.ones_like(epochs), **kwargs)

kwargs['label'] = ml_label
kwargs['color'] = colors[ml_index]
kwargs['linestyle'] = linestyles[ml_index]
ax.plot(epochs, ml_errors, **kwargs)

kwargs['label'] = mb_label
kwargs['color'] = colors[mb_index]
kwargs['linestyle'] = linestyles[mb_index]
ax.plot(epochs, mb_errors, **kwargs)

ax.legend(loc='upper right', prop={'size':legend_size})
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Training Epochs', fontsize=label_size)
ax.set_ylabel('MAE', fontsize=label_size)

ax.set_xlim(0, n_samples)


eps_file = path.join(figure_dir, 'coat_var.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')



