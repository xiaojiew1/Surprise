from config import data_dir, curve_dir, figure_dir, tune_coat_file
from config import width, height, pad_inches
from config import line_width, marker_size, legend_size, tick_size, label_size

from os import path
from sys import stdout

import config

import matplotlib.pyplot as plt
import numpy as np
import operator
import time

def load_error(alg_kwargs):
  errors = []
  kwargs_file = config.get_p_file(alg_kwargs)
  with open(kwargs_file) as fin:
    for line in fin.readlines():
      error = line.split()[1]
      errors.append(float(error))
  errors = np.asarray(errors)
  return errors

gsearch_file = tune_coat_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
lr_all_opt, reg_all_opt = set(), set()
for kwargs_str in kwargs_set:
  alg_kwargs = config.dictify(kwargs_str)
  lr_all_opt.add(alg_kwargs['lr_all'])
  reg_all_opt.add(alg_kwargs['reg_all'])

n_epochs = 1024
epochs = np.arange(1, 1+n_epochs)
fig, ax = plt.subplots(1, 1)
for lr_all in lr_all_opt:
  if lr_all < 5e-4 or lr_all > 5e-3:
    continue
  alg_kwargs = {'lr_all':lr_all,}
  errors = load_error(alg_kwargs)
  kwargs_str = config.stringify(alg_kwargs)
  ax.plot(epochs, errors, **{'label':kwargs_str,})
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=3, mode='expand')
eps_file = path.join(figure_dir, 'coat_lr_all.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')

fig, ax = plt.subplots(1, 1)
for reg_all in reg_all_opt:
  if reg_all < 5e-3 or reg_all > 1e-2:
    continue
  alg_kwargs = {'reg_all':reg_all,}
  errors = load_error(alg_kwargs)
  kwargs_str = config.stringify(alg_kwargs)
  ax.plot(epochs, errors, **{'label':kwargs_str,})
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=3, mode='expand')
eps_file = path.join(figure_dir, 'coat_reg_all.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')




