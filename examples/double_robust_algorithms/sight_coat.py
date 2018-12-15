from config import figure_dir, tune_coat_file
from config import coat_n_epochs

from os import path
from sys import stdout

import config

import itertools
import matplotlib.pyplot as plt
import numpy as np
import operator
import time

gsearch_file = tune_coat_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
lr_all_opt, reg_all_opt = set(), set()
for kwargs_str in kwargs_set:
  alg_kwargs = config.dictify(kwargs_str)
  lr_all_opt.add(alg_kwargs['lr_all'])
  reg_all_opt.add(alg_kwargs['reg_all'])

n_epochs = coat_n_epochs
epochs = np.arange(1, 1+n_epochs)

lr_all_opt = sorted(list(lr_all_opt))
reg_all_opt = sorted(list(reg_all_opt))
lr_part_opt = [lr_all_opt[i] for i in [2,]]
lr_part_opt = [lr_all_opt[i] for i in [0, 1, 2,]]
reg_part_opt = [reg_all_opt[i] for i in [0, 2, 4, 6,]]
reg_part_opt = [reg_all_opt[i] for i in [0, 3, 6,]]
reg_part_opt = [reg_all_opt[i] for i in [2, 3, 4,]]
reg_part_opt = [reg_all_opt[i] for i in [4,]]
print(lr_all_opt)
print(lr_part_opt)
print(reg_all_opt)
print(reg_part_opt)
fig, ax = plt.subplots(1, 1)
for lr_all, reg_all in itertools.product(lr_part_opt, reg_part_opt):
  alg_kwargs = {'lr_all':lr_all, 'reg_all':reg_all,}
  kwargs_file = config.get_coat_file(alg_kwargs)
  errors = config.load_error(kwargs_file)
  kwargs_str = config.stringify(alg_kwargs)
  ax.plot(epochs, errors, **{'label':kwargs_str,})
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=2, mode='expand')
eps_file = path.join(figure_dir, 'coat_lr_reg.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')
exit()

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




