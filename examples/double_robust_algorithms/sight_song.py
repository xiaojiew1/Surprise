from config import figure_dir, tune_song_file

from os import path
from sys import stdout

import config

import itertools
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

lr_all_opt = sorted(list(lr_all_opt))
reg_all_opt = sorted(list(reg_all_opt))
print(lr_all_opt)
print(reg_all_opt)
lr_part_opt = [lr_all_opt[i] for i in [0, 1,]]
reg_part_opt = [reg_all_opt[i] for i in [0, 1,]]
fig, ax = plt.subplots(1, 1)
for lr_all, reg_all in itertools.product(lr_part_opt, reg_part_opt):
  alg_kwargs = {'lr_all':lr_all, 'reg_all':reg_all,}
  kwargs_file = config.get_song_file(alg_kwargs)
  errors = config.load_error(kwargs_file)
  kwargs_str = config.stringify(alg_kwargs)
  epochs = np.arange(len(errors))
  ax.plot(epochs, errors, **{'label':kwargs_str,})
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=2, mode='expand')
eps_file = path.join(figure_dir, 'song_lr_reg.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight')
