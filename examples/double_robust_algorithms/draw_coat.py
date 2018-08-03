from config import data_dir, curve_dir, figure_dir, rec_coat_file
from config import width, height, pad_inches
from config import line_width, marker_size, legend_size, tick_size, label_size

from util_coat import trainset, testset
from surprise import accuracy
from surprise import MFREC


from os import path
from sys import stdout

import config

import matplotlib.pyplot as plt
import numpy as np
import operator
import time

e_index = 1

def load_error(kwargs_file):
  maes = []
  with open(kwargs_file) as fin:
    for line in fin.readlines():
      fields = line.split()
      mae = float(fields[e_index])
      maes.append(mae)
  return np.asarray(maes)

bt_name = 'COAT_BIASED_false_LR-ALL_5e-2_N-EPOCHS_400_N-FACTORS_256_REG-ALL_5e-3_VAR-ALL_1e-10.p'
bl_name = 'COAT_BIASED_false_LR-ALL_5e-2_N-EPOCHS_400_N-FACTORS_256_REG-ALL_5e-3_VAR-ALL_1e-2.p'
bt_name = 'COAT_BIASED_false_LR-ALL_5e-3_N-EPOCHS_400_N-FACTORS_256_REG-ALL_5e-3.p'
bl_name = 'COAT_BIASED_false_LR-ALL_1e-3_N-EPOCHS_400_N-FACTORS_256_REG-ALL_5e-3.p'
bt_name = 'COAT_BIASED_false_LR-ALL_5e-2_N-EPOCHS_400_N-FACTORS_256_REG-ALL_1e-2.p'
bl_name = 'COAT_BIASED_false_LR-ALL_5e-2_N-EPOCHS_400_N-FACTORS_256_REG-ALL_1e-3.p'

bt_name = 'COAT_BIASED_false_LR-ALL_5e-2_N-EPOCHS_400_N-FACTORS_256_REG-ALL_1e-1.p'
bl_name = 'COAT_BIASED_false_LR-ALL_4e-3_N-EPOCHS_400_N-FACTORS_256_REG-ALL_2e-1.p'
bt_file = path.join(curve_dir, bt_name)
bl_file = path.join(curve_dir, bl_name)

bt_maes = load_error(bt_file)
bl_maes = load_error(bl_file)
assert len(bt_maes) == len(bl_maes)
n_times = int((len(bt_maes) + len(bl_maes)) / 2)
epochs = np.arange(1, 1+n_times)

indexes = np.arange(0, n_times, 1)
indexes = np.arange(0, int(n_times / 2), 1)
bt_maes = bt_maes[indexes]
bl_maes = bl_maes[indexes]
epochs = epochs[indexes]

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)
kwargs = {'linewidth': line_width, 'markersize': marker_size,}

## ips estimator
kwargs['label'] = 'bt'
# kwargs['marker'] = 'x'
kwargs['linestyle'] = '-'
ax.plot(epochs, bt_maes, **kwargs)

## snips estimator
kwargs['label'] = 'bl'
# kwargs['marker'] = 'x'
kwargs['linestyle'] = ':'
ax.plot(epochs, bl_maes, **kwargs)

ax.legend(loc='upper right', prop={'size':legend_size})

# ax.set_xticks(np.arange(0.20, 1.05, 0.20))
# ax.tick_params(axis='both', which='major', labelsize=tick_size)
# ax.set_xlabel('Selection Bias $\\alpha$', fontsize=label_size)
# ax.set_xlim(0.1, 1.0)

# ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

# if risk_name == 'mae':
#   yticks = np.arange(0.000, 0.070, 0.020)
#   ax.set_yticks(yticks)
#   ax.set_yticklabels([('%.2f' % ytick)[1:] for ytick in yticks])
# else:
#   yticks = np.arange(0.00, 0.35, 0.10)
#   ax.set_yticks(yticks)
#   ax.set_yticklabels([('%.1f' % ytick)[1:] for ytick in yticks])

eps_file = path.join(figure_dir, 'coat_curve.eps')
config.make_file_dir(eps_file)
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)






