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

def rec_coat(kwargs):
  kwargs = config.dictify(kwargs)
  kwargs_str = config.stringify(kwargs)
  kwargs_file = path.join(curve_dir, 'COAT_%s.p' % kwargs_str)
  if not path.isfile(kwargs_file):
    config.make_file_dir(kwargs_file)
    kwargs['n_epochs'] = n_epochs
    kwargs['eval_space'] = eval_space
    kwargs['kwargs_file'] = kwargs_file

    algo = MFREC(**kwargs)
    algo.fit(trainset, testset)
    predictions = algo.test(testset)
    mae = accuracy.mae(predictions, **{'verbose':False})
    mse = pow(accuracy.rmse(predictions, **{'verbose':False}), 2.0)
    print('%.4f %.4f %s' % (mae, mse, kwargs_str))
    stdout.flush()
  return kwargs_file

def load_mae(kwargs_file):
  maes = []
  with open(kwargs_file) as fin:
    for line in fin.readlines():
      fields = line.split()
      mae = float(fields[1])
      maes.append(mae)
  return np.asarray(maes)

mae_index = 0
arg_index = 2
n_epochs = 400
eval_space = trainset.n_ratings

gsearch_file = rec_coat_file
err_kwargs, kwargs_set = config.read_gsearch(gsearch_file)
err_kwargs = sorted(err_kwargs, key=operator.itemgetter(mae_index))
if len(err_kwargs) == 0:
  raise Exception('first tune coat')
bt_kwargs = err_kwargs[0][arg_index]

bl_kwargs = {'reg_all': 0.005,}
bl_kwargs = config.stringify(bl_kwargs)
bls_kwargs = [err_kwarg for err_kwarg in err_kwargs if bl_kwargs in err_kwarg[arg_index]]
bls_kwargs = sorted(bls_kwargs, key=operator.itemgetter(arg_index))
bl_kwargs = bls_kwargs[0][arg_index]

s_time = time.time()
bt_file = rec_coat(bt_kwargs)
bl_file = rec_coat(bl_kwargs)
e_time = time.time()
print('%.2fs' % (e_time - s_time))

bt_maes = load_mae(bt_file)
bl_maes = load_mae(bl_file)
n_times = n_epochs * trainset.n_ratings // eval_space
epochs = np.arange(1, 1+n_times)

indexes = np.arange(0, n_times, 1)
bt_maes = bt_maes[indexes]
bl_maes = bl_maes[indexes]
epochs = epochs[indexes]

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)
kwargs = {'linewidth': line_width, 'markersize': marker_size,}

## ips estimator
kwargs['label'] = 'bt'
ax.plot(epochs, bt_maes, **kwargs)

## snips estimator
kwargs['label'] = 'bl'
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






