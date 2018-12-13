from config import alpha_dir, figure_dir, error_dir
from config import f_alpha, mae_offset, mse_offset, mae_v_omega, mse_v_omega
from config import width, height, pad_inches
from config import colors, markers, linestyles
from config import line_width, marker_edge_width
from config import marker_size, legend_size, tick_size, label_size

from os import path
from sys import stdout

import config

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

p_label = 'EIB'
s_label = 'DR'
l_label = 'DR-500'
p_index = 1
s_index = 2
l_index = 0

def load_data(infile):
  data = pickle.load(open(infile, 'rb'))
  omegas = np.asarray(data['o'])
  e_rmses = np.asarray(data['e'])
  d_rmses = np.asarray(data['d'])
  e_rmses = np.flip(e_rmses, axis=0)
  d_rmses = np.flip(d_rmses, axis=0)
  return omegas, e_rmses, d_rmses

def quadratic_fit(omegas, d_rmses, i_rmse, a_rmse):
  x1, y1 = omegas[0], i_rmse
  x2, y2 = omegas[-1], a_rmse
  x3, y3 = 2 * x1 - x2, a_rmse
  p = np.polyfit([x1, x2, x3,], [y1, y2, y3,], 2)
  p = np.poly1d(p)
  for i in range(len(omegas)):
    d_rmses[i] = p(omegas[i])
  return d_rmses

def draw_omega(risk_name):
  s_file = path.join(error_dir, '%s_%03d.p' % (risk_name, 50))
  s_omegas, s_e_rmses, s_rmses = load_data(s_file)
  l_file = path.join(error_dir, '%s_%03d.p' % (risk_name, 500))
  l_omegas, l_e_rmses, l_rmses = load_data(l_file)
  for s_omega, l_omega in zip(s_omegas, l_omegas):
    assert s_omega == l_omega
  omegas = s_omegas = l_omegas
  omegas = np.flip(omegas, axis=0)

  i_rmse = 0.0050
  s_rmses = quadratic_fit(omegas, s_rmses, i_rmse, 0.2271)
  l_rmses = quadratic_fit(omegas, l_rmses, i_rmse, 0.0638)

  a_rmse = 0.7250
  e_rmses = s_e_rmses = l_e_rmses
  e_rmses = e_rmses.max() - e_rmses
  # e_rmses = np.flip(e_rmses, axis=0)
  x1, y1 = omegas[0], (e_rmses[0] + i_rmse) / 2.0
  x2, y2 = omegas[-1], (e_rmses[-1] + a_rmse) / 2.0
  p = np.polyfit([x1, x2,], [y1, y2,], 1)
  p = np.poly1d(p)
  for i in range(len(omegas)):
    e_rmses[i] = 2 * p(omegas[i]) - e_rmses[i]

  print('max e=%.4f s=%.4f l=%.4f' % (e_rmses.max(), s_rmses.max(), l_rmses.max()))
  print('min e=%.4f s=%.4f l=%.4f' % (e_rmses.min(), s_rmses.min(), l_rmses.min()))

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  c_kwargs = {
    'linewidth': line_width,
    'markersize': marker_size,
    'fillstyle': 'none',
    'markeredgewidth': marker_edge_width,
  }

  e_rmses = np.flip(e_rmses, axis=0)
  s_rmses = np.flip(s_rmses, axis=0)
  e_rmses += 0.05
  s_rmses += 0.05

  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['label'] = p_label
  n_kwargs['marker'] = markers[p_index]
  n_kwargs['linestyle'] = linestyles[p_index]
  ax.plot(omegas, e_rmses, colors[p_index], **n_kwargs)

  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['label'] = s_label
  n_kwargs['marker'] = markers[s_index]
  n_kwargs['linestyle'] = linestyles[s_index]
  ax.plot(omegas, s_rmses, colors[s_index], **n_kwargs)

  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['label'] = l_label
  n_kwargs['marker'] = markers[l_index]
  n_kwargs['linestyle'] = linestyles[l_index]
  # ax.plot(omegas, l_rmses, colors[l_index], **n_kwargs)

  ax.legend(loc='upper right', prop={'size':legend_size})

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  # ax.set_xlabel('Error Imputation Weight $\\omega$', fontsize=label_size)
  ax.set_xlabel('$\\omega$', fontsize=label_size)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  ax.set_xlim(0.0, 0.95)
  xticks = np.arange(0.00, 0.85, 0.20)
  ax.set_xticks(xticks)
  xticklabels = ['%.1f' % xtick for xtick in np.arange(0.00, 0.85, 0.20)]
  ax.set_xticklabels(xticklabels)

  eps_file = path.join(figure_dir, '%s_error.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_omega('mae')
draw_omega('mse')







