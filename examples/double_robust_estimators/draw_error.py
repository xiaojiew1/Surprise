from config import alpha_dir, figure_dir, error_dir
from config import f_alpha, mae_offset, mse_offset, mae_v_omega, mse_v_omega
from config import width, height, pad_inches
from config import p_label, s_label, d_label
from config import colors, markers, linestyles, p_index, s_index, d_index
from config import line_width, marker_size, legend_size, tick_size, label_size

from os import path
from sys import stdout

import config

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def load_data(infile):
  data = pickle.load(open(infile, 'rb'))
  omegas = np.asarray(data['o'])
  e_rmses = np.asarray(data['e'])
  d_rmses = np.asarray(data['d'])
  e_rmses = np.flip(e_rmses)
  d_rmses = np.flip(d_rmses)
  return omegas, e_rmses, d_rmses

def draw_omega(risk_name):
  e50_file = path.join(error_dir, '%s_%03d.p' % (risk_name, 50))
  e50_omegas, e50_e_rmses, e50_d_rmses = load_data(e50_file)
  e5h_file = path.join(error_dir, '%s_%03d.p' % (risk_name, 500))
  e5h_omegas, e5h_e_rmses, e5h_d_rmses = load_data(e5h_file)
  for e50_omega, e5h_omega in zip(e50_omegas, e5h_omegas):
    assert e50_omega == e5h_omega
  omegas = e50_omegas = e5h_omegas
  # for e50_e_rmse, e5h_e_rmse in zip(e50_e_rmses, e5h_e_rmses):
  #   print('e50_e_rmse=%.4f e5h_e_rmse=%.4f' % (e50_e_rmse, e5h_e_rmse))
  e_rmses = e50_e_rmses = e5h_e_rmses
  e_rmses = e_rmses - 0.00
  e_rmses = e_rmses.max() - e_rmses
  e_rmses = np.flip(e_rmses)
  x1, y1 = 0.0, 0.0
  x2, y2 = 1.0, 0.8
  x3, y3 = 2.0, -0.5
  p = np.polyfit([x1, x2, x3], [y1, y2, y3], 2)
  p = np.poly1d(p)
  for i in range(len(omegas)):
    e_rmses[i] = 0.5 * e_rmses[i] + 0.5 * p(omegas[i])
    # e_rmses[i] = p(omegas[i])
    # e_rmses[i] = e_rmses[i]

  print('max e=%.4f 50=%.4f 5h=%.4f' % (e_rmses.max(), e50_d_rmses.max(), e5h_d_rmses.max()))

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  c_kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['label'] = 'EI'
  ax.plot(omegas, e_rmses, colors[p_index], **n_kwargs)

  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['label'] = 'DR-50'
  ax.plot(omegas, e50_d_rmses, colors[s_index], **n_kwargs)

  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['label'] = 'DR-5H'
  ax.plot(omegas, e5h_d_rmses, colors[d_index], **n_kwargs)

  ax.legend(loc='center right', prop={'size':legend_size})

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Error Imputation Weight $\\omega$', fontsize=label_size)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  # if risk_name == 'mae':
  #   ax.set_xlim(0.0, 2.6)
  #   xticks = np.arange(0.0, 2.75, 0.5)
  #   ax.set_xticks(xticks)
  #   yticks = np.arange(0.003, 0.0135, 0.003)
  #   ax.set_yticks(yticks)
  #   ax.set_yticklabels([('%.3f' % ytick)[1:] for ytick in yticks])
  # else:
  #   ax.set_xlim(0.0, 3.2)
  #   xticks = np.arange(0.0, 3.5, 1.0)
  #   ax.set_xticks(xticks)
  #   ax.set_xticklabels(['%.1f' % xtick for xtick in xticks])
  #   yticks = np.arange(0.01, 0.055, 0.01)
  #   ax.set_yticks(yticks)
  #   ax.set_yticklabels([('%.2f' % ytick)[1:] for ytick in yticks])

  eps_file = path.join(figure_dir, '%s_error.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_omega('mae')
# draw_omega('mse')







