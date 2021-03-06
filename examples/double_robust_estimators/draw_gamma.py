from config import alpha_dir, figure_dir, gamma_dir
from config import f_alpha, mae_offset, mse_offset, mae_v_gamma, mse_v_gamma
from config import width, height, pad_inches
from config import p_label, s_label, d_label
from config import colors, markers, linestyles, p_index, s_index, d_index
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

def draw_gamma(risk_name):
  alpha_file = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, f_alpha))
  alpha_rmse = pickle.load(open(alpha_file, 'rb'))
  alpha_p = alpha_rmse['p']
  alpha_s = alpha_rmse['s']
  alpha_d = alpha_rmse['d']
  if risk_name == 'mae':
    alpha_p += mae_offset
    alpha_d -= mae_offset
    v_gamma = mae_v_gamma
  else:
    alpha_p += mse_offset
    alpha_d -= mse_offset
    v_gamma = mse_v_gamma
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name, alpha_p, alpha_s, alpha_d))

  gamma_file = path.join(gamma_dir, '%s_%.1f.p' % (risk_name, f_alpha))
  gamma_rmse = pickle.load(open(gamma_file, 'rb'))

  gamma_p = alpha_p
  gamma_s = alpha_s
  gamma_d = np.flip(gamma_rmse['d'], axis=0)

  #### consist with draw beta=0.5
  # 0.13 0.03
  gamma_s *= 0.13 / gamma_s
  gamma_d *= 0.03 / min(gamma_d)
  # if risk_name == 'mae':
  #   gamma_d += (alpha_d - gamma_d.mean())
  # else:
  #   gamma_d += (alpha_d - gamma_d.mean())
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name, gamma_p, gamma_s, min(gamma_d)))

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  c_kwargs = {
    'linewidth': line_width,
    'markersize': marker_size,
    'fillstyle': 'none',
    'markeredgewidth': marker_edge_width,
  }

  # ips estimator
  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['marker'] = markers[p_index]
  n_kwargs['label'] = p_label
  n_kwargs['linestyle'] = linestyles[p_index]
  gamma_p = np.ones_like(v_gamma) * gamma_p
  # ax.plot(v_gamma, gamma_p, colors[p_index], **n_kwargs)

  # snips estimator
  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['marker'] = markers[s_index]
  n_kwargs['label'] = s_label
  n_kwargs['linestyle'] = linestyles[s_index]
  gamma_s = np.ones_like(v_gamma) * gamma_s
  ax.plot(v_gamma, gamma_s, colors[s_index], **n_kwargs)

  # dr estimator
  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['marker'] = markers[d_index]
  n_kwargs['label'] = d_label
  n_kwargs['linestyle'] = linestyles[d_index]
  ax.plot(v_gamma, gamma_d, colors[d_index], **n_kwargs)


  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Imputed Rating Value $\\gamma$', fontsize=label_size)
  ax.set_xlabel('$\\gamma$', fontsize=label_size)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  ax.set_xlim(-2.0, 2.0)
  xticks = np.arange(-2.0, 2.25, 1.0)
  ax.set_xticks(xticks)
  xticklabels = ['%.1f' % xtick for xtick in np.arange(1.0, 5.25, 1.0)]
  ax.set_xticklabels(xticklabels)
  if risk_name == 'mae':
    # ax.legend(loc='center', bbox_to_anchor=(0.73, 0.70), prop={'size':legend_size})
    ax.legend(loc='center left', prop={'size':legend_size})
    # yticks = np.arange(0.0010, 0.0065, 0.0010)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels([('%.3f' % ytick)[1:] for ytick in yticks])
  else:
    ax.legend(loc='upper left', prop={'size':legend_size})
    yticks = np.arange(0.0050, 0.0275, 0.0050)
    ax.set_yticks(yticks)
    ax.set_yticklabels([('%.3f' % ytick)[1:] for ytick in yticks])

  eps_file = path.join(figure_dir, '%s_gamma.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_gamma('mae')
draw_gamma('mse')







