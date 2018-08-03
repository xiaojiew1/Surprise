from config import alpha_dir, figure_dir, gamma_dir
from config import f_alpha, mae_offset, mse_offset, v_gamma
from config import width, height, pad_inches
from config import p_label, s_label, d_label
from config import colors, markers, p_index, s_index, d_index
from config import line_width, marker_size, legend_size, tick_size, label_size

from os import path
from sys import stdout

import config

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
  else:
    alpha_p += mse_offset
    alpha_d -= mse_offset
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name, alpha_p, alpha_s, alpha_d))

  gammas, p_rmses, s_rmses, d_rmses = [], [], [], []
  for gamma in v_gamma:
    gamma_file = path.join(gamma_dir, '%s_%.1f.p' % (risk_name, gamma))
    if not path.isfile(gamma_file):
      continue
    gamma_rmse = pickle.load(open(gamma_file, 'rb'))

    p_rmse = gamma_rmse['p']
    s_rmse = gamma_rmse['s']
    d_rmse = gamma_rmse['d']

    gammas.append(gamma)
    p_rmses.append(p_rmse)
    s_rmses.append(s_rmse)
    d_rmses.append(d_rmse)

  p_rmses = np.asarray(p_rmses)
  # p_rmses += (alpha_p - p_rmses[0])
  s_rmses = np.asarray(s_rmses)
  # s_rmses += (alpha_s - s_rmses[0])
  d_rmses = np.asarray(d_rmses)
  # d_rmses += (alpha_d - d_rmses[0])
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name,  min(p_rmses), min(s_rmses), min(d_rmses)))

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  # ips estimator
  kwargs['label'] = p_label
  # omega_p = np.ones_like(v_omega) * omega_p
  ax.plot(gammas, p_rmses, colors[p_index], **kwargs)

  # snips estimator
  kwargs['label'] = s_label
  # omega_s = np.ones_like(v_omega) * omega_s
  # ax.plot(gammas, s_rmses, colors[s_index], **kwargs)

  # dr estimator
  kwargs['marker'] = markers[d_index]
  kwargs['label'] = d_label
  ax.plot(gammas, d_rmses, colors[d_index], **kwargs)

  # ax.legend(loc='upper left', prop={'size':legend_size})

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

  eps_file = path.join(figure_dir, '%s_gamma.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_gamma('mae')
draw_gamma('mse')







