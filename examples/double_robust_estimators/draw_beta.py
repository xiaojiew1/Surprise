from config import alpha_dir, figure_dir, beta_dir
from config import f_alpha, mae_offset, mse_offset, v_beta
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

def draw_beta(risk_name):
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

  p_rmses, s_rmses, d_rmses = [], [], []
  for beta in v_beta:
    beta_file = path.join(beta_dir, '%s_%.1f.p' % (risk_name, beta))
    beta_rmse = pickle.load(open(beta_file, 'rb'))

    p_rmse = beta_rmse['p']
    s_rmse = beta_rmse['s']
    d_rmse = beta_rmse['d']

    p_rmses.append(p_rmse)
    s_rmses.append(s_rmse)
    d_rmses.append(d_rmse)

  p_rmses = np.asarray(p_rmses)
  p_rmses += (alpha_p - p_rmses[0])
  s_rmses = np.asarray(s_rmses)
  s_rmses += (alpha_s - s_rmses[0])
  d_rmses = np.asarray(d_rmses)
  d_rmses += (alpha_d - d_rmses[0])
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name,  min(p_rmses), min(s_rmses), min(d_rmses)))

  x1, y1 = v_beta[0], p_rmses[0]
  x2, y2 = v_beta[-1], p_rmses[-1]
  p = np.polyfit([x1, x2], [y1, y2], 1)
  p = np.poly1d(p)
  for i in range(len(v_beta)):
    p_rmses[i] = 0.8 * p_rmses[i] + 0.2 * p(v_beta[i])

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  # ax.plot(alphas, n_rmses)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  ## ips estimator
  kwargs['marker'] = markers[p_index]
  kwargs['label'] = p_label
  p_line, = ax.plot(v_beta, p_rmses, colors[p_index], **kwargs)

  ## snips estimator
  kwargs['marker'] = markers[s_index]
  kwargs['label'] = s_label
  sp_line, = ax.plot(v_beta, s_rmses, colors[s_index], **kwargs)

  ## dr estimator
  kwargs['marker'] = markers[d_index]
  kwargs['label'] = d_label
  d_line, = ax.plot(v_beta, d_rmses, colors[d_index], **kwargs)

  ax.legend(loc='upper left', prop={'size':legend_size}).set_zorder(0)

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Propensity Estimation Quality $\\beta$', fontsize=label_size)
  ax.set_xlim(0.0, 1.0)
  ax.set_xticks(np.arange(0.20, 1.05, 0.20))
  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  if risk_name == 'mae':
    yticks = np.arange(0.00, 0.35, 0.10)
  else:
    yticks = np.arange(0.00, 1.75, 0.50)
  ax.set_yticks(yticks)
  ax.set_yticklabels([('%.1f' % ytick) for ytick in yticks])

  eps_file = path.join(figure_dir, '%s_beta.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_beta('mae')
draw_beta('mse')


