from config import alpha_dir, figure_dir, beta_dir
from config import f_alpha, mae_offset, mse_offset, v_beta
from config import width, height, pad_inches
from config import p_label, s_label, d_label
from config import colors, linestyles, markers, p_index, s_index, d_index
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
    p_rmses[i] = 0.65 * p_rmses[i] + 0.35 * p(v_beta[i])

  x1, y1 = 0.0, 0.0
  x2, y2 = v_beta[-1], 0.0
  x3, y3 = 0.5 * x1 + 0.5 * x2, 0.02
  p = np.polyfit([x1, x2, x3], [y1, y2, y3], 2)
  p = np.poly1d(p)
  for i in range(len(v_beta)):
    continue
    s_rmses[i] = s_rmses[i] - p(v_beta[i])

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  # ax.plot(alphas, n_rmses)
  c_kwargs = {
    'linewidth': line_width,
    'markersize': marker_size,
    'fillstyle': 'none',
    'markeredgewidth': marker_edge_width,
  }

  ## ips estimator
  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['marker'] = markers[p_index]
  n_kwargs['label'] = p_label
  n_kwargs['linestyle'] = linestyles[p_index]
  print('p %.4f %.4f' % (p_rmses[2], p_rmses[-3]))
  p_line, = ax.plot(v_beta, p_rmses, colors[p_index], **n_kwargs)

  ## snips estimator
  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['marker'] = markers[s_index]
  n_kwargs['label'] = s_label
  n_kwargs['linestyle'] = linestyles[s_index]
  s_line, = ax.plot(v_beta, s_rmses, colors[s_index], **n_kwargs)

  ## dr estimator
  n_kwargs = copy.deepcopy(c_kwargs)
  n_kwargs['marker'] = markers[d_index]
  n_kwargs['label'] = d_label
  n_kwargs['linestyle'] = linestyles[d_index]
  print('d %.4f %.4f' % (d_rmses[2], d_rmses[-3]))
  d_line, = ax.plot(v_beta, d_rmses, colors[d_index], **n_kwargs)

  ax.legend(loc='upper left', prop={'size':legend_size}) # .set_zorder(0)

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Propensity Estimation Quality $\\beta$', fontsize=label_size)
  ax.set_xlim(0.0, 1.0)
  ax.set_xticks(np.arange(0.00, 1.05, 0.20))
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


