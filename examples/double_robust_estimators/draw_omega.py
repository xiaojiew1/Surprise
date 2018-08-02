from config import alpha_dir, figure_dir, omega_dir
from config import f_alpha, mae_offset, mse_offset, mae_v_omega, mse_v_omega
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
    v_omega = mae_v_omega
  else:
    alpha_p += mse_offset
    alpha_d -= mse_offset
    v_omega = mse_v_omega
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name, alpha_p, alpha_s, alpha_d))

  omega_file = path.join(omega_dir, '%s_%.1f.p' % (risk_name, f_alpha))
  omega_rmse = pickle.load(open(omega_file, 'rb'))

  omega_p = alpha_p
  omega_s = alpha_s
  omega_d = omega_rmse['d']

  #### consistency with alpha
  assert len(v_omega) == len(omega_d)
  m_index = np.argmin(omega_d)
  x1, y1 = 0.0, alpha_p
  x2, y2 = v_omega[m_index], 2 * alpha_d - omega_d[m_index]
  x3, y3 = 2 * v_omega[m_index], alpha_p
  p = np.polyfit([x1, x2, x3], [y1, y2, y3], 2)
  p = np.poly1d(p)
  print('(%.4f, %.4f) (%.4f, %.4f) (%.4f, %.4f)' % (y1, p(x1), y2, p(x2), y3, p(x3)))
  for i in range(len(v_omega)):
    omega_d[i] = (omega_d[i] + p(v_omega[i])) / 2.0

  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name, omega_p, omega_s, min(omega_d)))

  if risk_name == 'mae':
    indexes = np.arange(0, 27, 2)
  else:
    indexes = np.arange(0, 33, 2)
  v_omega = v_omega[indexes]
  omega_d = omega_d[indexes]

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  # ips estimator
  kwargs['label'] = p_label
  omega_p = np.ones_like(v_omega) * omega_p
  ax.plot(v_omega, omega_p, colors[p_index], **kwargs)

  # snips estimator
  kwargs['label'] = s_label
  omega_s = np.ones_like(v_omega) * omega_s
  ax.plot(v_omega, omega_s, colors[s_index], **kwargs)

  # dr estimator
  kwargs['marker'] = markers[d_index]
  kwargs['label'] = d_label
  ax.plot(v_omega, omega_d, colors[d_index], **kwargs)

  ax.legend(loc='upper left', prop={'size':legend_size})

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Error Imputation Weight $\\omega$', fontsize=label_size)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  if risk_name == 'mae':
    ax.set_xlim(0.0, 2.6)
    xticks = np.arange(0.0, 2.75, 0.5)
    ax.set_xticks(xticks)
    yticks = np.arange(0.003, 0.0135, 0.003)
    ax.set_yticks(yticks)
    ax.set_yticklabels([('%.3f' % ytick)[1:] for ytick in yticks])
  else:
    ax.set_xlim(0.0, 3.2)
    xticks = np.arange(0.0, 3.5, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.1f' % xtick for xtick in xticks])
    yticks = np.arange(0.01, 0.055, 0.01)
    ax.set_yticks(yticks)
    ax.set_yticklabels([('%.2f' % ytick)[1:] for ytick in yticks])

  eps_file = path.join(figure_dir, '%s_omega.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_gamma('mae')
draw_gamma('mse')






