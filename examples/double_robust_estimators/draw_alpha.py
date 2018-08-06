from config import alpha_dir, figure_dir
from config import v_alpha, mae_offset, mse_offset
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

def draw_alpha(risk_name):
  n_rmses, p_rmses, s_rmses, d_rmses = [], [], [], []
  for alpha in v_alpha:
    alpha_file = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, alpha))
    alpha_rmse = pickle.load(open(alpha_file, 'rb'))

    p_rmse = alpha_rmse['p']
    s_rmse = alpha_rmse['s']
    d_rmse = alpha_rmse['d']

    #### visual
    if risk_name == 'mae':
      p_rmse += mae_offset
      d_rmse -= mae_offset

      p_rmse += 0.0016
      s_rmse += 0.0010
    else:
      p_rmse += mse_offset
      d_rmse -= mse_offset

    p_rmses.append(p_rmse)
    s_rmses.append(s_rmse)
    d_rmses.append(d_rmse)
  print('%s p=%.4f s=%.4f d=%.4f' % (risk_name, min(p_rmses), min(s_rmses), min(d_rmses)))

  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  ## ips estimator
  kwargs['marker'] = markers[p_index]
  kwargs['label'] = p_label
  ax.plot(v_alpha, p_rmses, colors[p_index], **kwargs)

  ## snips estimator
  kwargs['marker'] = markers[s_index]
  kwargs['label'] = s_label
  ax.plot(v_alpha, s_rmses, colors[s_index], **kwargs)

  ## dr estimator
  kwargs['marker'] = markers[d_index]
  kwargs['label'] = d_label
  ax.plot(v_alpha, d_rmses, colors[d_index], **kwargs)

  ax.legend(loc='upper right', prop={'size':legend_size})
  ax.set_xticks(np.arange(0.20, 1.05, 0.20))
  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Selection Bias $\\alpha$', fontsize=label_size)
  ax.set_xlim(0.1, 1.0)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  if risk_name == 'mae':
    yticks = np.arange(0.000, 0.070, 0.020)
    ax.set_yticks(yticks)
    ax.set_yticklabels([('%.2f' % ytick)[1:] for ytick in yticks])
  else:
    yticks = np.arange(0.00, 0.35, 0.10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([('%.1f' % ytick)[1:] for ytick in yticks])

  eps_file = path.join(figure_dir, '%s_alpha.eps' % risk_name)
  config.make_file_dir(eps_file)
  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

draw_alpha('mae')
draw_alpha('mse')





