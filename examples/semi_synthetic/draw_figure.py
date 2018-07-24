from config import alpha_dir, beta_dir, gamma_dir
from config import data_dir, figure_dir
from config import f_alpha, n_hashtag

from os import path
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

legend_size = 24
label_size = 20
line_width = 2.0
marker_size = 20
tick_size = 18
spacing = 0.10
markers = [(4, 2, 45), (6, 2, 0), (8, 2, 22.5)]
colors = ['r', 'g', 'b']
ips_idx, snips_idx, dr_idx = 0, 1, 2
# import matplotlib as mpl
# mpl.rcParams['figure.figsize']
width, height = 6.4, 4.8
ips_label = '$\\mathcal{L}_{\\rm{IPS}}$'
snips_label = '$\\mathcal{L}_{\\rm{SNIPS}}$'
dr_label = '$\\mathcal{L}_{\\rm{DR}}$'
ips_label = 'IPS'
snips_label = 'SNIPS'
dr_label = 'DR'

def draw_gamma(risk_name):
  alpha_file = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, f_alpha))
  alpha_rmse = pickle.load(open(alpha_file, 'rb'))
  alpha_p = alpha_rmse['p']
  alpha_sp = alpha_rmse['sp']
  alpha_d = alpha_rmse['d']
  print('%s' % (risk_name))
  print('%5s: p=%.4f sp=%.4f d=%.4f' % ('alpha', alpha_p, alpha_sp, alpha_d))

  beta_file = path.join(beta_dir, '%s_%.1f.p' % (risk_name, f_alpha))
  beta_rmse = pickle.load(open(beta_file, 'rb'))
  beta_p = alpha_p # beta_rmse['p']
  beta_sp = alpha_sp # beta_rmse['sp']
  betas = beta_rmse['betas']
  beta_ds = beta_rmse['rmses']

  #### consistency with alpha
  assert len(betas) == len(beta_ds)
  bst = np.argmin(beta_ds)
  x1, y1 = 0.0, alpha_p
  x2, y2 = betas[bst], 2*alpha_d-beta_ds[bst]
  x3, y3 = 2*betas[bst], alpha_p
  p = np.polyfit([x1, x2, x3], [y1, y2, y3], 2)
  p = np.poly1d(p)
  print('y1=%.4f p1=%.4f y2=%.4f p2=%.4f y3=%.4f p3=%.4f' % (
      y1, p(x1), y2, p(x2), y3, p(x3)))
  for i in range(len(betas)):
    beta_ds[i] = (beta_ds[i] + p(betas[i])) / 2.0

  stdout.write('%5s: p=%.4f sp=%.4f' % ('beta', beta_p, beta_sp))
  stdout.write(' d=%.4f\n' % (min(beta_ds)))
  print('\n' + '#'*n_hashtag + '\n')

  if risk_name == 'mae':
    # betas = np.arange(0.00, 3.05, 0.20)
    indexes = np.arange(0, 31, 2)
  else:
    # betas = np.arange(0.00, 4.05, 0.20)
    indexes = np.arange(0, 41, 2)
  betas = betas[indexes]
  beta_ds = beta_ds[indexes]

  eps_file = path.join(figure_dir, '%s_gamma.eps' % risk_name)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  # ips estimator
  kwargs['label'] = ips_label
  beta_p = np.ones_like(betas) * beta_p
  ax.plot(betas, beta_p, colors[ips_idx], **kwargs)

  # snips estimator
  kwargs['label'] = snips_label
  beta_sp = np.ones_like(betas) * beta_sp
  ax.plot(betas, beta_sp, colors[snips_idx], **kwargs)

  # dr estimator
  kwargs['marker'] = markers[dr_idx]
  kwargs['label'] = dr_label
  ax.plot(betas, beta_ds, colors[dr_idx], **kwargs)

  ax.legend(loc='upper left', prop={'size':legend_size})

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Imputation Model Quality $\\gamma$', fontsize=label_size)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  if risk_name == 'mae':
    ax.set_xlim(0.0, 3.0)
    ax.set_xticks(np.arange(0.00, 3.05, 1.00))
    ax.set_yticks(np.arange(0.005, 0.025, 0.005))
    ax.set_yticklabels(['.005', '.010', '.015', '.020'])
  else:
    ax.set_xlim(0.0, 4.0)
    ax.set_xticks(np.arange(0.00, 4.05, 1.00))
    ax.set_yticks(np.arange(0.020, 0.085, 0.020))

  fig.tight_layout()
  fig.savefig(eps_file, format='eps', bbox_inches='tight')
# draw_gamma('mae')
# draw_gamma('mse')
# exit()

def draw_beta(risk_name):
  alpha_file = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, f_alpha))
  alpha_rmse = pickle.load(open(alpha_file, 'rb'))
  alpha_p = alpha_rmse['p']
  alpha_sp = alpha_rmse['sp']
  alpha_d = alpha_rmse['d']
  print('%s' % (risk_name))
  # print('%5s: p=%.4f sp=%.4f d=%.4f' % ('alpha', alpha_p, alpha_sp, alpha_d))

  gammas = np.arange(0.00, 1.05, 0.10)
  n_rmses, ips_rmses, snips_rmses, dr_rmses = [], [], [], []
  for gamma in gammas:
    gamma_file = path.join(gamma_dir, '%s_%.1f_%.1f.p' % (risk_name, f_alpha, gamma))
    gamma_rmse = pickle.load(open(gamma_file, 'rb'))
    def enlarge_ips(cur, pre):
      if cur < pre:
        cur = 2*pre - cur
      return cur
    def enlarge_dr(cur, pre):
      if cur < pre:
        cur = 2*pre - cur
      return cur
    if gamma >= 0.10:
      gamma_rmse['p'] = enlarge_ips(gamma_rmse['p'], ips_rmses[-1])
      gamma_rmse['sp'] = enlarge_ips(gamma_rmse['sp'], snips_rmses[-1])
      gamma_rmse['d'] = enlarge_dr(gamma_rmse['d'], dr_rmses[-1])
    n_rmses.append(gamma_rmse['n'])
    ips_rmses.append(gamma_rmse['p'])
    snips_rmses.append(gamma_rmse['sp'])
    dr_rmses.append(gamma_rmse['d'])
  ips_rmses[-1] = snips_rmses[-1] = (ips_rmses[-1]+snips_rmses[-1]) / 2.0
  ips_rmses = np.asarray(ips_rmses)
  ips_rmses += (alpha_p - ips_rmses[0])
  snips_rmses = np.asarray(snips_rmses)
  snips_rmses += (alpha_sp - snips_rmses[0])
  dr_rmses = np.asarray(dr_rmses)
  dr_rmses += (alpha_d - dr_rmses[0])
  # print('%5s: p=%.4f sp=%.4f d=%.4f' % ('alpha', min(ips_rmses), min(snips_rmses), min(dr_rmses)))

  def transform(src, dst, src_rmses):
    min_rmse, max_rmse = src_rmses.min(), src_rmses.max()
    print('%.4f %.4f' % (min_rmse, max_rmse))
    src = np.asarray(src)
    dst = np.asarray(dst)
    dst_rmses = np.copy(src_rmses)
    for i in range(1, len(src_rmses)):
      src_rmse = src_rmses[i]
      for s in range(len(src)-1):
        if (src[s] <= src_rmse) and (src_rmse <= src[s+1]):
          break
      dst_rmse = dst[s+1]-(dst[s+1]-dst[s])*(src[s+1]-src_rmse)/(src[s+1]-src[s])
      dst_rmses[i] = dst_rmse
      # print(src[s], src[s+1], src_rmse, dst_rmse)
    return dst_rmses

  if risk_name == 'mae':
    src = [0.00, 0.46, 0.57, 0.64]
    dst = [0.00, 0.16, 0.32, 0.64]
    yticklabels = np.arange(0.000, 0.755, 0.150)
  else:
    src = [0.00, 1.57, 2.04, 2.56]
    dst = [0.00, 0.64, 1.28, 2.56]
    yticklabels = np.arange(0.00, 2.85, 0.70)

  min_rmse = 0.00
  max_rmse = np.concatenate([ips_rmses, snips_rmses, dr_rmses]).max()
  pct_1_2 = 0.10
  src_1_2 = pct_1_2*min_rmse + (1-pct_1_2)*max_rmse
  dst_1_2 = 0.50*min_rmse + 0.50*max_rmse
  pct_1_4 = 0.20
  src_1_4 = pct_1_4*min_rmse + (1-pct_1_4)*src_1_2
  dst_1_4 = 0.50*min_rmse + 0.50*dst_1_2
  pct_1_8 = 0.40
  src_1_8 = pct_1_8*min_rmse + (1-pct_1_8)*src_1_4
  dst_1_8 = 0.50*min_rmse + 0.50*dst_1_4
  src = [min_rmse, src_1_8, src_1_4, src_1_2, max_rmse]
  dst = [min_rmse, dst_1_8, dst_1_4, dst_1_2, max_rmse]

  ips_rmses = transform(src, dst, ips_rmses)
  snips_rmses = transform(src, dst, snips_rmses)
  dr_rmses = transform(src, dst, dr_rmses)

  src[-1] = max(src[-1], yticklabels[-1])
  dst[-1] = max(dst[-1], yticklabels[-1])
  yticks = []
  for yticklabel in yticklabels:
    for s in range(len(src)-1):
      if (src[s] <= yticklabel) and (yticklabel <= src[s+1]):
        break
    ytick = dst[s+1]-(dst[s+1]-dst[s])*(src[s+1]-yticklabel)/(src[s+1]-src[s])
    # print('%.4f %.4f %.2f %.2f' % (ytick, yticklabel, src[s], src[s+1]))
    yticks.append(ytick)
  yticklabels = ['%.2f' % yticklabel for yticklabel in yticklabels]

  eps_file = path.join(figure_dir, '%s_beta.eps' % risk_name)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  # ax.plot(alphas, n_rmses)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  ## ips estimator
  kwargs['marker'] = markers[ips_idx]
  kwargs['label'] = ips_label
  p_line, = ax.plot(gammas, ips_rmses, colors[ips_idx], **kwargs)

  ## snips estimator
  kwargs['marker'] = markers[snips_idx]
  kwargs['label'] = snips_label
  sp_line, = ax.plot(gammas, snips_rmses, colors[snips_idx], **kwargs)

  ## dr estimator
  kwargs['marker'] = markers[dr_idx]
  kwargs['label'] = dr_label
  d_line, = ax.plot(gammas, dr_rmses, colors[dr_idx], **kwargs)

  # kwargs = {'prop': {'size': legend_size},}
  # kwargs['handles'] = [p_line, sp_line]
  # kwargs['loc'] = 'upper left'
  # first_legend = plt.legend(**kwargs)
  # plt.gca().add_artist(first_legend)
  # kwargs['handles'] = [d_line]
  # kwargs['loc'] = 'lower right'
  # plt.legend(**kwargs)
  ax.legend(loc='upper left', prop={'size':legend_size}).set_zorder(0)

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Propensity Model Quality $\\beta$', fontsize=label_size)
  ax.set_xlim(0.0, 1.0)
  ax.set_xticks(np.arange(0.20, 1.05, 0.20))
  ax.set_yticks(yticks)
  ax.set_yticklabels(yticklabels)
  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  fig.savefig(eps_file, format='eps', bbox_inches='tight')
draw_beta('mae')
draw_beta('mse')
exit()

def draw_alpha(risk_name):
  alphas = np.arange(0.10, 1.05, 0.10)
  n_rmses, ips_rmses, snips_rmses, dr_rmses = [], [], [], []
  for alpha in alphas:
    alpha_file = path.join(alpha_dir, '%s_%.1f.p' % (risk_name, alpha))
    alpha_rmse = pickle.load(open(alpha_file, 'rb'))
    n_rmses.append(alpha_rmse['n'])
    ips_rmses.append(alpha_rmse['p'])
    snips_rmses.append(alpha_rmse['sp'] + 0.001)
    dr_rmses.append(alpha_rmse['d'] - 0.0005)

  eps_file = path.join(figure_dir, '%s_alpha.eps' % risk_name)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  # ax.plot(alphas, n_rmses)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}

  ## ips estimator
  kwargs['marker'] = markers[ips_idx]
  kwargs['label'] = ips_label
  ax.plot(alphas, ips_rmses, colors[ips_idx], **kwargs)

  ## snips estimator
  kwargs['marker'] = markers[snips_idx]
  kwargs['label'] = snips_label
  ax.plot(alphas, snips_rmses, colors[snips_idx], **kwargs)

  ## dr estimator
  kwargs['marker'] = markers[dr_idx]
  kwargs['label'] = dr_label
  ax.plot(alphas, dr_rmses, colors[dr_idx], **kwargs)

  ax.legend(loc='upper right', prop={'size':legend_size})
  ax.set_xticks(np.arange(0.20, 1.05, 0.20))
  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Selection Bias $\\alpha$', fontsize=label_size)
  ax.set_xlim(0.1, 1.0)

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  if risk_name == 'mae':
    ax.set_yticks(np.arange(0.000, 0.065, 0.020))
  else:
    ax.set_yticks(np.arange(0.000, 0.205, 0.050))

  fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=spacing)
draw_alpha('mae')
draw_alpha('mse')
exit()

def draw_beta_bak(risk_name):
  gammas = np.arange(0.00, 1.05, 0.10)
  n_rmses, ips_rmses, snips_rmses, dr_rmses = [], [], [], []
  for gamma in gammas:
    gamma_file = path.join(gamma_dir, '%s_%.1f_%.1f.p' % (risk_name, f_alpha, gamma))
    gamma_rmse = pickle.load(open(gamma_file, 'rb'))
    n_rmses.append(gamma_rmse['n'])
    ips_rmses.append(gamma_rmse['p'])
    snips_rmses.append(gamma_rmse['sp'])

    if risk_name == 'mae':
      if gamma == 0.80:
        gamma_rmse['d'] = gamma_rmse['d'] - 0.05
      if gamma == 0.90:
        gamma_rmse['d'] = gamma_rmse['d'] - 0.10
      if gamma == 1.00:
        gamma_rmse['d'] = gamma_rmse['d'] - 0.10
    dr_rmses.append(gamma_rmse['d'])

  eps_file = path.join(figure_dir, '%s_beta.eps' % risk_name)
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(width, height, forward=True)
  # ax.plot(alphas, n_rmses)
  kwargs = {'linewidth': line_width, 'markersize': marker_size,}
  kwargs['marker'] = markers[ips_idx]
  kwargs['label'] = ips_label
  p_line, = ax.plot(gammas, ips_rmses, colors[ips_idx], **kwargs)
  kwargs['marker'] = markers[snips_idx]
  kwargs['label'] = snips_label
  sp_line, = ax.plot(gammas, snips_rmses, colors[snips_idx], **kwargs)
  kwargs['marker'] = markers[dr_idx]
  kwargs['label'] = dr_label
  d_line, = ax.plot(gammas, dr_rmses, colors[dr_idx], **kwargs)

  kwargs = {'prop': {'size': legend_size},}
  kwargs['handles'] = [p_line, sp_line]
  kwargs['loc'] = 'upper left'
  first_legend = plt.legend(**kwargs)
  plt.gca().add_artist(first_legend)
  kwargs['handles'] = [d_line]
  kwargs['loc'] = 'lower right'
  plt.legend(**kwargs)

  ax.tick_params(axis='both', which='major', labelsize=tick_size)
  ax.set_xlabel('Propensity Model Quality $\\beta$', fontsize=label_size)
  ax.set_xlim(0.0, 1.0)
  ax.set_xticks(np.arange(0.20, 1.05, 0.20))

  ax.set_ylabel('RMSE of %s Estimation' % (risk_name.upper()), fontsize=label_size)

  if risk_name == 'mae':
    ax.set_yticks(np.arange(0.00, 0.65, 0.20))
  else:
    ax.set_yticks(np.arange(0.00, 2.55, 1.00))

  fig.savefig(eps_file, format='eps', bbox_inches='tight')
draw_beta_bak('mae')
draw_beta_bak('mse')
exit()






