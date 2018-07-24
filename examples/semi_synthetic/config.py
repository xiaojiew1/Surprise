from os import path
import os

def create_dir(out_dir):
  if not path.exists(out_dir):
    os.makedirs(out_dir)

data_dir = 'data'
create_dir(data_dir)
data_file = path.join(data_dir, 'semi_synthetic.txt')

figure_dir = path.join(data_dir, 'figure')
create_dir(figure_dir)

alpha_dir = path.join(data_dir, 'alpha')
beta_dir = path.join(data_dir, 'beta')
gamma_dir = path.join(data_dir, 'gamma')

max_rate = 5
min_rate = 1
sparsity = 0.05
n_trials = 50
n_hashtag = 64
f_alpha = 0.40

project_dir = path.expanduser('~/Projects')
music_dir = path.join(project_dir, 'drrec/data/music')
music_train = path.join(music_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
music_test = path.join(music_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
