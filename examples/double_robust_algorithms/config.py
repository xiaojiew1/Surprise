from decimal import Decimal
from os import path

import os

def make_dir(dirpath):
  if not path.exists(dirpath):
    os.makedirs(dirpath)

def make_file_dir(filepath):
  dirpath = path.dirname(filepath)
  make_dir(dirpath)

def sciformat(v):
  v = '%e' % Decimal(v)
  s_index = v.rfind('e')
  e_index = s_index - 1
  while v[e_index] == '0':
    e_index -= 1
    if v[e_index] == '.':
      e_index -= 1
      break
  e_index += 1
  v = v[:e_index] + v[s_index:]
  v = v.replace('e+0', 'e+').replace('e-0', 'e-')
  return v

def stringify(kwargs):
  kwargs_str = ''
  keys = sorted(kwargs.keys())
  for i in range(len(keys)):
    k = keys[i]
    v = kwargs[k]
    kwargs_str += k.upper().replace(separator, concatenator)
    kwargs_str += separator
    if type(v) == bool:
      kwargs_str += str(v).lower()
    elif type(v) == float:
      kwargs_str += sciformat(v)
    elif type(v) == int:
      kwargs_str += str(v)
    else:
      raise Exception('unknown format %s' % type(v))
    if i < len(keys) - 1:
      kwargs_str += separator
  return kwargs_str

def dictify(kwargs_str):
  kwargs = {}
  fields = kwargs_str.split(separator)
  for i in range(0, len(fields), 2):
    k = fields[i].lower().replace(concatenator, separator)
    v = fields[i+1]
    if k == 'n_factors':
      v = int(v)
    elif k == 'n_epochs':
      v = int(v)
    elif k == 'biased':
      if v == 'true':
        v = True
      elif v == 'false':
        v = False
      else:
        raise Exception('unknown value %s' % v)
    elif k == 'reg_all':
      v = float(v)
    elif k == 'lr_all':
      v = float(v)
    else:
      raise Exception('unknown key %s' % k)
    kwargs[k] = v
  return kwargs

def read_gsearch(infile):
  err_kwargs =  []
  if path.isfile(infile):
    with open(infile) as fin:
      for line in fin.readlines():
        fields = line.split()
        mae, mse, kwargs_str = fields[0], fields[1], fields[2]
        err_kwargs.append((float(mae), float(mse), kwargs_str))
  kwargs_set = set([t[2] for t in err_kwargs])
  return err_kwargs, kwargs_set

def write_gsearch(err_kwargs, outfile):
  make_file_dir(outfile)
  with open(outfile, 'w') as fout:
    for mae, mse, kwargs_str in err_kwargs:
      fout.write('%.16f %.16f %s\n' % (mae, mse, kwargs_str))

tmp_dir = 'tmp'
dnld_dir = path.expanduser('~/Downloads')
data_dir = 'data'
figure_dir = path.join(data_dir, 'figure')
curve_dir = path.join(data_dir, 'curve')
# rec_coat_file = path.join(data_dir, 'dev_coat.p')
# rec_song_file = path.join(data_dir, 'dev_song.p')
rec_coat_file = path.join(data_dir, 'rec_coat.p')
rec_song_file = path.join(data_dir, 'rec_song.p')

separator = '_'
concatenator = '-'

#### draw
width, height = 6.4, 4.8
legend_size = 26
label_size = 20
line_width = 2.0
marker_size = 20
tick_size = 18
pad_inches = 0.10

if __name__ == '__main__':
  print(sciformat(7.123456))
  print(sciformat(6.12345))
  print(sciformat(5.1234))
  print(sciformat(4.123))
  print(sciformat(3.12))
  print(sciformat(2.1))
  print(sciformat(1.0))

  print(sciformat(71.23456))
  print(sciformat(712.3456))
  print(sciformat(7123.456))
  print(sciformat(71234.56))
  print(sciformat(712345.6))
  print(sciformat(7123456.))

  print(sciformat(0.000001))
  print(sciformat(0.0000000001))

  print(sciformat(1000000))
  print(sciformat(10000000000))

  print(sciformat(0.0))




