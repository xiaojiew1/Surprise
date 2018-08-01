from decimal import Decimal

def sciformat(v):
  v = '%e' % Decimal(v)
  s_index = v.rfind('e')
  e_index = s_index - 1
  while v[e_index] == '0' or v[e_index] == '.':
    e_index -= 1
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
    kwargs_str += k.upper() + '_'
    if type(v) == bool:
      kwargs_str += str(v).lower()
    elif type(v) == float:
      kwargs_str += sciformat(v)
    elif type(v) == int:
      kwargs_str += str(v)
    else:
      raise Exception('unknown format %s' % type(v))
    if i < len(keys) - 1:
      kwargs_str += '_'
  return kwargs_str

tmp_dir = 'tmp'

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







