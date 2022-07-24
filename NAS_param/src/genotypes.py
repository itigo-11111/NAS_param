from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

PRIMITIVES_PARAM = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

MyNet = Genotype(
  normal=[
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_3x3', 0), 
    ('skip_connect', 2), 
    ('skip_connect', 2), 
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 0), 
    ('sep_conv_3x3', 1)], 
  normal_concat=range(2, 6), 
  reduce=[('max_pool_3x3', 0),
    ('sep_conv_5x5', 1),
    ('dil_conv_3x3', 2), 
    ('sep_conv_3x3', 1), 
    ('sep_conv_5x5', 2), 
    ('sep_conv_5x5', 1), 
    ('sep_conv_5x5', 1), 
    ('sep_conv_3x3', 0)], 
  reduce_concat=range(2, 6)
)

CarNet = Genotype(
  # normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0)], 
  # normal_concat=range(2, 6), 
  # reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0)], 
  # reduce_concat=range(2, 6)

  normal=[('skip_connect', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2)], 
  normal_concat=range(2, 6), 
  reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0)], 
  reduce_concat=range(2, 6)
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

min_param = Genotype(
  normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 2), ('max_pool_3x3', 4), ('avg_pool_3x3', 3)], 
  normal_concat=range(2, 6), 
  reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('skip_connect', 2), ('max_pool_3x3', 4), ('skip_connect', 3)], 
  reduce_concat=range(2, 6)
)

PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
PC_DARTS_image = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))

a1400000 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
a1500000 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
a1600000 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('skip_connect', 4), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))
a1700000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
a1800000 = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 2), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
a1900000 = Genotype(normal=[('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
a2000000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
a2100000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('dil_conv_5x5', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
a2200000 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
a2300000 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
a2400000 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
a2500000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
a2600000 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
a2700000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
a2800000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
a2900000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
a3000000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0)], reduce_concat=range(2, 6))
a3500000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 0), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
a4000000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
a4500000 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('skip_connect', 1)], reduce_concat=range(2, 6))
a5000000 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))

rens1 = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
rens2 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 3), ('dil_conv_3x3', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
rens3 = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 3), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
rens4 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 3), ('dil_conv_5x5', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
rens5 = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
rens10 = Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 0), ('skip_connect', 3), ('dil_conv_5x5', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
rens25 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))
rens30 = Genotype(normal=[('skip_connect', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 4)], reduce_concat=range(2, 6))
rens35 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_5x5', 3), ('skip_connect', 2), ('sep_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

cifar100_net = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 0), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))

PCDARTS = PC_DARTS_cifar

