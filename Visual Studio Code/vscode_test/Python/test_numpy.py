import numpy as np
from numpy.linalg import inv, qr


arr = np.random.randn(2, 2)
print(arr)
print('----------snip----------')
print(arr.repeat(2, axis=1))
print('----------snip----------')
# tile的功能是沿指定轴堆叠数组的副本
print(np.tile(arr, 2))   # 对于标量是水平铺设的
print('----------snip----------')
print(np.tile(arr, (3, 2)))
print('----------snip----------')
