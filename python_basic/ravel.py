import numpy as np

array = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15]])
print(f'array: {array}')
print(f'type: {type(array)}')
print(f'dimension : {array.ndim}')
print(f'shape : {array.shape}')
print(f'array length: {len(array)}')
print(f'first component: {array[0]}')
print(f'first component with slicing(array[0,:]): {array[0,:]}')
print(f'array[:,0:3]: {array[:,0:3]}')
print(f'array[0:2,2:5] : {array[0:2,2:5]}')
print(f'*10 = {array * 10}')


array = array.ravel()
print(f'array: {array}')
print(f'type: {type(array)}')
print(f'dimension : {array.ndim}')
print(f'shape : {array.shape}')
print(f'array length: {len(array)}')
