import numpy as np

py_list = [[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 10],
           [11, 12, 13, 14, 15]]

np_array = np.array([[1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15]])

print(f'py_list: {py_list}')
print(f'np_array: {np_array}')
print('-------------------------------------')
print(f'py_list data_type: {type(py_list)}')
print(f'np_array data_type: {type(np_array)}')
print('-------------------------------------')
# print(f'py_list+5: {py_list+5}')                   Error
print(f'np_array+5: {np_array+5}')
print('-------------------------------------')
print(f'py_list*5 = {py_list*5}')
print(f'np_array*5 = {np_array*5}')
print('-------------------------------------')
print(f'py_list  + py_list : {py_list + py_list}')
print(f'np_array + np_array: {np_array + np_array}')
print('-------------------------------------')
print(f'py_list length: {len(py_list)}')
print(f'np_array length: {len(np_array)}')
print('-------------------------------------')
