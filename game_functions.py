###############################################################################################
# Standalone functions for the 2048 game. 
###############################################################################################
import numpy as np 
from itertools import groupby, chain

def sum_elements_left(arr): 
    groupby_array =[list(g) for k , g in groupby(arr)]
    new = []
    for g in groupby_array: 
        if len(g)==3 and g[0] != 0: 
            new.append([g[0]+g[1]] + g[2:])
        elif len(g)==4 and g[0] != 0: 
            new.append([g[0] + g[1]] + [g[2] + g[3]])
        elif len(g) == 2 and g[0]!=0: 
            new.append([sum(g)])
        else: 
            new.append(g)
    new_chained = np.array(list(chain(*new)))
    return new_chained

def sum_elements_right(arr): 
    arr = np.flip(arr)
    groupby_array =[list(g) for k , g in groupby(arr)]
    new = []
    for g in groupby_array: 
        if len(g)==3 and g[0] != 0: 
            new.append([g[0]+g[1]] + g[2:])
        elif len(g)==4 and g[0] != 0: 
            new.append([g[0] + g[1]] + [g[2] + g[3]])
        elif len(g) == 2 and g[0]!=0: 
            new.append([sum(g)])
        else: 
            new.append(g)
    new_chained = np.flip(np.array(list(chain(*new))))
    return new_chained

def new_tile(arr): 
    zero_indices = np.where(arr == 0 )
    insert_two = (np.random.uniform() <= 0.9)
    position_of_new_tile = np.random.choice(len(zero_indices[0]))
    new_tile_index = (zero_indices[0][position_of_new_tile] , zero_indices[1][position_of_new_tile])
    arr_new = arr.copy() 
    if insert_two == True: 
        arr_new[new_tile_index] += 2 
    else: 
        arr_new[new_tile_index] += 4
    return arr_new

def push_left(arr):
    shifted_arrays = []
    for i in range(arr.shape[0]):
        arr_slice = arr[i]
        non_zero = arr_slice[arr_slice!=0]
        summed = sum_elements_left(non_zero)
        summed_pad = np.concatenate((summed , np.zeros(4 - len(summed))))
        shifted_arrays.append(summed_pad)
    shifted_arrays = np.array(shifted_arrays, dtype = np.int64)
    if (shifted_arrays == arr).all():
        arr = shifted_arrays
    else: 
        shifted_arrays_new = new_tile(shifted_arrays)
        arr = shifted_arrays_new
    return arr

def push_right(arr): 
    shifted_arrays = []
    for i in range(arr.shape[0]):
        arr_slice = arr[i]
        non_zero = arr_slice[arr_slice!=0]
        summed = sum_elements_right(non_zero)
        summed_pad = np.concatenate((np.zeros(4 - len(summed)), summed ))
        shifted_arrays.append(summed_pad)
    shifted_arrays = np.array(shifted_arrays, dtype = np.int64)
    if (shifted_arrays == arr).all():
        arr = shifted_arrays
    else: 
        shifted_arrays_new = new_tile(shifted_arrays)
        arr = shifted_arrays_new
    return arr

def push_up(arr): 
    shifted_arrays = []
    for i in range(arr.shape[0]):
        arr_slice = arr[: , i]
        non_zero = arr_slice[arr_slice!=0]
        summed = sum_elements_left(non_zero)
        summed_pad = np.concatenate((summed, np.zeros(4 - len(summed)) ))
        shifted_arrays.append(summed_pad)
    shifted_arrays =  np.array(shifted_arrays, dtype = np.int64).T
    if (shifted_arrays == arr).all():
        arr = shifted_arrays
    else: 
        shifted_arrays_new = new_tile(shifted_arrays)
        arr = shifted_arrays_new
    return arr 

def push_down(arr):
    shifted_arrays = []
    for i in range(arr.shape[0]):
        arr_slice = arr[: , i]
        non_zero = arr_slice[arr_slice!=0]
        summed = sum_elements_right(non_zero)
        summed_pad = np.concatenate((np.zeros(4 - len(summed)), summed  ))
        shifted_arrays.append(summed_pad)
    shifted_arrays = np.array(shifted_arrays, dtype = np.int64).T
    if (shifted_arrays == arr).all():
        arr = shifted_arrays
    else: 
        shifted_arrays_new = new_tile(shifted_arrays)
        arr = shifted_arrays_new
    return arr