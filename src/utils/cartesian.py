import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import jax 
import jax.numpy as jnp
from functools import partial

def stack_object(object, dimension):
    stacked_object = []
    for i in range(dimension):
        stacked_object.append(object)
    return stacked_object

def product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i]= a
    return arr.reshape(-1, la)

def power(interval, dimension=2):
    stacked_interval = stack_object(interval, dimension) 
    prod = product(*stacked_interval)
    return prod 
