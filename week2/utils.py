import numpy as py

def size(matrix):
  try:
    return py.size(matrix, 0), py.size(matrix, 1)
  except IndexError as e:
    return py.size(matrix, 0), -1