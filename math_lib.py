from math import (
    exp
)

def dot_product(vector1, vector2):
    prod = 0
    for i in range(len(vector1)):
        prod += vector1[i] * vector2[i]
    return prod

def sigmoid(x):
    return (1.0 / (1.0 + exp(-x)))