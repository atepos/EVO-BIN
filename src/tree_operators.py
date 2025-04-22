"""
tree_operators.py

Basic arithmetic and mathematical primitives for use in genetic programming.
Provides safe implementations of common operators and functions:
  - Binary operators: mul, add, sub, div, minimum, maximum, pow, if_then_else
  - Unary functions: log, sqrt, asin, acos, exp, tan, neg

Each function handles edge‐cases (zero division, domain limits, overflows) gracefully
to ensure robustness when evolving or evaluating expression trees.

Author:      Petr Kaška
Created:     2025-02-22
"""
import math

def mul(a, b):
    return a * b

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def minimum(a, b):
    return min(a, b)

def maximum(a, b):
    return max(a, b)

def div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1.0
    
def log(x):
    return math.log(abs(x) + 1e-10)

def sqrt(x):
    return math.sqrt(max(x, 0))

def asin(x):
    return math.asin(max(-1, min(1, x)))

def acos(x):
    return math.acos(max(-1, min(1, x)))

def if_then_else(condition, out1, out2):
    return out1 if condition >= 0 else out2

def exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')

def pow(x, y):
    try:
        return math.pow(x, y)
    except Exception:
        return 1.0

def tan(x):
    try:
        return math.tan(x)
    except Exception:
        return 1.0
    
def neg(x):
    return -x