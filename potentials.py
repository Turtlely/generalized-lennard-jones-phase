import numpy as np

# Lennard Jones function
def V(r,A=1,B=1):
    return np.subtract(A*np.power(r,-12, out=np.zeros_like(r), where=r!=0), B*np.power(r,-6, out=np.zeros_like(r), where=r!=0))

def F(r,A=1,B=1):
    return np.subtract(12*A*np.power(r,-13, out=np.zeros_like(r), where=r!=0), 6*B*np.power(r,-7, out=np.zeros_like(r), where=r!=0))