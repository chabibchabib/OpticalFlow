import numpy as np
from numba import float32, vectorize
################################################
@vectorize([float32(float32,float32)])
def fct1(x, lmbda):
    '''This function will add 8*lmbda to a matrix float32 using vectorize 
    Parameters:
        -x: a float32 matrix or a float32 scalar
        -lmbda: a float32 scalar or a float32 matrix
    returns:
        -x +8*lmbda (float32)'''
    return x +8*lmbda
################################################
@vectorize([float32(float32,float32)])
def fct2(x, y):
    '''computes the elemenwise product of two matrices float32 using vectorize 
    Parameters:
        -x: a float32 matrix or a float32 scalar
        -y a float32 matrix or a float32 scalar
    returns:
        -the elemenwise product'''
    return x * y
################################################   
def Mx(Ix2,Iy2,lmbda,N,M,x):
    '''
    Returns  the y=P^-1*x where P is a preconditionner 
    Parameters:
        -N,M: The image shape
        - Ix2: Ix.^2 where Ix is the spatial derivative
        - Iy2: Iy.^2 where Iy is the spatial derivative
        - Ixy: Ix.*Iy
        -lmbda: Parameter of Tikhonov
        - x is a 2*N*M vector 
    Returns:
        -the product P^-1*x 
    '''
    pix=N*M
    x1=np.reshape(x[:pix],(N,M),order='F')
    x2=np.reshape(x[pix:2*pix],(N,M),order='F')
    x1=((Ix2+8*lmbda))*x1
    x2=((Iy2+8*lmbda))*x2
    '''x1=fct2(x1,fct1(Ix2,lmbda))
    x2=fct2(x2,fct1(Iy2,lmbda))'''
    return np.vstack((np.reshape(x1,(pix,1),order='F'),np.reshape(x2,(pix,1),order='F')))

    



