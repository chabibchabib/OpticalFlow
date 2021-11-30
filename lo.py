import numpy as np 
from scipy.ndimage import laplace
from numba import njit, prange
import pep8

#######################################################################
@njit(parallel=True)
def laplacien2(Im):
    ''' Laplacien2 computes the laplacian of an Image  with reflecting boundary conditions using numba 
    Parameters:
        -Im: an Image
    Returns:
        -The laplacian'''
    N,M=Im.shape
    res=np.zeros((N,M)) 
    for i in prange(1,N-1):
        for j in prange(1,M-1):
            res[i,j]=Im[i-1,j]+Im[i+1,j]-4*Im[i,j]+Im[i,j-1]+Im[i,j+1]


    return res
#######################################################################
def mv(U,lmbda,Ix,Iy,N,M):
    '''The funcion mv will be used in order to avoide the construction of the matrix A
    In this function we will learn the krylov solver how compute the product of a matrix A and a vector U.
    Parameters:
        -N,M: are number of rows and columns of the images
        -U: is a vector of size 2*N*M
        -Ix,Iy: the  spatial derivatives of the image
        -lmbda: The regularization of Tikhonov
    Returns:
        the  product(vector) of A*U '''
    u0=U[:N*M]
    v0=U[N*M:]
    u0=np.reshape(u0,(N,M),order="F")
    v0=np.reshape(v0,(N,M),order="F")
    Ix2=Ix*Ix
    Iy2=Iy*Iy
    u1=Ix2*u0+Ix*Iy*v0-2*lmbda*laplace(u0)
    v1=Iy2*v0+Ix*Iy*u0-2*lmbda*laplace(v0)
    v1=np.reshape(v1,(N*M,1),order='F')
    u1=np.reshape(u1,(N*M,1),order='F')
    return np.vstack((u1,v1))

#############################################################