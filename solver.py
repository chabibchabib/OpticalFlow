import numpy as np 
from scipy.ndimage.filters import convolve as filter2, laplace
##########################################
def solver(Ix,Iy,It,lmbda,u,v,itmax,tol):
    ''' This is a free-matrix solver to solve the problem A*dU=b related to the 2D optical flow problems 
    Parameters:
        -Ix,Iy: the spatial derivatives of the image
        -It: temporal derivative 
        -u,v: te displacements
        -lmbda: Tikhonov parameter
        -itmax: The maximum number of itérations
        -tol: tolerance
    Returns:
        the new step of displacements u1 and v1 

     '''
    K=0.25*np.array([[0, 1 ,0 ],[1, 0 ,1 ],[0, 1 ,0 ]]) # Laplacian Filter

    #K=np.array([[1/12, 1/6 ,1/12 ],[1/6, 0 ,1/6 ],[1/12, 1/6 ,1/12 ]]);
    u0=u; v0=v   #Initialization  

    alpha=4*lmbda
    D=alpha+Ix*Ix+Iy*Iy # denominator
    u1=np.zeros_like(u0) 
    v1=np.zeros_like(v0)
    for i in range(itmax):
        u0=filter2(u0,K)+laplace(u)/4
        v0=filter2(v0,K)+laplace(v)/4
        P=(Ix*u0+Iy*v0+It)
    
        u1=u0-Ix*P/D
        v1=v0-Iy*P/D
        u0=u1
        v0=v1

        if (np.linalg.norm((Ix*u+Iy*v+It)/np.linalg.norm(D) ) <= tol): 
            #print('iteration:',i)
            break;    

    return [u1,v1]

###################################################
''''P1=np.round(10*np.random.rand( 10,5))
P2=np.round(10*np.random.rand(10,5))
lmbda=2
u=np.random.rand(10,5)
v=np.random.rand(10,5)
itmax=10
tol=0.001
It=P2-P1
Ix,Iy=np.gradient(P1)
[u1,v1]=solver(Ix,Iy,It,lmbda,u,v,itmax,tol)
print('u\n',u)
print('v\n',v)
print('u1\n',u1)
print('v1\n',v1)'''