from operator import matmul
import numpy as np 
from scipy.ndimage import laplace
from time import time as tm
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sparse 
from numpy import dot
from numpy import matmul as mul
from numpy.lib.scimath import sqrt
from numba import vectorize, float32
from precond import Mx
##########################################################################
def my_dot(Ix2,Iy2,Ixy,lmbda, U,N,M):
    ''' Computing The paroduct Matrix-vector without constructing the matrix A
    Parameters:
    
    -N,M: shape of the image
    -U: a 2*N*M vector
    -Ix2: Ix^2, where Ix is the spatial derivative of the image
    -Iy2: Iy^2, where Iy is the spatial derivative of the image
    -Ix2: Ix*Iy elementwise product of Ix and Iy
    Returns:
    a vector res=A*U '''
    npix=N*M
    u=np.reshape(U[:npix],(N,M),order='F')
    v=np.reshape(U[npix:2*npix],(N,M),order='F')  
    #code.interact(local=locals()) 
    u1=u*Ix2+Ixy*v-2*lmbda*laplace(u)
    v1=v*Iy2+Ixy*u-2*lmbda*laplace(v)

    u1=np.reshape(u1,(npix,1),order='F')
    v1=np.reshape(v1,(npix,1),order='F')
    res=np.vstack((u1,v1))
    return res
###########################################
@vectorize([float32(float32, float32,float32,float32)])
def fct2(x, y,z,w):
    ''' Compute for 4 float32 matrices x,y,z and w  the vectorized formula: x*y+z*w using numba
    -x: matrix
    -y: matrix
    -z: matrix
    -w: matrix
    Returns: 
    x*y+z*w '''
    return x * y+z*w

def my_dot2(Ix2,Iy2,Ixy,lmbda, U,N,M):
    ''' Computing The paroduct float32 Matrix-vector without constructing the matrix A and using numba
    Parameters:
    
    -N,M: shape of the image
    -U: a 2*N*M vector
    -Ix2: Ix^2, where Ix is the spatial derivative of the image
    -Iy2: Iy^2, where Iy is the spatial derivative of the image
    -Ixy: Ix*Iy elementwise product of Ix and Iy
    Returns:
    a vector res=A*U '''
    npix=N*M
    u=np.reshape(U[:npix],(N,M),order='F')
    v=np.reshape(U[npix:2*npix],(N,M),order='F')  
    u1=fct2(u,Ix2,Ixy,v)-2*lmbda*laplace(u)
    v1=fct2(v,Iy2,Ixy,u)-2*lmbda*laplace(v)

    u1=np.reshape(u1,(npix,1),order='F')
    v1=np.reshape(v1,(npix,1),order='F')
    res=np.empty((2*npix,1),dtype=np.float32)
    res[:npix]=u1
    res[npix:2*npix]=v1
    #np.vstack((u1,v1))
    return res


#####################################################################
def minres(Ix2,Iy2,Ixy,lmbda,b,maxiter,rtol,N,M,method):
    '''minres solves the 2*N*M  system of linear equations Ax = b
    where A is a symmetric matrix, and b is a given vector. A will not be constructed
    a preconditionner will be used to solve the system.
    Parameters: 
    -N,M: shape of the image
    -U: a 2*N*M vector
    -Ix2: Ix^2, where Ix is the spatial derivative of the image
    -Iy2: Iy^2, where Iy is the spatial derivative of the image
    -Ixy: Ix*Iy elementwise product of Ix and Iy
    -lmbda: Parameter of Tikhonov
    -b: the left-hand term 
    -maxiter:
    -rtol:estimates norm(r_k)
    -maxiter: max of iterations
    -method: the method used to compute the product Matrix-vector if method==1: the my_dot will be used, if ==2 my_dot2 will be chosen 
    and numpy vectorization will be used
                FOR more details: C. C. Paige and M. A. Saunders (1975),Solution of sparse indefinite systems of linear equations,SIAM J. Numer. Anal. 12(4), pp. 617-629.
    Returns:
        x: the solution of the problem  

      '''
    #Initialize
    eps=1e-11
    realmax=1.7977e+308
    istop=0; Anorm=0; Acond=0; rnorm=0; ynorm=0
    y=b
    r1=b
    '''if type(precond)=='numpy.ndarray':'''
    #y=mul(precond,b)
    y=Mx(Ix2,Iy2,lmbda,N,M,b)

    beta1=np.sum(b*y)
    if (beta1<=0):
        istop=9
    beta1=sqrt(beta1)
    oldb=0; beta=beta1; dbar=0; epsln=0; qrnorm=beta1; phibar=beta1; rhs1=beta1;
    rhs2=0; tnorm2=0; gmax=0; gmin=realmax
    cs=-1; sn=0;
    w=np.zeros_like(r1); w2=np.zeros_like(r1); x=np.zeros_like(r1)
    r2=r1
    itn=0

    #Main Loop
    while (itn <maxiter):
        itn=itn+1
        s=1/beta
        v=s*y
        #y=mul(A,v)
        if method==1:
            y=my_dot(Ix2,Iy2,Ixy,lmbda, v,N,M)

        if method==2:
            y=my_dot2(Ix2,Iy2,Ixy,lmbda, v,N,M)
        if (itn >=2):
            y=y-(beta/oldb)*r1
        alpha=np.sum(v*y)
        y=(-alpha/beta)*r2+y
        r1=r2
        r2=y
        '''if precond!=None:'''
        #y=mul(precond,r2)
        y=Mx(Ix2,Iy2,lmbda,N,M,r2)
        
        oldb=beta 
        beta=np.sum(r2*y)
        if beta<=0:
            istop=9
            break; 
        beta=sqrt(beta)
        tnorm2=tnorm2+alpha**2+oldb**2+beta**2
        if (itn==1):
            if((beta/beta1)<(10*eps)):
                istop=-1
                break
        '''Apply previous rotation Qk-1 to get
        [deltak epslnk+1] = [cs  sn][dbark    0   ]
        [gbar k dbar k+1]   [sn -cs][alfak betak+1]'''

        oldeps=epsln
        delta=cs*dbar+sn*alpha
        gbar=sn*dbar-cs*alpha
        epsln=sn*beta
        dbar=-cs*beta
        root=sqrt(gbar**2+dbar**2)
        Anorm=phibar*root    #||Ar{k-1}||
        #print("Anorm",root)
        
        #Compute the next plane rotation Qk

        gamma=sqrt(gbar**2+beta**2)
        gamma=max(gamma,eps)
        cs=gbar/gamma
        sn=beta/gamma
        phi=cs*phibar
        phibar=sn*phibar

        #Update  x

        denom=1/gamma
        w1=w2
        w2=w
        w=(v-oldeps*w1-delta*w2)*denom
        x=x+phi*w
        #Go round again

        gmax=max(gmax,gamma)
        gmin=min(gmin,gamma)
        z=rhs1/gamma
        rhs1=rhs2-delta*z
        rhs2=-epsln*z
        
        #Estimate various norms.

        Anorm  = sqrt( tnorm2 );
        ynorm  = np.linalg.norm(x);
        epsa   = Anorm*eps;
        epsx   = Anorm*ynorm*eps;
        epsr   = Anorm*ynorm*rtol;
        diag   = gbar;
        if diag==0:
            diag=epsa
        qrnorm = phibar;
        rnorm  = qrnorm;
        test1  = rnorm/(Anorm*ynorm);    #  ||r|| / (||A|| ||x||)
        test2  = root / Anorm;          # ||Ar{k-1}|| / (||A|| ||r_{k-1}||)
        Acond=gmax/gmin      #Estimate  cond(A)

        #See if any of the stopping criteria are satisfied

        if istop==0:

            t1=1+test1
            t2=1+test2
            if(t2<=1):
                istop=2
            if(t1<=1):
                istop=1
            if (itn>maxiter):

                istop=6
            if(Acond>=0.1/eps):
                istop=4
            if(epsx>=beta1):
                istop=3
            if(test2<=rtol):
                istop=2
        if(test1<=rtol):
            print(test1)
            istop=1
        
        if(istop!=0):
            print('istop:',istop) # Print reason why stop
            break

    print('itn:',itn) #Print number of iterations 
    return x 


#######################################################################
'''
#Test
N=400;  lmbda=2; M=40; maxiter=300; rtol=10**-5
Ix=np.random.rand(N,M)
Iy=np.random.rand(N,M)
u0=np.random.rand(N,M)
v0=np.random.rand(N,M)
b=np.random.rand(2*N*M,1)
Ix2=Ix*Ix
Iy2=Iy*Iy
Ixy=Ix*Iy
b=b.astype(np.float32)
Ix2=Ix2.astype(np.float32)
Iy2=Iy2.astype(np.float32)
Ixy=Ixy.astype(np.float32)




####################################################
def mv(U,lmbda,Ix,Iy,N,M):
                #code.interact(local=locals())
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

L = LinearOperator((2*M*N,2*M*N), matvec=lambda U: mv(U,lmbda,Ix,Iy,N,M))
P = LinearOperator((2*M*N,2*M*N), matvec=lambda x: Mx(Ix2,Iy2,lmbda,N,M,x))
###################################################### 
t1=tm()
x,exitcode=sparse.linalg.minres(L,b,M=P)
t2=tm()
xm=minres(Ix2,Iy2,Ixy,lmbda,b,maxiter,rtol,N,M,1)
t3=tm()          
xm2=minres(Ix2,Iy2,Ixy,lmbda,b,maxiter,rtol,N,M,2)
t4=tm()  
print('my x\n',xm[:10])
print('sparse x\n',x[:10],'exit ',exitcode)
print("SPARSE:", (t2-t1),"MY SOLVER: " , (t3-t2),"MY SOLVER DOT: " , (t4-t3))


print('norm sparse:\n',np.linalg.norm(mv(x,lmbda,Ix,Iy,N,M)-b))
print('norm minres:\n',np.linalg.norm(mv(xm,lmbda,Ix,Iy,N,M)-b))
print('norm minres dot :\n',np.linalg.norm(mv(xm2,lmbda,Ix,Iy,N,M)-b))
'''

