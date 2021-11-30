import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
import math
from scipy.ndimage.filters import convolve as filter2, laplace
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import scipy.sparse as sparse
import denoise_LO as lo
import time
from cupyx.scipy.sparse.linalg import gmres
import cupyx
import cupy as cp
import mpi4py as mpi 
from mpi4py import MPI
from multiprocessing import Process
import multiprocessing
import multi_pro as mp
from scipy import signal 
import sys
from scipy.sparse.linalg import LinearOperator
import code
from numba import njit, prange
import lo
import solver
import solveur as so
import solveur_precond as sop
import precond
###########################################################
def parallel_tasks(du,i,Ix2,npixels,pp_d):
    tmp=pp_d*np.reshape(Ix2,(npixels,1),'F')
    du[i]= spdiags(tmp.T, 0, npixels, npixels)
###########################################################
@njit(parallel=True)
def laplacien(Im):
    N,M=Im.shape
    #res=np.empty((N,M)) 
    res=np.zeros((N,M)) 
    for i in prange(1,N-1):
        for j in prange(1,M-1):
            res[i,j]=Im[i-1,j]+Im[i+1,j]-4*Im[i,j]+Im[i,j-1]+Im[i,j+1]
    res[0,1:M-1]=-4*Im[0,1:M-1]+Im[0,0:M-2]+Im[0,2:M] #First row
    res[N-1,1:M-1]=-4*Im[N-1,1:M-1]+Im[N-1,0:M-2]+Im[N-1,2:M] #Last row 
    res[1:N-1,0]=-4*Im[1:N-1,0]+Im[0:N-2,0]+Im[2:N,0] #First col
    res[1:N-1,M-1]=-4*Im[1:N-1,M-1]+Im[0:N-2,M-1]+Im[2:N,M-1] #Last col
    res[0,0]=-4*res[0,0] ; res[0,M-1]=-4*res[0,M-1];res[N-1,M-1]=-4*res[N-1,M-1];res[N-1,0]=-4*res[N-1,0] # Corners 
    
    return res
###########################################################
def warp_image2(Image,XI,YI,h):
 
    ''' We add the flow estimated to the second image coordinates, remap them towards the ogriginal image  and finally  calculate the derivatives of the warped image
    h: derivative kernel
    '''

    Image=np.array(Image,np.float32)
    XI=np.array(XI,np.float32)
    YI=np.array(YI,np.float32)
    WImage=cv2.remap(Image,XI,YI,interpolation=cv2.INTER_CUBIC)
    '''Ix=filter2(Image, h)
    Iy=filter2(Image, h.T)
    
    Iy=cv2.remap(Iy,XI,YI,interpolation=cv2.INTER_CUBIC)   
    Ix=cv2.remap(Ix,XI,YI,interpolation=cv2.INTER_CUBIC)

    return [WImage,Ix,Iy]'''
    return WImage
############################################
def derivatives(Image1,Image2,u,v,h,b,xx,yy,sz):
    '''This function compute the derivatives of the second warped image
    u: horizontal displacement 
    v: vertical displacement
    Image1, Image2: images sequence 
    h: derivative kernel 
    b: weight used for averaging 
    '''

    '''Ix=np.zeros((N,M))
    Iy=np.zeros((N,M))'''
    N=sz[0]; M=sz[1]
    xx2=xx+u
    yy2=yy+v
    '''xx2=u
    yy2=v'''
    WImage=cv2.remap(Image2,np.array(xx2,np.float32),np.array(yy2,np.float32),interpolation=cv2.INTER_CUBIC)
    #WImage1=cv2.remap(Image1,np.array(xx2,np.float32),np.array(yy2,np.float32),interpolation=cv2.INTER_CUBIC)
    I2x=filter2(WImage, h)
    I2y=filter2(WImage, h.T)
    It= WImage-Image1 # Temporal deriv
    
    """Ix=filter2(WImage1, h) # spatial derivatives for the first image 
    Iy=filter2(WImage1, h.T)

    Ix  = b*I2x+(1-b)*Ix           # Averaging 
    Iy  = b*I2y+(1-b)*Iy"""


    It=np.nan_to_num(It) #Remove Nan values on the derivatives 

    out_bound= np.where((yy2 > N-1) | (yy2<0) | (xx2> M-1) | (xx2<0))

    It[out_bound]=0
    '''Ix[out_bound]=0
    Iy[out_bound]=0'''

    #return [Ix,Iy,It]
    return It
############################################################

def conv_matrix(F,sz):
    '''Construction of Laplacien Matrix
    F: spacial filter it can be [-1,1] or [[-1],[1]] 
    sz: size of the used image
    I: rows of non zeros elements
    J: columns of non zeros elements
    K: The values of the matrix M 
    (IE: M(I,J)=K)
    We distinguish horizontal and vertical filters '''
    if(F.shape==(1,2)): 
        I=np.hstack((np.arange(sz[0],(sz[0]*(sz[1]))),np.arange(sz[0],(sz[0]*(sz[1])))))                                                        
        J=np.hstack((np.arange(sz[0],(sz[0]*(sz[1]))),np.arange(sz[0],(sz[0]*(sz[1]))))) 
        K=np.zeros(int(2*(sz[0]*(sz[1]-1))))
        K[:int(sz[0]*(sz[1]-1))]=1
        J[sz[0]*(sz[1]-1):2*(sz[0]*sz[1])]=J[sz[0]*(sz[1]-1):2*sz[0]*sz[1]]-sz[0]
        K[sz[0]*(sz[1]-1):2*(sz[0]*sz[1])]=-1
    
    if(F.shape==(2,1)):
        lI=[]
        for i in range(1,sz[0]*sz[1]):
            if(i%sz[0]!=0):
                lI.append(i)
        I=np.array(lI)
        nnzl=I.shape[0]
        I=np.hstack((I,I))
        
        
        J=np.hstack((np.array(lI),np.array(lI)-1))
        K=np.ones((2*nnzl))
        K[nnzl:2*nnzl]=-1
    M=sparse.lil_matrix( ( sz[0]*sz[1],sz[0]*sz[1] ),dtype=np.float32)
    M[I,J]=K

        
    
    return M
########################################################
def deriv_charbonnier_over_x(x,sigma,a):
    ''' Derivatives of the penality over x
     '''
    #y =2*a*(sigma**2 + x**2)**(a-1);
    #y = 2 / (a)
    y=2
    return y
def deriv_quadra_over_x(x,sigma):
    ''' Derivatives of the quadratique penality  penality over x '''
    y = 2 / (sigma**2)
    return y

# These lines make the 2 previous functions work on arrays (Define a vectorized function) 
'''charbonnier_over_x=np.vectorize(deriv_charbonnier_over_x)
quadr_ov_x=np.vectorize(deriv_quadra_over_x)'''
########################################################
'''def matrix_construct_loop(S,i,M1,M2,u,du,v,dv,npixels,eps,a):
    
    #This is just a task that we use in flow_operator function 
    #Read the comments of oprical_flow function for more details 
    
    if(S[i].shape==(1,2)):
        M=M1
    elif(S[i].shape==(2,1)):
        M=M2
    u_=sparse.lil_matrix.dot(M,np.reshape((u+du),(npixels,1),'F'))
    v_=sparse.lil_matrix.dot(M,np.reshape((v+dv),(npixels,1),'F'))

    pp_su=charbonnier_over_x(u_,eps,a)
    pp_sv=charbonnier_over_x(v_,eps,a)
    dic_FU =sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_su.T, 0, npixels, npixels),M))
    dic_FV=sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(pp_sv.T, 0, npixels, npixels),M))
    return[dic_FU,dic_FV]'''
########################################################
def flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels):
    ''' Returns a linear flow operator (equation) of the form A * x = b using the a penality already choosen in .  
    The flow equation is linearized around UV with the initialization INIT
    (e.g. from a previous pyramid level).  Using Charbonnier function deriv_charbonnier_over_x function 
    u,v:  horizontal and vertical displacement 
    du,dv:  horizontal and vertical increment steps
    It,Ix,Iy: temporal and spatial derivatives 
    S: contains the spatial filters used for Computing the term related to Laplacien S=[ [[-1,1]], [[] [-1],[1] ]]
    lmbda: regularization parameter 
    eps,a: are the parameter of the penality used 
    M1,M2: Matrix of convolution used to compute laplacien term 
    ret_Aand ret_b: shared variables between some threads where we store the matrix and the second term computed 
    '''

    Itx = It*Ix
    Ity = It*Iy
    pp_d=2
    #b=sparse.lil_matrix.dot(lmbda*MM, np.vstack((np.reshape(u+du, (npixels,1) ,'F'), np.reshape(v+du, (npixels,1) ,'F') ) ) ) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) )
    b=lmbda*np.vstack(( np.reshape(laplace(u+du), (npixels,1) ,'F') ,  np.reshape(laplace(v+dv), (npixels,1) ,'F')))
    b= b-np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) )
   
    return b 
#############################################################
def flow_operator_quadr(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_b,Ix2,Iy2,Ixy,MMq,AAq,npixels):
    ''' Returns a linear flow operator (equation) of the form A * x = b using the a Quadratic penality  .  
    The flow equation is linearized around UV with the initialization INIT
    (e.g. from a previous pyramid level).  Using Charbonnier function deriv_charbonnier_over_x function 
    u,v:  horizontal and vertical displacement 
    du,dv:  horizontal and vertical increment steps
    It,Ix,Iy: temporal and spatial derivatives 
    S: contains the spatial filters used for Computing the term related to Laplacien S=[ [[-1,1]], [[] [-1],[1] ]]
    lmbda: regularization parameter 
    sigma_qua: is a parameter related to the quadratic penality  
    M1,M2: Matrix of convolution used to compute laplacien term 
    ret_Aand ret_b: shared variables between some threads where we store the matrix and the second term computed 
    PS: This function is similar to flow_operator function, the only difference is that we use a quadratic penality here and a flexible penality in the other function 
    ''' 
    #npixels=Ix.shape[1]*Ix.shape[0]

    Itx = It*Ix #It*Ix
    Ity = It*Iy #It*Iy

    pp_d=(2/sigma_qua**2)

    
    b=sparse.lil_matrix.dot(lmbda*MMq, np.vstack((np.reshape(u, (npixels,1) ,'F'), np.reshape(v, (npixels,1) ,'F') ) )) - np.vstack( (pp_d*np.reshape(Itx,(npixels,1),'F'),  pp_d*np.reshape(Ity,(npixels,1),'F') ) ) 
    ret_b[1]=b
#############################################################
#def  compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,size_median_filter,h,coef,uhat,vhat,itersLO,lambda2,lambda3,remplacement,eps,a,sigma_qua,M1,M2,Ix,Iy,Ix2,Iy2,Ixy,xx,yy,sz,MM,AA,MMq,AAq):
def  compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,size_median_filter,h,coef,uhat,vhat,itersLO,lambda2,lambda3,remplacement,eps,a,sigma_qua,M1,M2,Ix,Iy,Ix2,Iy2,Ixy,xx,yy,sz,MM,AA):
    '''COMPUTE_FLOW_BASE   Base function for computing flow field using u,v displacements as an initialization
   - Image1,Image2: Image sequence
    -max_iter: warping iteration 
    -max_linear_iter:  maximum number of linearization performed per warping
    -alpha: a parameter tused to get a weighted energy: Ec=alpha*E_quadratic+(1-alpha)E_penality 
    -S: contains the spatial filters used for Computing the term related to Laplacien S=[ [[-1,1]], [[] [-1],[1] ]]
    -size_median_filter: is the size of the used median filter or the size of the neighbors used during LO optimization(The new median formula)
    -h: spatial derivative kernel 
    -coef: factor to average the derivatives of the second warped image and the first (used on derivatives functions to get Ix,Iy and It )
    -uhat,vhat: auxiliar displacement fields 
    -itersLO: iterations for LO formulation
    -lambda2: are the parameters 
    -lmbda: regularization parameter 
    -sigma_qua: is a parameter related to the quadratic penality   
    -lambda2: weight for coupling term 
    -lambda3: weight for non local term term
    remplacement: binary variable telling us to remplace the fileds by auixilary fields or not 
    M1,M2: Matrices of convolution used to compute laplacien term 
    '''
    npixels=u.shape[0]*u.shape[1]
    #charbonnier_over_x=np.vectorize(deriv_charbonnier_over_x)
    Lambdas=np.logspace(math.log(1e-4), math.log(lambda2),max_iter)
    lambda2_tmp=Lambdas[0] 
    #epsilon=[0.001,0.001,0.001,0.0001,0.0001,0.0001,0.0001,0.00001,0.00001,0.00001]
    epsilon=(10**-6)*np.ones((10))
    residu=[]
    for i in range(max_iter):
        du=np.zeros((u.shape)); dv=np.zeros((v.shape))
       
        #[Ix,Iy,It]=derivatives(Image1,Image2,u,v,h,coef)
        It=derivatives(Image1,Image2,u,v,h,coef,xx,yy,sz)
        #[Ix,Iy,It]=derivatives(Image1,Image2,u,v,h,coef,xx,yy,sz)
        residu.append(np.linalg.norm(It))
        #print('iter',i,'shapes',Ix.shape)
        for j in range(max_linear_iter):
            if (alpha==1):
                '''manager = multiprocessing.Manager()
                ret_b=manager.dict()
                ret_b[0]=0
                p1 = multiprocessing.Process(target=flow_operator_quadr, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_b,Ix2,Iy2,Ixy,MMq,AAq,npixels))
                p1.start()
                p1.join()

                #A=ret_A[1]

                A=AAq
                b=ret_b[1]'''
                ret_b=[]
                A=AA
                b=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels)
            elif(alpha>0 and alpha != 1): 
                
                ''' manager = multiprocessing.Manager()
                #ret_A = manager.dict()
                #managerb=multiprocessing.Manager()
                ret_b=manager.dict()
                p1 = multiprocessing.Process(target=flow_operator_quadr, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,sigma_qua,M1,M2,ret_b,Ix2,Iy2,Ixy,MMq,AAq,npixels))
                p2 = multiprocessing.Process(target=flow_operator, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels))
                p2.start()
                p1.start()

                p1.join()
                p2.join()
                
                #A=alpha*A+(1-alpha)*A1
                #b=alpha*b+(1-alpha)*b1
                A=alpha*ret_A[1]+(1-alpha)*ret_A[0]
                #b=alpha*ret_b[1]+(1-alpha)*ret_b[0]'''
                #b=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels)
                b=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels)
                #A=alpha*AAq+(1-alpha)*AA
                #A=AA
            elif(alpha==0):

                '''manager = multiprocessing.Manager()
                ret_b=manager.dict()
                p1 = multiprocessing.Process(target=flow_operator, args=(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels))
                p1.start()
                p1.join()

                A=AA
                b=ret_b[0]'''
                ret_b=[]
                A=AA
                b=flow_operator(u,v, du,dv, It, Ix, Iy,S,lmbda,eps,a,M1,M2,ret_b,Ix2,Iy2,Ixy,MM,AA,npixels)
            
            tmp0=np.reshape( np.hstack((u-uhat,v-vhat)) , (1,2*u.shape[0]*u.shape[1]) ,'F')
            #tmp1=deriv_charbonnier_over_x(tmp0,eps,a)
            #tmp1=charbonnier_over_x(tmp0,eps,a)
            tmp1=(2/a**2)*np.ones_like(tmp0)
            
            #tmpA=spdiags(tmp1,0,A.shape[0],A.shape[1])
            #A= A + lambda2_tmp*tmpA
            b=b - lambda2_tmp*tmp1.T*tmp0.T
            #b=np.ravel(b)

            #print("temps pour construire la matrice:",(t2-t1))
            #x=scipy.sparse.linalg.spsolve(A,b)  #Direct solvers 
            #y=scipy.sparse.linalg.gmres(A,b)  #Gmres  solver 
            #y=scipy.sparse.linalg.bicg(A,b) #BICg Solver 
            #y=scipy.sparse.linalg.lgmres(A,b) # LGMRES 
            #diag=1/A.diagonal()
            '''if(i==0):
                x=None'''
            N,M=Image1.shape
            #U=np.vstack((np.reshape(u,(N*M,1),order="F"),np.reshape(v,(N*M,1),order="F")) )

            '''U=np.vstack((np.reshape(u,(N*M),order='F' ) ,np.reshape(v,(N*M),order='F' )  ) )
            L = LinearOperator((2*M*N,2*M*N), matvec=lambda U: lo.mv(U,lmbda,Ix,Iy,N,M)) 
            P = LinearOperator((2*M*N,2*M*N), matvec=lambda x: precond.Mx(Ix2,Iy2,lmbda,N,M,x))  '''      
            #x,exitcode=scipy.sparse.linalg.minres(L,b,M=P)
            #x=so.minres(Ix2.astype(np.float32),Iy2.astype(np.float32),Ixy.astype(np.float32),lmbda,b.astype(np.float32),300,10**-5,N,M,2)
            x=sop.minres(Ix2.astype(np.float32),Iy2.astype(np.float32),Ixy.astype(np.float32),float(lmbda),b.astype(np.float32),300,10**-5,N,M,2)
            #x,exitcode=scipy.sparse.linalg.cgs(L, b)
            #P=spdiags(A.diagonal(), 0, A.shape[0],A.shape[1] ) #Precond de Jacobi
            #x,exitcode=scipy.sparse.linalg.minres(A,b)
            '''P1=scipy.sparse.diags(np.ravel(Ix*Ix))
            P2=scipy.sparse.diags(np.ravel(Iy*Iy))'''
            '''Ix2=Ix*Ix; Ix2[Ix2==0]=0.01; 
            Iy2=Iy*Iy; Iy2[Iy2==0]=0.01; 
            P=scipy.sparse.diags(np.hstack((np.ravel(Ix2,'F'),np.ravel(Iy2,'F'))))'''
            
            '''[u1,v1]=solver.solver(Ix,Iy,It,lmbda,u,v,500,0.0001)
            x=np.empty((2*M*N,1))
            x[:M*N]=np.reshape(u1,(M*N,1),'F')
            x[M*N:]=np.reshape(v1,(M*N,1),'F')'''
            '''x,exitcode=scipy.sparse.linalg.minres(A,b,M=P,tol=epsilon[i]) #entre 1 et 2   650 s  #Minres Solver using P as preconditioner''' 
            
            
            ''' bt=np.empty_like(b)
            #print("size of x", b.shape,np.ravel(1/Ix,order='F').shape,(np.reshape(np.ravel(1/Ix,order='F'),(N*M,1))*b[:N*M]).shape)
            
            #bt[:N*M]=np.ravel(1/Ix,order='F')*b[:N*M]
            bt[:N*M]=np.reshape(1/Ix,(N*M,1),order='F')*b[:N*M]
            bt[N*M:2*N*M]=np.reshape(1/Iy,(N*M,1),order='F')*b[N*M:2*N*M]'''
            
            #x,exitcode=scipy.sparse.linalg.minres(L,b)
            #x=np.empty_like(y)
            '''x[:N*M]=np.ravel(1/Ix,order='F')*x[:N*M]
            x[N*M:2*N*M]=np.ravel(1/Iy,order='F')*x[N*M:2*N*M]'''
            '''print("x\n",x[990:1000])'''
            '''Ix=np.reshape(Ix,(N,M),order='F')
            Iy=np.reshape(Iy,(N,M),order='F')'''
            



            #x=y[0]
            x[x>1]=1
            x[x<-1]=-1
            du=np.reshape(x[0:npixels], (u.shape[0],u.shape[1]),'F' )
            dv=np.reshape(x[npixels:2*npixels], (u.shape[0],u.shape[1]),'F' )

            
        print('\t \t \tWarping step',i)
        u = u + du
        v = v + dv
        '''uhat=lo.denoise_LO (u, size_median_filter, lambda2_tmp/lambda3, itersLO) # Denoising LO new formula of optimization  
        vhat=lo.denoise_LO (v, size_median_filter, lambda2_tmp/lambda3, itersLO)'''
        #[uhat,vhat]=lo.denoise_LO (u,v, size_median_filter, lambda2_tmp/lambda3, itersLO)
        uhat=median_filter(u,size=size_median_filter) # Denoising using a normal median filter 
        vhat=median_filter(v,size=size_median_filter)
        #print("Temps optimisation mediane",(t2-t1))
        if remplacement==True:
            u=uhat
            v=vhat
        if i!=max_iter-1:

            lambda2_tmp=Lambdas[i+1]
    print(residu)
    return [u,v,uhat,vhat]
