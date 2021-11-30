import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter,gaussian_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import correlate
from skimage.transform import resize
from skimage.util import dtype
import flow_operator as fo
import rescale_img as ri
import energies as en
import matplotlib.pyplot as plt
import numpy.matlib
import time 
import mpi4py as mpi 
from mpi4py import MPI
import matrices as mtx
from multiprocessing import Process
import multiprocessing
from scipy.sparse.linalg import LinearOperator

##############################################################
def compute_image_pyram(Im1,Im2,ratio,N_levels,gaussian_sigma):
    ''' This function creates a the images we will use in every level
    Parameters: 

        -Im1,Im2: Images 
        -ratio: downsampling factor 
        -N_levels: Number of levels 
        -Gaussian_sigma: standard deviation of the gaussian filter kernel (Not used in mechanics)
    Reurns:
        P1,P2 pyramids of images
        '''

    
    P1=[]
    P2=[]
    tmp1=ri.scale_image(Im1,0,255)
    tmp2=ri.scale_image(Im2,0,255)
    P1.append(tmp1)
    P2.append(tmp2)

    for lev in range(1,N_levels):

        sz=np.round(np.array(tmp1.shape,dtype=np.float32)*ratio)

        tmp1=resize(tmp1,(sz[0],sz[1]),anti_aliasing=False,mode='symmetric')
        tmp2=resize(tmp2,(sz[0],sz[1]),anti_aliasing=False,mode='symmetric')

        P1.append(tmp1)
        P2.append(tmp2)
    return [P1,P2]


def resample_flow_unequal(u,v,sz,ordre_inter):
    '''
    Resizes the flow field (u,v)
    Parameters:
        -u,v: the displacements 
        -sz: the shape
        - order_inter: the order of the interpolation used in the function skimage.transform.resize
    Returns: 
        u,v the new resized displacements 

    '''
    osz=u.shape
    ratioU=sz[0]/osz[0]
    ratioV=sz[1]/osz[1]
    u=resize(u,sz,order=ordre_inter)*ratioU
    v=resize(v,sz,order=ordre_inter)*ratioV
    return u,v
############################################################
def compute_flow(Im1,Im2,u,v,iter_gnc,gnc_pyram_levels,gnc_factor,gnc_spacing, pyram_levels,factor,spacing,ordre_inter, alpha,lmbda, size_median_filter,h,coef,S,max_linear_iter,max_iter,lambda2,lambda3,eps,a,sigma_qua):
    '''#param1=1/8; param2=100; param3=0.95; param4=False;
    #param1=1/10; param2=100; param3=0.5; param4=False 
    Im1,Imm1=ri.decompo_texture(Im1, param1, param2, param3, param4)
    Im2,Imm1=ri.decompo_texture(Im2, param1, param2, param3, param4)'''

    P1,P2=compute_image_pyram(Im1,Im2,1/factor,pyram_levels,math.sqrt(spacing)/math.sqrt(2))
    #P1_gnc,P2_gnc=compute_image_pyram(Im1,Im2,1/gnc_factor,gnc_pyram_levels,math.sqrt(gnc_spacing)/math.sqrt(2))
    ########################
    MM1=[]; MM2=[];
    IIx=[]; IIy=[]; IIxx=[]; IIyy=[]; IIxy=[]

    
    for it_l in range(int(pyram_levels)):
        sz2=np.round(np.array(Im1.shape,dtype=np.float32)*(1/factor**(it_l)))
        sz2=np.array(sz2,dtype=np.int)
        IIx.insert(it_l,filter2(P1[it_l], h)) # spatial derivatives for the first image 
        IIy.insert(it_l,filter2(P1[it_l], h.T))
        IIxx.insert(it_l,IIx[it_l]*IIx[it_l]) # Ix^2 
        IIyy.insert(it_l,IIy[it_l]*IIy[it_l]) # Iy^2
        IIxy.insert(it_l,IIx[it_l]*IIy[it_l]) # Ix*Iy
############################### pyram

    uhat=u; vhat=v; remplacement=True;
    itersLO=1
    for i in range(iter_gnc):
        print("Gnc itération",i)
        if i==(iter_gnc-1):
            remplacement=False

        else:
            remplacement=True

        if i==0:
            py_lev=pyram_levels
        else:
            py_lev=gnc_pyram_levels
            #py_lev=pyram_levels
        print('\t Pyram levels',py_lev)

        for lev in range(py_lev-1,-1,-1):

            sz2=np.round(np.array(Im1.shape,dtype=np.float32)*(1/factor**(lev)))
            sz2=np.array(sz2,dtype=np.int)
            
            print("\t \t Level Number",lev)
            if i==0:
                Image1=P1[lev]; Image2=P2[lev]
                sz= Image1.shape

            u,v=resample_flow_unequal(u,v,sz,ordre_inter)
            '''Ix=filter2(Image1, h)
            Iy=filter2(Image1, h.T)
            Ix2=Ix*Ix
            Iy2=Iy*Iy
            Ixy=Ix*Iy'''
            
            #U=np.ravel(np.hstack(u,v),order='F')
            if(py_lev==pyram_levels):
                #M1=MM1[lev]; M2=MM2[lev]
                Ix=IIx[lev];
                Iy=IIy[lev]
                Ix2=IIxx[lev]
                print("type Ix2",type(Ix2))

                Iy2=IIyy[lev]
                Ixy=IIxy[lev]
            '''elif(py_lev==gnc_pyram_levels):
                M1=MM1p[lev]; M2=MM2p[lev]
                Ix=IIxp[lev];
                Iy=IIyp[lev]
                Ix2=IIxxp[lev]
                Iy2=IIyyp[lev]
                Ixy=IIxyp[lev]'''
            N,M=sz
            #plt2.spy(Ix)
            yy=np.linspace(0,N-1,N)
            xx=np.linspace(0,M-1,M)
            xx,yy=np.meshgrid(xx,yy)
            #print("x shape:",xx.shape)
            uhat,vhat=resample_flow_unequal(uhat,vhat,sz,ordre_inter)
            #print('shapes,',u.shape,uhat.shape)

            if (lev==0) and (i==iter_gnc) :    #&& this.noMFlastlevel
                median_filter_size =0
            else: 
                median_filter_size=size_median_filter
#####################################################################################################################################
            npixels=M*N

            '''tmp=np.reshape(Ix2,(npixels),'F')
            tmp=np.nan_to_num(tmp)

            duu = spdiags(tmp, 0, npixels, npixels)
    
            tmp = np.reshape(Iy2,(npixels),'F')
            tmp=np.nan_to_num(tmp)

            dvv = spdiags(tmp, 0, npixels, npixels)
    
            tmp =np.reshape(Ixy,(npixels),'F')
            tmp=np.nan_to_num(tmp)
    
            dduv = spdiags(tmp, 0, npixels, npixels)'''
            '''def mv(U):
                u=U[:N*M]
                v=U[N*M:]
                Ix1=np.ravel(Ix,order='F')

                Iy1=np.ravel(Iy,order='F')
                Ix1[Ix1==0]=0.00001
                Iy1[Iy1==0]=0.00001
                Ix1=1/Ix1
                Iy1=1/Iy1
                #Ix1=np.nan_to_num(Ix1); Iy1=np.nan_to_num(Iy1)
                u1=u+v
                v1=u1
                u=u*Ix1
                u=scipy.ndimage.laplace(u)
                u=u*Ix1
                u1=u1+lmbda*u
                v=v*Iy1
                v=scipy.ndimage.laplace(v)
                v=v*Iy1
                v1=v1+lmbda*v
                return np.array(([u1,v1]))
            AA=LinearOperator((2*npixels,2*npixels),matvec=mv)'''
###########################################################################
            '''manager = multiprocessing.Manager()
            ret_A = manager.dict()
            #managerb=multiprocessing.Manager()
            ret_M=manager.dict()
            p1 = multiprocessing.Process(target=mtx.Matrix1, args=(npixels,S,M1,M2,duu,dvv,dduv,lmbda,ret_A,ret_M))
            p2 = multiprocessing.Process(target=mtx.Matrix2, args=(npixels,S,M1,M2,lmbda,duu,dvv,dduv,sigma_qua,ret_A,ret_M))
            p2.start()
            p1.start()

            p1.join()
            p2.join()'''
            '''AA=ret_A[0]; AAq=ret_A[1]
            MM=ret_M[0]; MMq=ret_M[1]'''
            #MM=fo.Laplacien_mat(sz)
            #print(AA.shape,AA.getnnz())
###########################################################################
            '''[AA,MM]=mtx.Matrix1(npixels,S,M1,M2,duu,dvv,dduv,lmbda)
            [AAq,MMq]=mtx.Matrix2(npixels,S,M1,M2,lmbda,duu,dvv,dduv,sigma_qua)'''
            #[AA,MM]=mtx.Matrix3(npixels,S,M1,M2,duu,dvv,dduv,lmbda)

####################################################################################################################################################
            MM=[];M1=[]; M2=[]; 
            AA=[]
            u,v,uhat,vhat=fo.compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,median_filter_size,h,coef,uhat,vhat,itersLO,lambda2,lambda3,remplacement,eps,a,sigma_qua,M1,M2,Ix,Iy,Ix2,Iy2,Ixy,xx,yy,sz,MM,AA)
            #u,v,uhat,vhat=fo.compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,median_filter_size,h,coef,uhat,vhat,itersLO,lambda2,lambda3,remplacement,eps,a,sigma_qua,M1,M2,Ix,Iy,Ix2,Iy2,Ixy,xx,yy,sz,ret_M[0],ret_A[0],ret_M[1],ret_A[1])


            if iter_gnc > 0:

                new_alpha  = 1 - (i+1)/ (iter_gnc)
                alpha = min(alpha, new_alpha)
                alpha = max(0,alpha)
            #print('iteration gnc',i)

    u=uhat
    v=vhat
    return u,v
####################################################
Im1=cv2.imread('Im11.png',0)
Im2=cv2.imread('Im22.png',0)
'''u=np.loadtxt('u0s.txt')
v=np.loadtxt('v0s.txt')'''
#u=np.zeros((Im1.shape)); v=np.zeros((Im1.shape))

'''Im1=cv2.imread('Img00.png',0)
Im2=cv2.imread('Img11.png',0)
#u=np.loadtxt('u0.txt')
#v=np.loadtxt('v0.txt')
u=np.loadtxt('u0_d0.txt')
v=np.loadtxt('v0_d0.txt')'''

Im1=np.array(Im1,dtype=np.float32)
Im2=np.array(Im2,dtype=np.float32)
u=np.zeros((Im1.shape)); v=np.zeros((Im1.shape))

#GNC params
iter_gnc=1
gnc_pyram_levels=1
'''gnc_factor=1.25
gnc_spacing=1.25'''
gnc_factor=2
gnc_spacing=2
#Pyram params 
pyram_levels=3
factor=2
spacing=2
ordre_inter=3
alpha=1
#alpha=0
size_median_filter=5
h=np.array([[-1 ,8, 0 ,-8 ,1 ]]); h=h/12
#h=np.array([[-1 ,1 ]]); h=h/2
coef=0.5
S=[]
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))
#Algo params
a=1
#eps=0.001
eps=0
max_linear_iter=1
max_iter=10
lmbda=10**4
lambda2=0.001
#lambda3=5
lambda3=1
#sigma_qua=50
sigma_qua=1

#pyram_levels=ri.compute_auto_pyramd_levels(Im1,spacing) #Computing the number of levels dinamically, in  the finest level we get images of 20 to 30 pixels 
t1=time.time()
u,v=compute_flow(Im1,Im2,u,v,iter_gnc,gnc_pyram_levels,gnc_factor,gnc_spacing, pyram_levels,factor,spacing,ordre_inter, 
alpha,lmbda, size_median_filter,h,coef,S,max_linear_iter,max_iter,lambda2,lambda3,eps,a,sigma_qua)
t2=time.time()
print('Elapsed time:',(t2-t1),'(s) What means : ', (t2-t1)/60, '(min)')
'''print('Im1')
print(Im1[0:5,0:5])

print('Im2')
print(Im2[0:5,0:5])

print('u')
print(u[0:5,0:5])

print('v')
print(v[0:5,0:5])
N,M=Im1.shape
y=np.linspace(0,N-1,N)
x=np.linspace(0,M-1,M)
x,y=np.meshgrid(x,y)
x2=x+u; y2=y+v
x2=np.array(x2,dtype=np.float32)
y2=np.array(y2,dtype=np.float32)
I=cv2.remap(np.array(Im1,dtype=np.float32),x2,y2,cv2.INTER_LINEAR)
norme=np.linalg.norm(I-Im2)/np.linalg.norm(Im2)
print(norme)
cv2.imwrite('I_3.png',I)
print(I[0:5,0:5])'''
step=20
Exy,Exx=np.gradient(u)
np.save('u_lin2.npy',u) 
np.save('v_lin2.npy',v)                                                                                                                                                                                                        
print("Energie Image: %E"%(en.energie_image(Im1,Im2,u,v)))
print("Energie Grad déplacement: %E"%(en.energie_grad_dep(u,v,lmbda)))    
plt.figure(); plt.imshow(Exx);plt.clim(-0.1,0.1);plt.colorbar();
#plt.imshow(Im1) 
plt.title("mf.size=5 lmbda=1e+03,Lambda2=1e-02 Lambda3=2.5"); plt.show(block=False) ;
plt.savefig('nc6_median')
                                  
