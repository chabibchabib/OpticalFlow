import numpy as np
import scipy.sparse as sparse
from scipy.sparse import spdiags
from scipy.ndimage.filters import convolve as filter2
from scipy.ndimage.filters import convolve1d as filter


def Matrix1(npixels,S,M1,M2,duu,dvv,dduv,lmbda,ret_A,ret_M):
    FU=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    FV=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    for i in range(len(S)):
        if(S[i].shape==(1,2)):
            M=M1
        elif(S[i].shape==(2,1)):
            M=M2
        FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),M))
    #FU        = sparse.lil_matrix.dot((M1+M2).T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),(M1+M2)))
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FV ) )  )) 
    #MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FU ) )  )) 
 
    #MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FU ) )  ))
    del(FU)
    #del(FV); del(M)
    pp_d=2
    AA = pp_d*sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM
    ret_A[0]=AA
    ret_M[0]=MM
    #return [AA,MM]
#############################################################################################################
def Matrix2(npixels,S,M1,M2,lmbda,duu,dvv,dduv,sigma_qua,ret_A,ret_M):
    FUq=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    FVq=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    for i in range(len(S)):
        if(S[i].shape==(1,2)):
            Mq=M1
        elif(S[i].shape==(2,1)):
            Mq=M2
        FUq      = FUq+ sparse.lil_matrix.dot(Mq.T,sparse.lil_matrix.dot(spdiags((2/sigma_qua**2)*np.ones((npixels)), 0, npixels, npixels),Mq))
        FVq        = FVq+ sparse.lil_matrix.dot(Mq.T,sparse.lil_matrix.dot(spdiags((2/sigma_qua**2)*np.ones((npixels)), 0, npixels, npixels),Mq))
    MMq = sparse.vstack( (sparse.hstack ( ( -FUq, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FVq ) )  ))  
    del(FUq); del(FVq); del(Mq)
    pp_d=2/sigma_qua**2
    AAq = pp_d*sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MMq
    ret_A[1]=AAq
    ret_M[1]=MMq
    #return[AAq,MMq]
#####################################
def Matrix3(npixels,S,M1,M2,duu,dvv,dduv,lmbda):
    FU=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    #FV=sparse.lil_matrix((npixels,npixels),dtype=np.float32)
    #FU=np.empty((npixels,npixels),dtype=np.float32)
    for i in range(len(S)):
        if(S[i].shape==(1,2)):
            M=M1
        elif(S[i].shape==(2,1)):
            M=M2
        '''FU        = FU+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),M))
        FV        = FV+ sparse.lil_matrix.dot(M.T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),M))'''
        FU=FU+sparse.lil_matrix.dot(M.T,M)
    FU=2*FU
        #filter( spdiags((2*np.ones((npixels )) ), 0, npixels, npixels),[[-1,1]]  )
        #FU=spdiags((2*np.ones((npixels )) ), 0, npixels, npixels)
        #print("shape of si",S[i].shape)

    #FU=sparse.lil_matrix.dot(M1.T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),M1))+sparse.lil_matrix.dot(M2.T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),M2))
    #MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FV ) )  ))
    #print(MM)
    '''FU        = sparse.lil_matrix.dot((M1).T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),(M1)))
    FU        = FU+sparse.lil_matrix.dot((M2).T,sparse.lil_matrix.dot(spdiags(2*np.ones((npixels)), 0, npixels, npixels),(M2)))'''
    MM = sparse.vstack( (sparse.hstack ( ( -FU, sparse.lil_matrix((npixels,npixels)) ) )  ,  sparse.hstack( ( sparse.lil_matrix((npixels,npixels)) , -FU ) )  ))
    del(FU); 
    #del(FV); del(M)
    pp_d=2
    AA = pp_d*sparse.vstack( (sparse.hstack ( ( duu, dduv ) )  ,  sparse.hstack( ( dduv , dvv ) )  )) - lmbda*MM

    return [AA,MM]