import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cholesky,solve_triangular
from scipy.optimize import nnls
from physics import STUDY,TIMES

INVERSE_T2=np.geomspace(.003,2.2,16)
INVERSE_EDGES=np.r_[np.linspace(0,1,21),2.,5.,10.]


def design_matrix():
    obs=np.load(STUDY/'data/observation_kernel_full_100.npz')
    z=obs['z_m']; groups=[]
    for lo,hi in zip(INVERSE_EDGES[:-1],INVERSE_EDGES[1:]):
        groups.append(obs['kernel_nV'][:,(z>=lo)&(z<hi),:].sum(axis=1))
    kernel=np.stack(groups,axis=1)
    kernel=interp1d(np.log(obs['t2_s']),kernel,axis=-1)(np.log(INVERSE_T2))
    a=(kernel[:,None,:,:]*np.exp(-TIMES[None,:,None,None]/INVERSE_T2)).reshape(24*32,-1)
    a=np.vstack([a.real,a.imag])
    nz,nt=len(INVERSE_EDGES)-1,len(INVERSE_T2)
    depth=np.zeros((19,nz))
    for i in range(19): depth[i,i]=-1;depth[i,i+1]=1
    spectral=np.diff(np.eye(nt),axis=0)
    q=.01*np.eye(nz*nt)+np.kron(depth.T@depth,np.eye(nt))+.3*np.kron(np.eye(nz),spectral.T@spectral)
    return a,q


class Inverter:
    def __init__(self,strength):
        self.a,self.penalty=design_matrix()
        self.scale=np.linalg.norm(self.a)/np.sqrt(self.a.shape[1])
        self.strength=strength
        self.h=self.a.T@self.a+(strength*self.scale)**2*self.penalty
        self.l=cholesky(self.h,lower=True)
        self.r=self.l.T.copy(order='F')

    def fit(self,signals):
        original=signals.shape[:-2]
        y=signals.reshape(-1,24*32).T
        y=np.vstack([y.real,y.imag])
        rhs=solve_triangular(self.l,self.a.T@y,lower=True)
        result=np.empty((y.shape[1],self.a.shape[1]))
        for i in range(y.shape[1]):
            result[i]=nnls(self.r,rhs[:,i],maxiter=10000)[0]
        modeled=self.a@result.T
        rms=np.sqrt(np.mean((modeled-y)**2,axis=0))
        f=result.reshape(*original,len(INVERSE_EDGES)-1,len(INVERSE_T2))
        return f,rms.reshape(original)


def profile_features(f):
    water=f[...,:20,:].sum(axis=-1)
    moment=(f[...,:20,:]*np.log(INVERSE_T2)).sum(axis=-1)
    return water,moment
