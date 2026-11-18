from functools import lru_cache
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY/'analysis'))
from physics import T2_GRID,SIGMA,WATER_DENSITY,GRAVITY,cylinder_modes,bin_modes

@lru_cache(maxsize=32)
def cumulative_lookup(alpha,n,rho,points=1025,quadrature=16):
    se=np.linspace(0,1,points);nodes,weights=leggauss(quadrature)
    v=se[:-1,None]+(nodes[None,:]+1)/(2*(points-1))
    m=1-1/n;c=2*SIGMA/(WATER_DENSITY*GRAVITY)
    radius=alpha*c/np.expm1(-np.log(v)/m)**(1/n)
    rate,amp,beta=cylinder_modes(radius,rho,16)
    water=weights[None,:,None]*amp/(2*(points-1))
    intervals=bin_modes(rate,water,T2_GRID)
    distribution=np.vstack([np.zeros(len(T2_GRID)),np.cumsum(intervals,axis=0)])
    assert np.min(np.diff(distribution,axis=0))>=-1e-15
    return se,distribution,dict(points=points,quadrature_per_interval=quadrature,
        maximum_water_error=float(abs(distribution.sum(axis=1)-se).max()))
