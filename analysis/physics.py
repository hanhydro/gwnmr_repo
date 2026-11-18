from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss

STUDY = Path(__file__).resolve().parents[1]
SIGMA = .0728
WATER_DENSITY = 999.97
GRAVITY = 9.81
DIFFUSIVITY = 2.0e-9
BULK_T2 = 2.2
RHO = (6.e-6, 8.e-6)
T2_GRID = np.geomspace(1.e-4, BULK_T2, 96)
TIMES = np.geomspace(.005, .5, 32)


def capillary_example(pressure, times, radius=1.e-5, rho=1.e-5,
                      angles=None, bulk=BULK_T2):
    pressure, times = np.asarray(pressure), np.asarray(times)
    area = np.pi * radius**2
    if angles is None:
        full = pressure <= 2 * SIGMA / radius
        return full[:, None] * np.exp(-times[None, :] * (1/bulk + 2*rho/radius))
    a = np.deg2rad(angles)
    sides = np.sqrt(2*area*np.sin(a)/(np.sin(np.roll(a,1))*np.sin(np.roll(a,2))))
    perimeter = sides.sum()
    factors = 1/np.tan(a/2) - (np.pi-a)/2
    angularity = factors.sum()
    critical_radius = perimeter/(2*(angularity+np.pi)+2*np.sqrt(np.pi*(angularity+np.pi)))
    full = pressure <= SIGMA/critical_radius
    meniscus = SIGMA/pressure
    corner_area = meniscus[:, None]**2*factors[None, :]
    wetted_perimeter = 2*meniscus[:, None]/np.tan(a[None,:]/2)
    rate = 1/bulk + rho*wetted_perimeter/corner_area
    signal = np.einsum('pc,pct->pt',corner_area/area,np.exp(-rate[:,:,None]*times))
    signal[full] = np.exp(-times*(1/bulk+rho*perimeter/area))
    return signal


@lru_cache(maxsize=4)
def cylinder_table(modes=16):
    beta = np.geomspace(1e-8,1e4,801)
    roots = np.zeros((len(beta),modes))
    z0,z1=jn_zeros(0,modes),jn_zeros(1,modes)
    for i,b in enumerate(beta):
        for k in range(modes):
            lower=1e-12 if k==0 else z1[k-1]+1e-12
            roots[i,k]=brentq(lambda x:x*j1(x)-b*j0(x),lower,z0[k]-1e-12,xtol=1e-13)
    return beta,roots


def cylinder_modes(radius,rho,modes=16):
    radius=np.asarray(radius)
    beta=rho*radius/DIFFUSIVITY
    btab,xtab=cylinder_table(modes)
    if beta.max()>btab[-1]:
        raise ValueError('Cylinder interpolation range exceeded')
    x=np.stack([np.interp(np.log(beta),np.log(btab),np.log(xtab[:,k]))
                for k in range(modes)],axis=-1)
    x=np.exp(x)
    tiny=beta<btab[0]
    x[...,0]=np.where(tiny,np.sqrt(2*beta)*(1-beta/8),x[...,0])
    b=beta[...,None]
    amplitude=4*b*b/(x*x*(x*x+b*b))
    rate=1/BULK_T2+DIFFUSIVITY*x*x/radius[...,None]**2
    return rate,amplitude,beta


def bin_modes(rates,weights,tgrid=T2_GRID):
    t=1/rates
    location=np.interp(np.log(t),np.log(tgrid),np.arange(len(tgrid)))
    lo=np.floor(location).astype(int); hi=np.minimum(lo+1,len(tgrid)-1)
    frac=location-lo
    rows=np.broadcast_to(np.arange(rates.shape[0])[:,None,None],rates.shape).ravel()
    out=np.zeros((rates.shape[0],len(tgrid)))
    np.add.at(out,(rows,lo.ravel()),(weights*(1-frac)).ravel())
    np.add.at(out,(rows,hi.ravel()),(weights*frac).ravel())
    return out


def pore_lookup(alpha,n,rho,tgrid=T2_GRID,quadrature=48,modes=16,points=513):
    se=np.linspace(0,1,points)
    nodes,weights=leggauss(quadrature)
    v=se[1:,None]*(nodes[None,:]+1)/2
    m=1-1/n
    c=2*SIGMA/(WATER_DENSITY*GRAVITY)
    radius=alpha*c/np.expm1(-np.log(v)/m)**(1/n)
    rate,amp,beta=cylinder_modes(radius,rho,modes)
    water=se[1:,None,None]*weights[None,:,None]/2*amp
    distribution=np.vstack([np.zeros(len(tgrid)),bin_modes(rate,water,tgrid)])
    omitted=np.max(1-amp.sum(axis=-1))
    fd_water=(se[1:,None]*weights[None,:]/2*(beta>.1)).sum(axis=1)
    return se,distribution,dict(max_omitted_mode_amplitude=float(omitted),
                              max_beta=float(beta.max()),
                              saturated_fraction_radius_below_2nm=float((weights/2*(radius[-1]<2e-9)).sum()),
                              saturated_fraction_beta_above_0p1=float(fd_water[-1]))


def bloch_transverse(omega,rate,duration=.005,offset=0.):
    omega=np.asarray(omega); rate=np.asarray(rate)
    total=np.sqrt(omega**2+offset**2)
    denom=rate**2+total**2
    e=np.exp(-rate*duration)
    y=omega/denom*(rate+e*(total*np.sin(total*duration)-rate*np.cos(total*duration)))
    if offset==0:
        return y
    sinc=np.sinc(total*duration/np.pi)*duration
    x=omega*offset/denom*(1-e*np.cos(total*duration)-rate*e*sinc)
    return y+1j*x
