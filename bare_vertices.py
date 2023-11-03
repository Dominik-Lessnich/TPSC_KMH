import numpy as np

def Kanemele_gauge_Uk(k1, k2, nk):
    """
    calculates gauge matrices U(k), also returns adjoint
    """
    zrs = np.zeros(nk)
    ones = np.ones(nk)
    phi = np.exp(1j * 2 * np.pi * (k1 * 1./3 + k2 * 1./3)).reshape(nk) 

    phis = np.moveaxis(np.concatenate((ones,zrs,zrs,phi),axis=0).reshape(2,2,nk),-1,0)
    Uk = np.kron(phis,np.eye(2))

    Ukd = np.conjugate(np.transpose(Uk,(0,2,1)))

    return Uk, Ukd

def Kanemele_dHk(k1, k2, nk, t=1., lmb=0., lmbr=0.):
    """
    calculates k derivatives of H(k), no gauge included here
    """
    zrs = np.zeros(nk)
    sigz = np.array([[1., 0.], [0., - 1.]])

    dkxhk = -1j * t * (1.* np.exp(1j * 2.0 * np.pi * k1) + 0.5 * np.exp(1j * 2.0 * np.pi * k2)).reshape(nk)
    dkyhk = -1j * t * (0.* np.exp(1j * 2.0 * np.pi * k1) + np.sqrt(3)/2. * np.exp(1j * 2.0 * np.pi * k2)).reshape(nk)

    dkxgak = 2. * lmb * (1. * np.cos(2. * np.pi * k1) - 0.5 * np.cos(2. * np.pi * k2) + (-1. + 0.5) * np.cos(2. * np.pi * (-k1 + k2))).reshape(nk)
    dkygak = 2. * lmb * (0. * np.cos(2. * np.pi * k1) - np.sqrt(3)/2. * np.cos(2. * np.pi * k2) + (-0. + np.sqrt(3)/2.) * np.cos(2. * np.pi * (-k1 + k2))).reshape(nk)

    graphenex = np.moveaxis(np.concatenate((zrs,dkxhk,np.conjugate(dkxhk),zrs),axis=0).reshape((2,2,nk)),-1,0)
    graphenex_sf = np.kron(graphenex,np.eye(2))

    lmbx_term = np.moveaxis(np.concatenate((dkxgak,zrs,zrs,-1.*dkxgak),axis=0).reshape((2,2,nk)),-1,0)
    socx = np.kron(lmbx_term,sigz)

    dkxH0k = graphenex_sf + socx

    grapheney = np.moveaxis(np.concatenate((zrs,dkyhk,np.conjugate(dkyhk),zrs),axis=0).reshape((2,2,nk)),-1,0)
    grapheney_sf = np.kron(grapheney,np.eye(2))

    lmby_term = np.moveaxis(np.concatenate((dkygak,zrs,zrs,-1.*dkygak),axis=0).reshape((2,2,nk)),-1,0)
    socy = np.kron(lmby_term,sigz)

    dkyH0k = grapheney_sf + socy

    return dkxH0k, dkyH0k
    

def Kanemele_gauge_dUk(k1, k2, nk, num_wn):
    """
    calculates k derivatives of gauge matrices U(k), also returns their adjoints
    """
    zrs = np.zeros(nk)

    dkxphik = 1j * 1./3 * (1. + 0.5) * np.exp(1j * 2. * np.pi * (1./3 * k1 + 1./3 * k2)).reshape(nk)
    dkyphik = 1j * 1./3 * (0. + np.sqrt(3)/2.) * np.exp(1j * 2. * np.pi * (1./3 * k1 + 1./3 * k2)).reshape(nk)

    dkxphis = np.moveaxis(np.concatenate((zrs,zrs,zrs,dkxphik),axis=0).reshape(2,2,nk),-1,0)
    dkyphis = np.moveaxis(np.concatenate((zrs,zrs,zrs,dkyphik),axis=0).reshape(2,2,nk),-1,0)

    dkxUk = np.kron(dkxphis,np.eye(2))
    dkyUk = np.kron(dkyphis,np.eye(2))

    dkxUkd = np.conjugate(np.transpose(dkxUk,(0,2,1)))
    dkyUkd = np.conjugate(np.transpose(dkyUk,(0,2,1)))
    
    return dkxUk, dkyUk, dkxUkd, dkyUkd
   
def Vez_KM():
    """returns the volume of the unit cell in the Kane Mele model"""
    return np.sqrt(3)/2