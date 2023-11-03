import numpy as np
    
def KaneMele(k1, k2, nk, t=1., lmb=0., lmbr=0.):
    hk = -t * (1 + np.exp(1j * 2. * np.pi * k1) + np.exp(1j * 2. * np.pi * k2)).reshape(nk)
    zrs = np.zeros(nk)
    graphene = np.moveaxis(np.concatenate((zrs,hk,np.conjugate(hk),zrs),axis=0).reshape((2,2,nk)),-1,0)
    graphene_sf = np.kron(graphene,np.eye(2))

    gak = 2 * lmb * (np.sin(2. * np.pi * k1) -np.sin(2. * np.pi * k2) + np.sin(2. * np.pi * (-k1 + k2))).reshape(nk)
    lmb_term = np.moveaxis(np.concatenate((gak,zrs,zrs,-1.*gak),axis=0).reshape((2,2,nk)),-1,0)
    sigz = np.array([[1., 0.], [0., - 1.]])
    soc = np.kron(lmb_term,sigz)

    return graphene_sf + soc
   