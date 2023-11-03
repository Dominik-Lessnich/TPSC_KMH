import Dispersions
import numpy as np


class Mesh2DSquare:
    """
    Holding class for k-mesh and sparsely sampled imaginary time 'tau' / Matsubara frequency 'iw_n' grids.
    Additionally it defines the Fourier transform routines 'r <-> k'  and 'tau <-> l <-> wn'.
    This is valid for the 2D square lattice case, but other 2D lattices can be mapped onthat case by k = k_1 b_1 + k_2 b_2
    Credit for parts of the code: Niklas Witt
    https://spm-lab.github.io/sparse-ir-tutorial/src/TPSC_py.html
    """
    def __init__(self,IR_basis_set, nk1, nk2, T,dispersion = Dispersions.KaneMele , hops = (1.,0.,0.)):
        self.IR_basis_set = IR_basis_set
        self.T = T

        # generate k-mesh and dispersion
        self.nk1, self.nk2, self.nk = nk1, nk2, nk1*nk2
        self.k1, self.k2 = np.meshgrid(np.arange(self.nk1)/self.nk1, np.arange(self.nk2)/self.nk2)
        self.hk = dispersion(self.k1, self.k2, self.nk, *hops)
        self.norb = np.shape(self.hk)[2]
        self.norbsl = int(self.norb/2)
        self.ek = np.linalg.eigvals(self.hk)
        self.ek = np.sort(self.ek).real

        # lowest Matsubara frequency index
        self.iw0_f = np.where(self.IR_basis_set.wn_f == 1)[0][0]
        self.iw0_b = np.where(self.IR_basis_set.wn_b == 0)[0][0]

        ### Generate a frequency-momentum grid for iw_n and Hk (in preparation for calculating the Green function)
        # frequency mesh (for Green function)
        self.iwn_f = 1j * self.IR_basis_set.wn_f * np.pi * self.T
        self.iwn_f_ = np.tensordot(self.iwn_f, np.ones(self.nk), axes=0)

    def smpl_obj(self, statistics):
        """ Return sampling object for given statistic """
        smpl_tau = {'F': self.IR_basis_set.smpl_tau_f, 'B': self.IR_basis_set.smpl_tau_b}[statistics]
        smpl_wn  = {'F': self.IR_basis_set.smpl_wn_f,  'B': self.IR_basis_set.smpl_wn_b }[statistics]
        return smpl_tau, smpl_wn
    
    def tau_to_wn(self, statistics, obj_tau):
        """ Fourier transform from tau to iw_n via IR basis """
        smpl_tau, smpl_wn = self.smpl_obj(statistics)

        obj_tau = obj_tau.reshape((smpl_tau.tau.size, self.nk1, self.nk2, self.norb, self.norb))
        obj_l   = smpl_tau.fit(obj_tau, axis=0)
        obj_wn  = smpl_wn.evaluate(obj_l, axis=0).reshape((smpl_wn.wn.size, self.nk, self.norb, self.norb))
        return obj_wn

    def wn_to_tau(self, statistics, obj_wn):
        """ Fourier transform from tau to iw_n via IR basis """
        smpl_tau, smpl_wn = self.smpl_obj(statistics)

        obj_wn  = obj_wn.reshape((smpl_wn.wn.size, self.nk1, self.nk2, self.norb, self.norb))
        obj_l   = smpl_wn.fit(obj_wn, axis=0)
        obj_tau = smpl_tau.evaluate(obj_l, axis=0).reshape((smpl_tau.tau.size, self.nk, self.norb, self.norb))
        return obj_tau

    def k_to_r(self,obj_k):
        """ Fourier transform from k-space to real space, normalization here devide by Nk """
        obj_k = obj_k.reshape(-1, self.nk1, self.nk2, self.norb, self.norb)
        obj_r = np.fft.ifftn(obj_k,axes=(1,2))
        obj_r = obj_r.reshape(-1, self.nk, self.norb, self.norb)
        return obj_r

    def k_to_mr(self,obj_k):
        """ Fourier transform from k-space to real space, but -r, normalization here devide by Nk """
        obj_k = obj_k.reshape(-1, self.nk1, self.nk2, self.norb, self.norb)
        obj_r = np.fft.fftn(obj_k,axes=(1,2),norm='forward')
        obj_r = obj_r.reshape(-1, self.nk, self.norb, self.norb)
        return obj_r

    def r_to_k(self,obj_r):
        """ Fourier transform from real space to k-space, normalization here no devide Nk """
        obj_r = obj_r.reshape(-1, self.nk1, self.nk2, self.norb, self.norb)
        obj_k = np.fft.fftn(obj_r,axes=(1,2))
        obj_k = obj_k.reshape(-1, self.nk, self.norb, self.norb)
        return obj_k

    def mr_to_k(self,obj_r):
        """ Fourier transform from real space -r to k-space, normalization here no devide Nk """
        obj_r = obj_r.reshape(-1, self.nk1, self.nk2, self.norb, self.norb)
        obj_k = np.fft.ifftn(obj_r,axes=(1,2),norm='forward')
        obj_k = obj_k.reshape(-1, self.nk, self.norb, self.norb)
        return obj_k