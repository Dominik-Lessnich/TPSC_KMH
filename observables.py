import numpy as np
from GF0 import *
from IntGF import *
from Mesh import *
from TPSC import *


class obs:
    def __init__(self, tpsc):

        self.tpsc = tpsc


    def calc_chis_afm(self):
        """
        calculates afm susceptibilies for charge and spin only valid for the Honeycomb lattice
        afm (Sa - Sb)(Sa - Sb)
        """
        self.chixx_afm = (self.tpsc.chixx[:,:,0,0] - self.tpsc.chixx[:,:,0,1] - self.tpsc.chixx[:,:,1,0] + self.tpsc.chixx[:,:,1,1]).reshape(-1,self.tpsc.mesh.nk1,self.tpsc.mesh.nk2)
        self.chizz_afm = (self.tpsc.chizz[:,:,0,0] - self.tpsc.chizz[:,:,0,1] - self.tpsc.chizz[:,:,1,0] + self.tpsc.chizz[:,:,1,1]).reshape(-1,self.tpsc.mesh.nk1,self.tpsc.mesh.nk2)
        self.chicc_afm = (self.tpsc.chicc[:,:,0,0] - self.tpsc.chicc[:,:,0,1] - self.tpsc.chicc[:,:,1,0] + self.tpsc.chicc[:,:,1,1]).reshape(-1,self.tpsc.mesh.nk1,self.tpsc.mesh.nk2)


    def calc_Z(self):
        '''
        Calculates the band gap renormalization Z(0,K) from by 1/(1-Im(Sigma(iw0,K)/w0)))
        Assumes a diagonal self-energy wit identical entries
        '''
        w0 = np.pi * self.tpsc.mesh.T
        self.Sigmaw0 = np.reshape(self.tpsc.selfEnergy[self.tpsc.mesh.iw0_f,:,:,:],(self.tpsc.mesh.nk1,self.tpsc.mesh.nk1,self.tpsc.mesh.norb, self.tpsc.mesh.norb)) 
        self.Z0 = 1./(1- np.imag(self.Sigmaw0)/w0)

        #K= (2/3,1/3) in reduced coordinats
        Ki = np.array([int(2/3 * self.tpsc.mesh.nk1),int(1/3 * self.tpsc.mesh.nk1)])
        self.ZK = self.Z0[Ki[0],Ki[1],0,0]
        

    def calc_xi(self, chi_use):
        """
        calculates the correlation length xi for given chi[iqm,kx,ky]
        assumes peak position at (0,0)
        xi = 1/q_HM , where q_HM is the value such that chi_max/2 = chi(q_HM)
        """
        chi1d = chi_use[self.tpsc.mesh.iw0_b,0,:].real
        chihalf = chi1d[0]/2.
        qyi = 0
        chitemp = chi1d[qyi]

        while (qyi < self.tpsc.mesh.nk1-1 and chitemp > chihalf):
            qyi+= 1
            chitemp = chi1d[qyi]
        
        q0 = 2 * np.pi * (qyi -1)/self.tpsc.mesh.nk1 #need to do interpolation so -1 in index
        q_HM = q0 + 2 * np.pi/self.tpsc.mesh.nk1 *(chi1d[qyi-1]- chihalf)/(chi1d[qyi-1]-chi1d[qyi])#correction with linear interpolation because of finite grid
        xi = 1./q_HM

        if qyi == self.tpsc.mesh.nk1-1: #if it is not possible to determine something return -1
            return -1.

        return xi

    def calc_afm_xis(self):
        """
        Calculates interacting correlation lengths for chis xx,zz,cc        
        """
        self.xi_afm_z = self.calc_xi(self.chizz_afm)
        self.xi_afm_x = self.calc_xi(self.chixx_afm)
        self.xi_afm_c = self.calc_xi(self.chicc_afm)
