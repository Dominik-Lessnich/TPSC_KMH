import numpy as np
from scipy.optimize import brentq
from Mesh import *

class GF0:
    def __init__(self, mesh, n):
        """
        Class to create a non-interacting Green function
        Inputs:
        mesh: Mesh object from the Mesh.py file
        n: input density
        Credits for part of the code: Niklas Witt, Chloé Gauvin-Ndiaye
        """
        # Initialize the input quantities
        self.mesh = mesh
        self.n = n

        # Calculate the chemical potential
        self.calcMuFromDensity()

    def calcMuFromDensity(self):
        """
        Calculate the chemical potential from the density n given at initialization.
        Sets mu.
        """
        self.mu = brentq(lambda m: self.calcNfromMu(m)-self.n/2., np.amin(self.mesh.ek), np.amax(self.mesh.ek), disp=True) #self.n = 2 corresponds to full
        

    def calcNfromMu(self, mu):
        """
        Calculate the non-interacting density n from an input chemical potential mu.
        Returns n from mu. n is filling per orbital for symmetrical orbitals
        """
        f_k = 1./(np.exp((self.mesh.ek-mu)/self.mesh.T)+1)
        return 1./(self.mesh.nk * self.mesh.norb)*np.sum(f_k)
    
    def calcGkiwn(self):
        """
        Calculate the non-interactin Green function G(k,iwn)
        """
        self.gkiwn = np.linalg.inv(np.tensordot(self.mesh.iwn_f_ + self.mu, np.eye(self.mesh.norb),axes = 0) - self.mesh.hk )

    def calcGrtau(self):
        """
        Calculate real space Green function G(tau,r) for calculating chi0 and sigma
        """
        # Fourier transform
        # Calculation of G
        grtau = self.mesh.k_to_r(self.gkiwn)
        self.grtau = self.mesh.wn_to_tau('F', grtau)

    def calcGmrtau(self):
        """
        Calculate real space Green function G(tau,-r) for calculating chi0 and sigma
        """
        # Fourier transform
        # Calculation of G
        gmrtau = self.mesh.k_to_mr(self.gkiwn)
        self.gmrtau = self.mesh.wn_to_tau('F', gmrtau)
    
    def calcTraceG(self):
        """
        Function to calculate the trace of G0 without an input mu
        Used to compute the chemical potential for the Green function G2, no e^-iwn0^- with 1/iwn correction
        """
        # Trace of G0
        gio  = np.sum(np.matrix.trace(self.gkiwn,axis1=2,axis2=3),axis=1)/self.mesh.nk
        g_l  = self.mesh.IR_basis_set.smpl_wn_f.fit(gio)
        g_tau0p = self.mesh.IR_basis_set.basis_f.u(0)@g_l #interpolation from positiv frequencies to 0+
        g_tau0m = g_tau0p + 1. * self.mesh.norb 
        return g_tau0m.real
            
    