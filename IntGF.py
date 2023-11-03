import numpy as np
from scipy.optimize import brentq
from Mesh import *

class GF2:
    def __init__(self, mesh, n, selfEnergy):
        """
        Class to create an interacting Green function
        Inputs:
        mesh: Mesh object from the Mesh.py file
        n: input density
        selfEnergy: Table with the self-energy 
        Credits for part of the code: Niklas Witt, Chloé Gauvin-Ndiaye
        """
        # Initialize the input quantities
        self.mesh = mesh
        self.n = n

        # Set the self-energy
        self.selfEnergy = selfEnergy

    
    def calcGkiwnFromMu(self, mu):
        """
        Calculate Green function G2(k,iwn) from an input chemical potential
        """
        self.gkiwn = np.linalg.inv(np.tensordot(self.mesh.iwn_f_ + mu, np.eye(self.mesh.norb),axes = 0) - self.mesh.hk - self.selfEnergy)

    def calcGrtau(self):
        """
        Calculate real space Green function G(tau,r) 
        """
        # Fourier transform
        # Calculation of G
        grtau = self.mesh.k_to_r(self.gkiwn)
        self.grtau = self.mesh.wn_to_tau('F', grtau)
    
    def calcTraceG(self, mu):
        """
        Function to calculate the trace of G, no e^-iwn0^- with 1/iwn correction
        Used to compute the chemical potential for the Green function G2.
        """
        self.calcGkiwnFromMu(mu)
        # Trace of G
        g2io = np.sum(np.matrix.trace(self.gkiwn,axis1=2,axis2=3),axis=1)/self.mesh.nk
        g2_l  = self.mesh.IR_basis_set.smpl_wn_f.fit(g2io)
        g2_tau0p = self.mesh.IR_basis_set.basis_f.u(0)@g2_l #interpolation from positiv frequencies to 0+
        g2_tau0m = g2_tau0p + 1. * self.mesh.norb 
        return g2_tau0m.real
    
    def calcMu2(self, traceG0):
        """
        Calculate the chemical potential for the Green function G2
        """
        self.mu = brentq(lambda m: self.calcTraceG(m)-traceG0, np.amin(self.mesh.ek), np.amax(self.mesh.ek), disp=True)
