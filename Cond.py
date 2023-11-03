import numpy as np
from GF0 import *
from IntGF import *
from Mesh import *
from TPSC import *
import bare_vertices


class Cond:
    def __init__(self, tpsc, dHk_func = bare_vertices.Kanemele_dHk, gauge = bare_vertices.Kanemele_gauge_Uk, dgauge = bare_vertices.Kanemele_gauge_dUk ,fVez = bare_vertices.Vez_KM, hops = (1.,0.,0.)):
        """
        Holding class for objects that are necesarry for the calculation of the spin Hall conductivity
        Input:
        tpsc: TPSC object from TPSC.py
        dHk_func: function that returns bare vertices from file bare_vertices.py
        gauge: function that returns gauge matrices from file bare_vertices.py
        dgauge: function that returns gauge matrices from file bare_vertices.py
        fVez: function that returns volume of elementary unitcell from file file bare_vertices.py
        hops: hopping elements
        """
        self.tpsc = tpsc

        #bare vertices and gauge matrices, gauge matrices so that orbital position in the unit cell is included
        self.Vez = fVez()
        self.Uks, self.Uksd = gauge(self.tpsc.mesh.k1, self.tpsc.mesh.k2, self.tpsc.mesh.nk) #gauge matrix to go in the gauge where A derivs can be replaced by k derivs

        #bare vertices in periodic gauge and derivatives of gauge matrix, simpler for SHC with vertex corrections
        dkHk = dHk_func(self.tpsc.mesh.k1, self.tpsc.mesh.k2, self.tpsc.mesh.nk, *hops), #just the derivative of H(k) in the periodic gauge
        self.dkxHk = dkHk[0][0]
        self.dkyHk = dkHk[0][1]
        self.dkxUks, self.dkyUks, self.dkxUksd, self.dkyUksd  = dgauge(self.tpsc.mesh.k1, self.tpsc.mesh.k2, self.tpsc.mesh.nk,self.tpsc.g0.gkiwn.shape[0])#derivatives of the gauge matrix

        #gauge change of the Greens function
        self.gkiwn_g = np.matmul(self.Uksd,np.matmul(self.tpsc.g2out.gkiwn,self.Uks))


    def calc_SHC_G(self,no_vertex_corr=True):
        """
        Calculates the SHC with TPSC 
        Vertex corrections from k- and frequency dependence of the self-energy are included
        Vertex corrections corresponds to maki-thompson type contributions in TPSC
        For no_vertex_corr=True it also calculates the SHC from the conductivity bubble
        """

        #construct vertex corrections

        #G and non-intvertex
        G1dkxHkG1 = np.matmul(np.matmul(self.tpsc.g0.gkiwn,self.dkxHk),self.tpsc.g0.gkiwn)
        G1dkyHkG1 = np.matmul(np.matmul(self.tpsc.g0.gkiwn,self.dkyHk),self.tpsc.g0.gkiwn)
        G1G1 = np.matmul(self.tpsc.g0.gkiwn,self.tpsc.g0.gkiwn)

        G1dkxHkG1taur = self.tpsc.mesh.wn_to_tau('F', self.tpsc.mesh.k_to_r(G1dkxHkG1))
        G1dkyHkG1taur = self.tpsc.mesh.wn_to_tau('F', self.tpsc.mesh.k_to_r(G1dkyHkG1))
        G1G1taur = self.tpsc.mesh.wn_to_tau('F', self.tpsc.mesh.k_to_r(G1G1))

        del G1dkxHkG1, G1dkyHkG1, G1G1 #free memory

        #spin fluctuation potential part in tau ,r 
        A = self.tpsc.A
        sigx = np.kron(np.eye(self.tpsc.mesh.norbsl),np.array([[0.,1.],[1.,0.]]))

        chiptaumr = self.tpsc.mesh.wn_to_tau('B', self.tpsc.mesh.k_to_mr(self.tpsc.chip))
        Vltaumr = self.tpsc.U/4 * np.matmul(sigx,np.transpose(np.matmul(np.matmul(A, self.tpsc.Gammap),np.matmul(chiptaumr, A)) ,(0,1,3,2) ))
        VltaumrTR = self.tpsc.U/4 * np.matmul(np.transpose(np.matmul(np.matmul(A, chiptaumr),np.matmul(self.tpsc.Gammap, A)) ,(0,1,3,2) ),sigx)

        chipttaumr = self.tpsc.mesh.wn_to_tau('B',self.tpsc.mesh.k_to_mr(self.tpsc.chipt))
        Vttaumr = -self.tpsc.U/4 * np.matmul(np.matmul(sigx,np.transpose(np.matmul(np.matmul(A, self.tpsc.Gammapt),np.matmul(chipttaumr, A)) ,(0,1,3,2) )),sigx)

        Vlmtaumr = 0.5 * (Vltaumr[::-1,:] + VltaumrTR[::-1,:]) #-tau here, V(-tau) = V(beta-tau)
        Vtmtaumr = Vttaumr[::-1,:] #-tau here, V(-tau) = V(beta-tau)

        del chiptaumr, Vltaumr, VltaumrTR, chipttaumr, Vttaumr #free memory

        #put GvertG and V together for vertex correction in tau r, spinflip for transversal part, then FT
        dkxSigtaur = 0.5 * (G1dkxHkG1taur * Vlmtaumr + np.matmul(sigx,np.matmul(G1dkxHkG1taur,sigx)) * Vtmtaumr)
        dkySigtaur = 0.5 * (G1dkyHkG1taur * Vlmtaumr + np.matmul(sigx,np.matmul(G1dkyHkG1taur,sigx)) * Vtmtaumr)
        diwnSigtaur = -0.5 * (G1G1taur * Vlmtaumr + np.matmul(sigx,np.matmul(G1G1taur,sigx)) * Vtmtaumr) #minus here is important

        del G1dkxHkG1taur, G1dkyHkG1taur, G1G1taur, Vlmtaumr, Vtmtaumr #free memory

        dkxSig = self.tpsc.mesh.tau_to_wn('F',self.tpsc.mesh.r_to_k(dkxSigtaur))
        dkySig = self.tpsc.mesh.tau_to_wn('F',self.tpsc.mesh.r_to_k(dkySigtaur))
        diwnSig = self.tpsc.mesh.tau_to_wn('F',self.tpsc.mesh.r_to_k(diwnSigtaur))

        del dkxSigtaur, dkySigtaur, diwnSigtaur #free memory

        #add gauges and derivatives of gauges in case of dkx and dky
        dkxSig_g = np.matmul(np.matmul(self.Uksd,dkxSig),self.Uks) + np.matmul(np.matmul(self.dkxUksd,self.tpsc.selfEnergy), self.Uks) + np.matmul(np.matmul(self.Uksd,self.tpsc.selfEnergy), self.dkxUks) 
        dkySig_g = np.matmul(np.matmul(self.Uksd,dkySig),self.Uks) + np.matmul(np.matmul(self.dkyUksd,self.tpsc.selfEnergy), self.Uks) + np.matmul(np.matmul(self.Uksd,self.tpsc.selfEnergy), self.dkyUks)
        diwnSig_g = np.matmul(np.matmul(self.Uksd,diwnSig),self.Uks)

        del dkxSig, dkySig, diwnSig #free memory

        # the bare vertices in the gauge 
        dkxHk_g = np.matmul(np.matmul(self.Uksd,self.dkxHk),self.Uks) + np.matmul(np.matmul(self.dkxUksd,self.tpsc.mesh.hk), self.Uks) + np.matmul(np.matmul(self.Uksd,self.tpsc.mesh.hk), self.dkxUks) 
        dkyHk_g = np.matmul(np.matmul(self.Uksd,self.dkyHk),self.Uks) + np.matmul(np.matmul(self.dkyUksd,self.tpsc.mesh.hk), self.Uks) + np.matmul(np.matmul(self.Uksd,self.tpsc.mesh.hk), self.dkyUks)

        #Now construct the dressed vertices dnuG^-1(k) with gauge phases included here
        
        Lambdax = - dkxHk_g - dkxSig_g
        Lambday = - dkyHk_g - dkySig_g
        Lambdaiwn = (np.eye(self.tpsc.mesh.norb) - diwnSig_g)
        
        #evaluate the SHC triangle formular, sigz for spin Hall cond
        GdkxGm1 = np.matmul(self.gkiwn_g, Lambdax)
        GdkyGm1 = np.matmul(self.gkiwn_g, Lambday)
        GdiwnGm1 = np.matmul(self.gkiwn_g, Lambdaiwn)

        sigz = np.kron(np.eye(self.tpsc.mesh.norbsl),np.array([[1.,0.],[0.,-1.]])) #sigma_z so we get signs for spin Hall

        GdGm1x3s = np.matmul(np.matmul(sigz, GdiwnGm1),np.matmul(GdkxGm1,GdkyGm1))
        GdGm1x3s_asym = np.matmul(np.matmul(sigz, GdiwnGm1),np.matmul(GdkyGm1,GdkxGm1))
        
        triangle = GdGm1x3s - GdGm1x3s_asym #antisymmetric part
        prefactor = -1j * (2 * np.pi ) * 0.5 / self.Vez #2 pi from transforming continous integral to matsubara frequencies, 0.5 from antisym, i also from integral
        
        tr_ksum_triangle = np.sum(np.matrix.trace(triangle, axis1=2,axis2=3), axis=1)/self.tpsc.mesh.nk
        tr_ksum_triangle_l = self.tpsc.mesh.IR_basis_set.smpl_wn_f.fit(tr_ksum_triangle)
        
        self.SHC = np.real(prefactor * self.tpsc.mesh.IR_basis_set.basis_f.u(0)@tr_ksum_triangle_l)
        
        #evaluate the conductivity bubble if no_vertex_corr=True
        if no_vertex_corr:
            Lambdax = - dkxHk_g
            Lambday = - dkyHk_g
            Lambdaiwn = (np.eye(self.tpsc.mesh.norb) - diwnSig_g) # contains the vertex corrections on the frequency vertex

            #evaluate the SHC triangle formular, sigz for spin Hall cond
            GdkxGm1 = np.matmul(self.gkiwn_g, Lambdax)
            GdkyGm1 = np.matmul(self.gkiwn_g, Lambday)
            GdiwnGm1 = np.matmul(self.gkiwn_g, Lambdaiwn)

            sigz = np.kron(np.eye(self.tpsc.mesh.norbsl),np.array([[1.,0.],[0.,-1.]])) #sigma_z so we get signs for spin Hall

            GdGm1x3s = np.matmul(np.matmul(sigz, GdiwnGm1),np.matmul(GdkxGm1,GdkyGm1))
            GdGm1x3s_asym = np.matmul(np.matmul(sigz, GdiwnGm1),np.matmul(GdkyGm1,GdkxGm1))
        
            triangle = GdGm1x3s - GdGm1x3s_asym #antisymmetric part
            prefactor = -1j * (2 * np.pi ) * 0.5 / self.Vez #2 pi from transforming continous integral to matsubara frequencies, 0.5 from antisym, i also from integral
        
            tr_ksum_triangle = np.sum(np.matrix.trace(triangle, axis1=2,axis2=3), axis=1)/self.tpsc.mesh.nk
            tr_ksum_triangle_l = self.tpsc.mesh.IR_basis_set.smpl_wn_f.fit(tr_ksum_triangle)

            self.SHC_nvc = np.real(prefactor * self.tpsc.mesh.IR_basis_set.basis_f.u(0)@tr_ksum_triangle_l)
        


