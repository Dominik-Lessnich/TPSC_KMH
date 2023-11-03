import numpy as np
from scipy.optimize import root
from GF0 import *
from IntGF import *
from Mesh import *

class TPSC:
    def __init__(self, mesh, U, n):
        """
        Class to perform a multiband TPSC calculation with SOC.
        Written by Dominik Lessnich 
        Based on a single-band code by Chloé Gauvin-Ndiaye based on a Sparse-ir Tutorial by Niklas Witt 
        mesh: Mesh object from the Mesh.py file
        U: Hubbard U
        n: input density
        """
        # Initialize the variables
        self.mesh = mesh
        self.U = U
        self.n = n

        # Set the density for the sum rules, use particle hole symmetry to map to the hole doped case in case of electron doping, note no modifications to H(k)
        if (self.n>1):
            self.nsr = 2. - self.n
        else:
            self.nsr = self.n

        #Set the non-interacting Green's function
        self.set_G0()
        
    def calcFirstLevelApprox(self):
        """
        Do the TPSC calculation with chi0 to obtain chixx, chiyy, chizz and chicc from the sum rules
        """
        # Calculate chi0 (and check the sum rule not commented in)
        self.calcChi0()
        self.calcChi0p()

        # Calculate Uxx and Uyy first also sets chixx and chiyy
        self.calcUxx(self.chi0xx)

        # Calculate the double occupancy
        self.docc = self.calcDoubleOccupancy()

        # Calculate Uzz and Ucc, truncated problem as a starting guess
        self.calcUzz_decoupled(self.chi0zz)
        self.calcUcc_decoupled()

        # Calculation of of Uzz and Ucc with the coupled equations, also sets interacting chis
        self.calcUzzUcc()
    
    def calcSecondLevelApprox(self):
        """
        Function to calculate the self-energy in the second level of approximation
        Restores crossing symmetry and time-reversal by averaging over
        Note: The function calcFirstLevelApprox must be called before this one
        """
        
        chipmr = self.mesh.k_to_mr(self.chip)
        chiptaumr = self.mesh.wn_to_tau('B', chipmr)
        del chipmr #free memory

        #self-energy longitudinal channel
        # Get V(-r,tau) and its time-reversed partner 
        sigx = np.kron(np.eye(self.mesh.norbsl),np.array([[0.,1.],[1.,0.]]))

        Vmr = self.U/4 * np.matmul(sigx,np.transpose(np.matmul(np.matmul(self.A, self.Gammap),np.matmul(chiptaumr, self.A)) ,(0,1,3,2) ))
        VmrTR = self.U/4 * np.matmul(np.transpose(np.matmul(np.matmul(self.A, chiptaumr),np.matmul(self.Gammap, self.A)) ,(0,1,3,2) ),sigx)
        del chiptaumr #free memory

        # Calculate the self-energy in (r,tau) space, TR averaging
        selfEnergyl = 0.5 * (self.g0.grtau * (Vmr[::-1,:] + VmrTR[::-1,:])) #V(-tau) = V(beta-tau)
        del Vmr, VmrTR #free memory

        #self-energy transversal channel
        chiptmr = self.mesh.k_to_mr(self.chipt)
        chipttaumr = self.mesh.wn_to_tau('B', chiptmr)
        del chiptmr #free memory

        # Get V(-r,tau), no restoring of time-reversal symmetry needed here
        Vtmr = -self.U/4 * np.transpose(np.matmul(np.matmul(self.A, self.Gammapt),np.matmul(chipttaumr, self.A)) ,(0,1,3,2) )
        del chipttaumr #free memory

        selfEnergyt = np.matmul(sigx,np.matmul(self.g0.grtau , sigx)) * np.matmul(sigx,np.matmul(Vtmr[::-1,:], sigx))
        del Vtmr #free memory

        # Fourier transform
        selfEnergyl = self.mesh.r_to_k(selfEnergyl) 
        selfEnergyl = self.mesh.tau_to_wn('F', selfEnergyl)

        selfEnergyt = self.mesh.r_to_k(selfEnergyt) 
        selfEnergyt = self.mesh.tau_to_wn('F', selfEnergyt)

        #averaging of longitudinal channel and transversal channel
        self.selfEnergy = 0.5 * (selfEnergyl + selfEnergyt)

        #Hartree term, not needed can be absorbed in chemical potential
        
        # Calculate G2
        self.g2out = GF2(self.mesh, self.n, self.selfEnergy)
        traceG0 = self.g0.calcTraceG()
        self.g2out.calcMu2(traceG0)
        self.g2out.calcGkiwnFromMu(self.g2out.mu)
        self.g2out.calcGrtau()
    
    def calcUxx(self,chi0xx):
        """
        Function to calculate Uxx and Uyy from sum rule
        """
        # Bounds on the value of Uxx
        Uxxmin = 0.
        Uxxmax = 2./np.amax(np.linalg.eigvals(chi0xx)).real-1e-7

        # Calculate Uxx and Uyy
        if self.U==0:
            self.Uxx = 0
            self.Uyy = 0
        else:
            self.Uxx = brentq(lambda m: self.calcSumChisp(m,chi0xx)-self.calcSumRuleChisp(m), Uxxmin, Uxxmax, disp=True)
            self.Uyy = self.Uxx #spin rotation sym arround z axis

        #setting chipt from given Uxx and Uyy
        self.Gammapt = np.kron(np.eye(self.mesh.norbsl),-np.diag([self.Uxx,-self.Uyy])) #note the minus sign, so that one has positive quantities for Uxx and Uyy like in truncated rootsearch
        self.chipt = np.matmul(np.linalg.inv(np.eye(self.mesh.norb)+0.5*np.matmul(self.chi0pt,self.Gammapt)) , self.chi0pt)

        #setting individual chits
        self.chixx = self.chipt[:,:,::2,::2]
        self.chimyy = self.chipt[:,:,1::2,1::2]


    def calcSumChisp(self, Usp, chi0sp):
        """
        Function to compute the trace of chi0sp/(1-Usp/2*chi0sp), for symmetric orbitals looking at the first one is sufficiet
        tr stands for truncated here, because we neglect coupling chi0s like chi0cz
        """
        chisptr = np.matmul(np.linalg.inv(np.eye(self.mesh.norbsl)-0.5*Usp*chi0sp), chi0sp)
        chisptr_trace = np.sum(chisptr[:,:,0,0], axis=1)/self.mesh.nk
        chisptr_trace_l  = self.mesh.IR_basis_set.smpl_wn_b.fit(chisptr_trace)
        chisptr_trace = self.mesh.IR_basis_set.basis_b.u(0)@chisptr_trace_l

        return chisptr_trace.real
    
    def calcSumRuleChisp(self, Usp):
        """
        Calculate the spin susceptibility sum rule for a specific Usp
        """
        if self.U == 0:
            return self.nsr-self.nsr*self.nsr/2.
        else:
            return self.nsr-Usp*self.nsr*self.nsr/(2.*self.U)

    def calcDoubleOccupancy(self):
        """
        Function to compute the double occupancy.
        Note: the function calcUxx has to be called before this one
        """
        if self.U == 0:
            return self.n*self.n/4
        else:
            if (self.n<1):
                return self.Uxx/self.U*self.n*self.n/4
            else:
                return self.Uxx/(4*self.U)*self.nsr*self.nsr-1+self.n

    def calcUzz_decoupled(self,chi0zz):
        """
        Function to compute Uzz from chi0 and the sum rule, decoupled for starting guess of vertex
        Note: calcUxx has to be called before this function
        """
        # Bounds on the value of Uzz
        Uzzmin = 0.
        Uzzmax = 2./np.amax(np.linalg.eigvals(chi0zz)).real-1e-7

        # Calculate Uzz
        if self.U==0:
            self.Uzz = 0
        else:
            self.Uzz = brentq(lambda m: self.calcSumChisp(m,chi0zz)-self.calcSumRuleChisp(self.Uxx), Uzzmin, Uzzmax, disp=True)


    def calcUcc_decoupled(self, Uccmin=0., Uccmax=100.):
        """
        Function to compute Ucc from chi0 and the sum rule, decoupled for starting guess of vertex
        Note: calcUxx has to be called before this function
        """
        # Calculate Ucc
        if self.U==0:
            self.Ucc = 0
        else:
            self.Ucc = brentq(lambda m: self.calcSumChicc(m)-self.calcSumRuleChicc(self.Uxx), Uccmin, Uccmax, disp=True) 

    def calcSumChicc(self, Ucc):
        """
        Function to compute the trace of chi0/(1+Ucc/2*chi0)
        tr for truncated here, because we neglect chi0sc
        """
        self.chicctr = np.matmul(np.linalg.inv(np.eye(self.mesh.norbsl)+0.5*Ucc*self.chi0cc) , self.chi0cc)
        chicctr_trace = np.sum(self.chicctr[:,:,0,0], axis=1)/self.mesh.nk
        chicctr_trace_l  = self.mesh.IR_basis_set.smpl_wn_b.fit(chicctr_trace)
        chicctr_trace = self.mesh.IR_basis_set.basis_b.u(0)@chicctr_trace_l

        return chicctr_trace.real
    
    def calcSumRuleChicc(self, Usp):
        """
        Calculate the charge susceptibility sum rule for a specific Usp
        """
        if self.U == 0:
            return self.nsr+self.nsr*self.nsr/2-self.nsr*self.nsr
        else:
            return self.nsr+Usp*self.nsr*self.nsr/(2*self.U)-self.nsr*self.nsr

    def calcUzzUcc(self):
        """
        Calculate Uzz and Ucc with multidimensional rootsearch and the full expressions for the susceptibilities
        As a starting point use Uzz and Ucc from the truncated root finding problem
        """

        self.Uzztr = self.Uzz
        self.Ucctr = self.Ucc

        # Calculate Ucc and Uzz 
        if self.U==0:
            num_small = self.rootfunction_chicczz([0.,0.]) #chis have to be set here
            self.Ucc = 0.
            self.Uzz = 0.
        else:
            sol = root(self.rootfunction_chicczz, np.array([self.Ucc, self.Uzz]), method='hybr',tol=2.e-12)
            self.Ucc= sol.x[0]
            self.Uzz= sol.x[1]

        #check that chizz and chicc are positiv on the diagonals, to have a physical solution
        self.chiccmin=np.amin(np.diagonal(self.chicc, axis1=2, axis2=3))
        self.chizzmin=np.amin(np.diagonal(self.chizz, axis1=2, axis2=3))

        if (self.chiccmin < -1e-10):
            print('chiccmin = ', self.chiccmin)
            chiccdiag = np.diagonal(self.chicc, axis1=2, axis2=3)
            print(np.unravel_index(np.argmin(chiccdiag),chiccdiag.shape))

        if (self.chizzmin < -1e-10):
            print('chizzmin = ', self.chizzmin)
            chizzdiag = np.diagonal(self.chizz, axis1=2, axis2=3)
            print(np.unravel_index(np.argmin(chizzdiag),chizzdiag.shape))     

    def rootfunction_chicczz(self,Gamma):
        """
        function to calculate the root of to search Gamma elements
        Gamma consists of list [Ucc,Uzz]
        """
        return self.calcSumChicczz(Gamma[0],Gamma[1]) - self.calcSumRuleChicczz(self.Uxx)

    def calcSumChicczz(self, Ucc, Uzz):
        """
        function to calculate the trace of chizz and chicc with the full expressions
        used in the multidimensional rootsearch
        also sets all chip and Gammap
        """
        #setting chip from given Ucc and Uzz
        self.Gammap = np.kron(np.eye(self.mesh.norbsl),np.diag([Ucc,-Uzz]))
        self.chip = np.matmul(np.linalg.inv(np.eye(self.mesh.norb)+0.5*np.matmul(self.chi0p,self.Gammap)) , self.chi0p)

        #calculate the trace of chi, assuming identical orbitals 
        chicc_trace_l1 = np.sum(self.chip[:,:,0,0], axis=1)/self.mesh.nk
        chizz_trace_l1 = np.sum(self.chip[:,:,1,1], axis=1)/self.mesh.nk

        chicc_trace_l = self.mesh.IR_basis_set.smpl_wn_b.fit(chicc_trace_l1)
        chicc_trace = self.mesh.IR_basis_set.basis_b.u(0)@chicc_trace_l
        chizz_trace_l = self.mesh.IR_basis_set.smpl_wn_b.fit(chizz_trace_l1)
        chizz_trace = self.mesh.IR_basis_set.basis_b.u(0)@chizz_trace_l

        #setting individual chis
        self.chicc = self.chip[:,:,::2,::2]
        self.chizz = self.chip[:,:,1::2,1::2]
        self.chicz = self.chip[:,:,::2,1::2]
        self.chizc = self.chip[:,:,1::2,::2]

        return np.array([chicc_trace,chizz_trace]).real

    def calcSumRuleChicczz(self, Usp):
        """
        Calculates the charge and spin susceptibility sum rule for a specific Usp and U
        used in the multidimensional rootsearch
        """
        return np.array([self.calcSumRuleChicc(Usp), self.calcSumRuleChisp(Usp)])

    def set_G0(self):
        """
        Function to calculate G0(iwn,k), G0(r,tau), G0(,r,tau), (if not passed at initialization)
        """
        # Calculate the Green function G0 (if not passed at initialization)
        self.g0 = GF0(self.mesh, self.n)
        self.g0.calcGkiwn() 

        # Calculate G0(r,tau)
        self.g0.calcGrtau()
        self.g0.calcGmrtau()

    def calcChi0(self):
        """
        Function to calculate chi0(q,iqn), longitudinal and transversal ones
        """
        # Calculate longitudinal susceptibilities chi0(r,tau) 
        self.chi0 = self.g0.grtau * np.transpose(self.g0.gmrtau[::-1, :,:,:],(0,1,3,2)) #no minus because of anti-periodicity relation

        # Fourier transform
        self.chi0 = self.mesh.r_to_k(self.chi0)
        self.chi0 = self.mesh.tau_to_wn('B', self.chi0)

        #transversal susceptibilities 
        sigx = np.kron(np.eye(self.mesh.norbsl),np.array([[0.,1.],[1.,0.]]))
        self.chi0t = np.matmul(sigx,self.g0.grtau) * np.matmul(np.transpose(self.g0.gmrtau[::-1, :,:,:],(0,1,3,2)),sigx) #no minus because of anti-periodicity relation

        # Fourier transform 
        self.chi0t = self.mesh.r_to_k(self.chi0t)
        self.chi0t = self.mesh.tau_to_wn('B', self.chi0t)

    def calcChi0p(self):
        """
        rotates to physical chis i.e. ((ch,cz),(zc,zz)) and ((xx,mixy),(iyx,myy))
        """
        self.A = np.kron(np.eye(self.mesh.norbsl),np.array([[1.,1.],[1.,-1.]]))
        self.chi0p = np.matmul(np.matmul(self.A,self.chi0),self.A)

        self.chi0cc = self.chi0p[:,:,::2,::2]
        self.chi0zz = self.chi0p[:,:,1::2,1::2]
        self.chi0cs = self.chi0p[:,:,::2,1::2]
        self.chi0sc = self.chi0p[:,:,1::2,::2]

        #transversal susceptibilities
        self.chi0pt = np.matmul(np.matmul(self.A,self.chi0t),self.A)
        
        self.chi0xx = self.chi0pt[:,:,::2,::2]
        self.chi0myy = self.chi0pt[:,:,1::2,1::2]

    def calcTraceSelfG(self,G_use):
        """
        Calculate 1/2 trace of Self-Energy*G_use, with 1/iwn correction term
        Note: functions to calculate first and second levels of approximation must be called before this one
        """
        trace = 0.5 * np.sum(np.matrix.trace(np.matmul(self.selfEnergy,G_use), axis1=2,axis2=3), axis=1)/self.mesh.nk
        trace_l  = self.mesh.IR_basis_set.smpl_wn_f.fit(trace)
        self.traceSG = self.mesh.IR_basis_set.basis_f.u(0)@trace_l
        self.traceSG += self.U/4.0 * self.n  * self.n * self.mesh.norbsl #1/iwn correction from e^.iwn0^- times Hartree term
        self.traceSG = 1.0 /self.mesh.norbsl *self.traceSG #normalize on to per spinless orbital







