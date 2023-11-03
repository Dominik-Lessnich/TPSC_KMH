#%%
import numpy as np
from GF0 import *
from IntGF import *
from Mesh import *
from TPSC import *
from Cond import *
from observables import *
import sparse_ir
import matplotlib.pyplot as plt


#%%
# Calculation parameters

n = 1.0         #Nummber of electrons per orbital, 1 is each orbital is half filled
T = 0.01        #Temperature
U = 3.          #Hubbard U
t1 = 1.         #Nearest neighbor hopping 
t2 = 0.02       #Spin-orbit coupling parameter
nkx = 90        #k-points in x and y direction, nkx = nky
wmax = 10       #Sparse-ir frequency window
IR_tol = 1e-10  #Sparse-ir tolerance parameter


#%%
#First level approximation for the TPSC vertices

IR_basis_set = sparse_ir.FiniteTempBasisSet(1./T, wmax, eps=IR_tol)
mesh = Mesh2DSquare(IR_basis_set, nkx, nkx, T, dispersion = Dispersions.KaneMele, hops = (t1,t2,0.))
tpsc = TPSC(mesh, U, n)
tpsc.calcFirstLevelApprox()
print("Gamma_cc: {}".format(tpsc.Ucc))
print("Gamma_zz: {}".format(tpsc.Uzz))
print("Gamma_xx: {}".format(tpsc.Uxx))
print("<n_up n_down>: {}".format(tpsc.docc))


# %%
#Second level approximation for the self-energy
tpsc.calcSecondLevelApprox()

# %%
#Tr(Sigma G) consistency test
tpsc.calcTraceSelfG(tpsc.g2out.gkiwn)
print("1/(2*n_orb) Tr(Sigma G): {}".format(tpsc.traceSG))
print("1/(2*n_orb) Tr(Sigma G) - U <n_up n_down>  : {}".format(tpsc.traceSG - tpsc.U * tpsc.docc ))


# %%
#obsevables xi_afm and Z(K)
observables = obs(tpsc)
observables.calc_chis_afm()
observables.calc_afm_xis()
print('xi^afm_z = ',observables.xi_afm_z)
print('xi^afm_x = ',observables.xi_afm_x)

observables.calc_Z()
print('Band gap renormalization factor Z(K) = ', observables.ZK)
print('Bare band gap = ', t2 * np.sqrt(3) * 6)

# %% 
# Spin Hall conductivity (SHC)
cond = Cond(tpsc, dHk_func = bare_vertices.Kanemele_dHk, gauge = bare_vertices.Kanemele_gauge_Uk, dgauge = bare_vertices.Kanemele_gauge_dUk, fVez = bare_vertices.Vez_KM, hops = (t1,t2,0.))
cond.calc_SHC_G()
print("SHC with vertex corrections: {} e^2/h".format(cond.SHC))
print("SHC without vertex corrections: {} e^2/h".format(cond.SHC_nvc))


#%%
font = {'size'   : 13}
plt.rc('font', **font)


#%%
#plot Im Sigma_11_upup at high sym kpts as function of iwn

Gamma = np.array([0,0]) #Gamma = (0.,0.), K= (2/3,1/3), X = (1/2,0.) in red coordinats
Ki = np.array([int(2/3*nkx),int(1/3*nkx)])
Xi = np.array([int(0.5*nkx),int(0.5*nkx)])
wn_array = np.linspace(1,201,num = 101)


Sigma = np.reshape(tpsc.selfEnergy,(-1,nkx,nkx,mesh.norb,mesh.norb))
smpl_wn = mesh.smpl_obj(statistics='F')[1]
obj_l_Gamma = smpl_wn.fit(Sigma[:,Gamma[0],Gamma[1],0,0], axis=0)
obj_l_K = smpl_wn.fit(Sigma[:,Ki[0],Ki[1],0,0], axis=0)
obj_l_X = smpl_wn.fit(Sigma[:,Xi[0],Xi[1],0,0], axis=0)
# We evaluate obj_l on the specified reduced matsubara frequencies
# using the uhat_l(iwn) basis functions
calculated_obj_wn_Gamma =  np.einsum("ij, ...i -> j...", mesh.IR_basis_set.basis_f.uhat(wn_array), obj_l_Gamma)
calculated_obj_wn_K =  np.einsum("ij, ...i -> j...", mesh.IR_basis_set.basis_f.uhat(wn_array), obj_l_K)
calculated_obj_wn_X =  np.einsum("ij, ...i -> j...", mesh.IR_basis_set.basis_f.uhat(wn_array), obj_l_X)

fs = 13
plt.plot(T * np.pi * wn_array,np.imag(calculated_obj_wn_Gamma),'x',label = r'$k = \Gamma$', markersize=4)
plt.plot(T * np.pi * wn_array,np.imag(calculated_obj_wn_K), 'x', label = r'$k = K$', markersize=4)
plt.plot(T * np.pi * wn_array,np.imag(calculated_obj_wn_X), 'x', label = r'$k = X$', markersize=4)
plt.ylabel(r'$Im \Sigma^{(2)11}_{\uparrow }(i \omega_n , k)$', fontsize = fs)
plt.legend(loc=1, fontsize = fs)
plt.xlim(0,T * np.pi * wn_array[-1])
plt.tick_params(axis='both', which='major', labelsize=fs*1.2)
plt.xlabel(r'$\omega_n$' , fontsize = fs)

plt.tight_layout()
plt.show()


# %%
#Re Sigma in momentum space for the first Matsubara frequency
#matrix indices are combined orbital and spin indices in the format
# ((1up 1up, 1up 1down, 1up 2up , 1up 2 down),(1down 1up ,...),...) 
# the momentum space is parameterized (convention used throughout) as
# k = k_1 G_1 + k_2 G_2 where G_1 and G_1 are the reciprocal lattice vectors for the KMH model

fig, axes = plt.subplots(nrows=tpsc.mesh.norb, ncols=tpsc.mesh.norb, figsize=(12,12))
axes = axes.flatten()

for i in range(mesh.norb):
    for j in range(mesh.norb):

        ind = i*mesh.norb + j        
        cont = axes[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.real(tpsc.g2out.selfEnergy[mesh.iw0_f,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axes[ind].set_xlabel('$k_1$')
        axes[ind].set_xlim([0,1])
        axes[ind].set_ylabel('$k_2$')
        axes[ind].set_ylim([0,1])
        #axes[ind].set_aspect('equal')
        fig.suptitle('Re $\Sigma(i\omega_0,k)$')
        fig.colorbar(cont, ax= axes[ind])

fig.tight_layout()
plt.show()


# %%
#Im Sigma in momentum space for the first Matsubara frequency
#matrix indices are combined orbital and spin indices in the format
# ((1up 1up, 1up 1down, 1up 2up , 1up 2 down),(1down 1up ,...),...) 

fig, axes = plt.subplots(nrows=tpsc.mesh.norb, ncols=tpsc.mesh.norb, figsize=(12,12))
axes = axes.flatten()

for i in range(mesh.norb):
    for j in range(mesh.norb):

        ind = i*mesh.norb + j        
        cont = axes[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.imag(tpsc.selfEnergy[mesh.iw0_f,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axes[ind].set_xlabel('$k_1$')
        axes[ind].set_xlim([0,1])
        axes[ind].set_ylabel('$k_2$')
        axes[ind].set_ylim([0,1])
        #axes[ind].set_aspect('equal')
        fig.suptitle('Im $\Sigma(i\omega_0,k)$')
        fig.colorbar(cont, ax= axes[ind])

fig.tight_layout()
plt.show()


# %%
#Re G in momentum space for the first Matsubara frequency
#matrix indices are combined orbital and spin indices in the format
# ((1up 1up, 1up 1down, 1up 2up , 1up 2 down),(1down 1up ,...),...) 

fig, axes = plt.subplots(nrows=tpsc.mesh.norb, ncols=tpsc.mesh.norb, figsize=(12,12))
axes = axes.flatten()

for i in range(mesh.norb):
    for j in range(mesh.norb):

        ind = i*mesh.norb + j        
        cont = axes[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.real(tpsc.g2out.gkiwn[mesh.iw0_f,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axes[ind].set_xlabel('$k_1$')
        axes[ind].set_xlim([0,1])
        axes[ind].set_ylabel('$k_2$')
        axes[ind].set_ylim([0,1])
        #axes[ind].set_aspect('equal')
        fig.suptitle('Re $G(i\omega_0,k)$')
        fig.colorbar(cont, ax= axes[ind])

plt.tight_layout()
plt.show()


# %%
#Im G in momentum space for the first Matsubara frequency
#matrix indices are combined orbital and spin indices in the format
# ((1up 1up, 1up 1down, 1up 2up , 1up 2 down),(1down 1up ,...),...) 

fig, axes = plt.subplots(nrows=tpsc.mesh.norb, ncols=tpsc.mesh.norb, figsize=(12,12))
axes = axes.flatten()

for i in range(mesh.norb):
    for j in range(mesh.norb):

        ind = i*mesh.norb + j        
        cont = axes[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.imag(tpsc.g2out.gkiwn[mesh.iw0_f,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axes[ind].set_xlabel('$k_1$')
        axes[ind].set_xlim([0,1])
        axes[ind].set_ylabel('$k_1$')
        axes[ind].set_ylim([0,1])
        #axes[ind].set_aspect('equal')
        fig.suptitle('Im $G(i\omega_0,k)$')
        fig.colorbar(cont, ax= axes[ind])
plt.tight_layout()
plt.show()


# %%
#plot chixx_afm in momenum space for zero frequency

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
axlst = np.array([axes]).flatten()

for i in [0]:
        ind = i       
        cont = axlst[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), observables.chixx_afm[mesh.iw0_b].real, 30)
        axlst[ind].set_xlabel('$q_1$')
        axlst[ind].set_xlim([0,1])
        axlst[ind].set_ylabel('$q_2$')
        axlst[ind].set_ylim([0,1])
        axlst[ind].set_aspect('equal')
        fig.colorbar(cont, ax= axlst[ind])
        axlst[ind].set_title('$\chi_{xx}^{afm}(iq_0,q)$, $\lambda_{SOC}$/t='+str(t2)+' T/t= '+str(T)+' U/t= '+str(U))

plt.tight_layout()
plt.show()


# %%
#plot Re chixx in momenum space for zero frequency
#matrix indices are orbital indices

fig, axes = plt.subplots(nrows=tpsc.mesh.norbsl, ncols=tpsc.mesh.norbsl, figsize=(12,12))
axlst = np.array([axes]).flatten()

for i in range(mesh.norbsl):
    for j in range(mesh.norbsl):

        ind = i*mesh.norbsl + j        
        cont = axlst[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.real(tpsc.chixx[mesh.iw0_b,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axlst[ind].set_xlabel('$q_1$')
        axlst[ind].set_xlim([0,1])
        axlst[ind].set_ylabel('$q_2$')
        axlst[ind].set_ylim([0,1])
        axlst[ind].set_aspect('equal')
        fig.suptitle('Re $\chi_{xx}(iq_0,q)$')
        fig.colorbar(cont, ax= axlst[ind])

plt.tight_layout()
plt.show()


# %%
#plot Im chixx in momenum space for zero frequency
#matrix indices are orbital indices

fig, axes = plt.subplots(nrows=tpsc.mesh.norbsl, ncols=tpsc.mesh.norbsl, figsize=(12,12))
axlst = np.array([axes]).flatten()

for i in range(mesh.norbsl):
    for j in range(mesh.norbsl):

        ind = i*mesh.norbsl + j        
        cont = axlst[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.imag(tpsc.chixx[mesh.iw0_b,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axlst[ind].set_xlabel('$q_1$')
        axlst[ind].set_xlim([0,1])
        axlst[ind].set_ylabel('$q_2$')
        axlst[ind].set_ylim([0,1])
        axlst[ind].set_aspect('equal')
        fig.suptitle('Im $\chi_{xx}(iq_0,q)$')
        fig.colorbar(cont, ax= axlst[ind])

plt.tight_layout()
plt.show()


# %%
#plot Re chicc in momenum space for zero frequency
#matrix indices are orbital indices

fig, axes = plt.subplots(nrows=tpsc.mesh.norbsl, ncols=tpsc.mesh.norbsl, figsize=(12,12))
axlst = np.array([axes]).flatten()

for i in range(mesh.norbsl):
    for j in range(mesh.norbsl):

        ind = i*mesh.norbsl + j        
        cont = axlst[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.real(tpsc.chicc[mesh.iw0_b,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axlst[ind].set_xlabel('$q_1$')
        axlst[ind].set_xlim([0,1])
        axlst[ind].set_ylabel('$q_2$')
        axlst[ind].set_ylim([0,1])
        axlst[ind].set_aspect('equal')
        fig.suptitle('Re $\chi_{cc}(iq_0,q)$')
        fig.colorbar(cont, ax= axlst[ind])

plt.tight_layout()
plt.show()


# %%
#plot Re chizc in momenum space for zero frequency
#matrix indices are orbital indices

fig, axes = plt.subplots(nrows=tpsc.mesh.norbsl, ncols=tpsc.mesh.norbsl, figsize=(12,12))
axlst = np.array([axes]).flatten()

for i in range(mesh.norbsl):
    for j in range(mesh.norbsl):

        ind = i*mesh.norbsl + j        

        cont = axlst[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.real(tpsc.chizc[mesh.iw0_b+1,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axlst[ind].set_xlabel('$k_1$')
        axlst[ind].set_xlim([0,1])
        axlst[ind].set_ylabel('$k_2$')
        axlst[ind].set_ylim([0,1])
        axlst[ind].set_aspect('equal')
        fig.suptitle('Re $\chi_{zc}(iq_1,q)$')
        fig.colorbar(cont, ax= axlst[ind])
plt.tight_layout()
plt.show()


# %%
#plot Im chizc in momenum space for zero frequency
#matrix indices are orbital indices

fig, axes = plt.subplots(nrows=tpsc.mesh.norbsl, ncols=tpsc.mesh.norbsl, figsize=(12,12))
axlst = np.array([axes]).flatten()

for i in range(mesh.norbsl):
    for j in range(mesh.norbsl):

        ind = i*mesh.norbsl + j        
        cont = axlst[ind].contourf(mesh.k1.reshape(nkx,nkx), mesh.k2.reshape(nkx,nkx), np.imag(tpsc.chizc[mesh.iw0_b+1,:,i,j].reshape(mesh.nk1,mesh.nk2)), 30)
        axlst[ind].set_xlabel('$k_1$')
        axlst[ind].set_xlim([0,1])
        axlst[ind].set_ylabel('$k_2$')
        axlst[ind].set_ylim([0,1])
        axlst[ind].set_aspect('equal')
        fig.suptitle('Im $\chi_{zc}(iq_1,q)$')
        fig.colorbar(cont, ax= axlst[ind])

plt.tight_layout()
plt.show()



# %%
