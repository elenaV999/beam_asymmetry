import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


##########################
#### CMB realizations ####
##########################

def read_Cls(fname, plot=False):
    '''
    Real theoretical Cls from file --> Return: (ls, Cls)
    Args: fname= file with theoretical Cls  --- PLA file contains Dl=l*(l+1)/2pi Cl
    Return: 
    '''
    #add a test to see if they are Cls or Dls: check that Dl[15] has the same order of magnitude (10^3) as Cl[2]
    #check if multipoles start from l=2 or l=0

    #read Cls
    l2, Dl = np.genfromtxt(fname, dtype=[('l', int), ('Dl', float)], comments="#", usecols=(0,1), unpack=True) #CLASS output: normalized witrh l(l+1)/2pi 
    normCl2=l2*(l2+1)/(2*np.pi)
    Cl=Dl/normCl2
    lmax=l2[-1]
    print('lmax = ', lmax)

    #add monopole and dipole
    if l2[0]: #should be false if l2[0]=0, True if l2[0]=1,2 
        l = np.concatenate(([0, 1], l2))
        Cl = np.concatenate(([0, 0], Cl)) 
        normCl = np.concatenate(([1, 1/np.pi ], normCl2))
    
    if plot==True: 
        fig=plt.figure(figsize=(7,5))
        plt.plot(l2, Dl, label=r'$D_\ell=\frac{\ell(\ell+1)}{2\pi}\,\,C_\ell$')
        plt.plot(l, Cl, label=r'$C_\ell$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\ell$', fontsize=20)
        plt.ylabel(r'spectra $[\mu K^2]$', fontsize=16)
        plt.legend(fontsize=15, loc='lower center')
        plt.show() 

    return l, Cl


def get_tau_arr(tau, lmax, plot=False):
    ll=np.arange(lmax+1)

    #original damping - not good
    lcut=30
    y = np.linspace(0, 3, lcut-1)
    yy = np.tanh(y)
    damping_factor=np.concatenate( (yy, np.ones(lmax-lcut+2)) )
    tau_arr=tau*damping_factor

    #new damping (with hyperbolic tangent)
    lcut2=8
    damping_orders=3
    width=lcut2/ 1.2 #increase number for steeper damping
    transition = 0.5 * (1 + np.tanh((ll - lcut2) / width))
    damping_factor2 = 10**(-damping_orders * (1 - transition))
    tau_arr2=tau*damping_factor2


    if plot==True: 
        plt.figure(figsize=(8,3.5))
        plt.loglog(ll, tau_arr, color='k', label='damping 1')
        plt.loglog(ll, tau_arr2, color='r', label='damping 2')
        plt.xlabel(r'$\ell$', fontsize=20)
        plt.ylabel(r'$\tau_{X_i}$', fontsize=20)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.show()
        
    return  tau_arr


def generate_realizations(l, Cl_th, tau_xi, plot=False):
    '''
    Args:
    - theoretical (l,Cls) starting from l=0
    - tau_xi = value of resonant scattering optical depth at plateau (float) 
    Return: 2 maps
    '''
    lmax=int(l[-1])
    print('lmax = ', lmax)
    if l[0]!=0: 
        print('ERROR: Cl-array does not start at l=0') 

    #generate tau array
    tau_arr = get_tau_arr(tau_xi, lmax)

    #generate maps
    alm = hp.synalm(Cl_th, lmax=lmax) #realization of theoretical Cls - CLEAN MAP
    alm_xi=alm-hp.almxfl(alm, tau_arr) #blur the alm realization - RESONANT SCATTERING MAP

    nside_raw = (lmax + 1) / 3
    nside = 2 ** int(np.floor(np.log2(nside_raw)))
    print('nside = ', nside)

    map_cmb = hp.alm2map(alm, nside=nside, lmax=lmax)
    map_xi=hp.alm2map(alm_xi, nside=nside, lmax=lmax)


    #plot Cls
    if plot==True: 
        Cl_xi_th=Cl_th-2*tau_arr*Cl_th
        Cls_map_cmb =hp.sphtfunc.anafast(map_cmb, lmax=lmax) 
        Cls_map_xi = hp.sphtfunc.anafast(map_xi,lmax=lmax)

        opt=1  # 1) Cls  2) Dls 
        if opt==1: 
            normCl=np.ones(len(Cl_th))
            ylabel=r'$C_\ell\,[\mu K^2]$'
        elif opt==2: 
            normCl=l*(l+1)/(2*np.pi)  
            ylabel=r'$\frac{\ell(\ell+1)}{2\pi}\,\,C_\ell\,[\mu K^2]$'
        
        fig=plt.figure(figsize=(7,5))
        plt.plot(l, Cl_th, label=r'CMB th.', color='darkblue')
        plt.plot(l, Cl_xi_th, label=r'CMB$+X_i$ th.', color='tab:red', ls='--')
        plt.plot(l, Cls_map_cmb, label=r'CMB map', color='dodgerblue', alpha=0.4)
        plt.plot(l, Cls_map_xi, label=r'CMB$+X_i$ map', color='red',  alpha=0.4, ls='--')
        plt.plot(l, Cl_th-Cl_xi_th, label=r'Cls diff th', color='blue')
        plt.plot(l, Cls_map_cmb-Cls_map_xi, label=r'Cla diff maps', color='red', ls='--')
        #plt.ylim(1e-5, 1e4)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\ell$', fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.legend(fontsize=15)
        plt.show() 

    #plot maps
    if plot==True: 
        hp.mollview(map_xi, title="Blurred CMB map" , norm='hist', unit=r'$\mu K$')


    return map_cmb, map_xi














