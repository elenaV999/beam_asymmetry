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
    print('lcut_tau = ', lcut)
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
        
    return  tau_arr, tau_arr2


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
        Cls_map_cmb =hp.sphtfunc.anafast(map_cmb, lmax=lmax, use_pixel_weights=True) 
        Cls_map_xi = hp.sphtfunc.anafast(map_xi,lmax=lmax, use_pixel_weights=True)

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


##########################
#### NOISE STATISTICS ####
##########################

# convert Temperature units from CMB to Rayleigh-Jeans: T_RJ = factor*TCMB
def Tunits(nu_in):
    """
    arg: frequency in GHz
    Returns: conversion factor (dimensionless)
    """
    c = 299792458.      # speed of light [m/s]
    k = 1.3806488e-23   # Boltzmann constant [J/K]
    h = 6.62606957e-34  # Planck constant [J⋅s]
    T_cmb = 2.725       # CMB temperature [K]

    nu = nu_in * 1e9    # convert GHz to Hz
    x = h*nu/(k*T_cmb)  # dimensionless frequency
    
    factor=x**2*np.exp(x)/(np.exp(x) - 1.)**2

    return factor


#convert noise sigma: [μK⋅deg] in RJ T-units --> [μK⋅pix] in CMB T-units (used to build noise maps)
def convert_noise_sigma(nu_in, nside, print_opt=False):
    '''
    args: 
    - nu_in: frequency in GHz (float or array)
    - nside: of noise map to be generated
    return: 
    - sigma noise [μK⋅pix] in CMB T-units  --> use to generate noise maps or alms 
    - expected amplitude of noise Cls 
    '''

    #Planck noise: from table 2 of 1907.11642, in μK⋅deg and RJ T-units
    sigma_dict = {
        30:  2.45,
        44:  2.57,
        70:  3.08,
        100: 1.00,
        143: 0.333,
        217: 0.261,
        353: 0.198,
        545: 0.0855,
        857: 0.0319
    }   
    nu_in = np.atleast_1d(nu_in)

    nu_arr = np.atleast_1d(nu_in)
    sigma_muKdeg = np.array([sigma_dict[int(nu)] for nu in nu_arr]) #in RJ units
    tunits_factor = np.array([Tunits(nu) for nu in nu_arr]) #conversion factor
    sigma_CMBdeg=sigma_muKdeg/tunits_factor #in CMB units 

    pixArea_deg=hp.nside2pixarea(nside, degrees=True) #SCALE noise to pixel size (CMB T-units)
    sigma_CMB_pix=sigma_CMBdeg/np.sqrt(pixArea_deg)

    pixArea_rad=hp.nside2pixarea(nside, degrees=False) 
    NCls_levels=pixArea_rad*sigma_CMB_pix**2

    if print_opt==True: 
        print('f[GHz]\tσ_RJ[μK⋅deg]\tσ_CMB[μK⋅deg]\tσ_CMB[μK⋅pix]\t noiseCls level[μK²]')
        for ii in range(len(nu_in)):
            print(f'{nu_arr[ii]:.0f} \t{sigma_muKdeg[ii]:.3f} \t\t {sigma_CMBdeg[ii]:.3f} \t\t{sigma_CMB_pix[ii]:.3f}\t\t{NCls_levels[ii]:.3e}')

    return sigma_CMB_pix, NCls_levels


#generate white noise alms realizations: 
def get_noise_alm(sigma_alms, lmax, seed=None):
    """
    sigma_alms: sqrt(Cls amplitude) or sigma_pix*sqrt(pix_area)  -- in T-CMB units 
    lmax :  maximum multipole of alms to produce 
    seed : (int, optional) Random seed for reproducibility (same seed will give the same alms)
    Returns: alm : complex128 ndarray, shape (nalm,) Noise alms in HEALPix ordering convention
    """
    rng = np.random.default_rng(seed)


    nalm = hp.Alm.getsize(lmax)
    alm  = np.empty(nalm, dtype=np.complex128)

    # m = 0 modes: real, full variance
    ells_m0  = np.arange(0, lmax + 1)
    idx_m0   = hp.Alm.getidx(lmax, ells_m0, 0)
    alm[idx_m0] = sigma_alms * rng.standard_normal(lmax + 1)

    # m > 0 modes: complex, variance split equally between real and imag
    for m in range(1, lmax + 1):
        ells   = np.arange(m, lmax + 1)
        idx    = hp.Alm.getidx(lmax, ells, m)
        n      = len(ells)
        alm[idx] = sigma_alms / np.sqrt(2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))  #every call of rng draws from the same sequence moving foreward, so variables are all independent 

    return alm
### make 2 alms of different sized the same: 



##########################
#### MATCHED FILTER   ####
##########################


# tau from matched filtering from N noise realizations, with 3 different templates.  
def matched_filter_noise_ell(map0, map1, map_beam0, map_beam1, map_cmb, sigmaN0, sigmaN1, lmin_tauA, lmax_tauA, N):  ## CASE: template = map0
    '''
    Args: 
    - map0, map1: maps at 2 frequencies --> to compute difference map (DATA)
    - map_beam0, map_beam1: beam maps of the 2 maps --> use to deconvolve noisy maps 
    - sigmaN0, sigmaN1: noise sigma in CMB T-units (μK⋅pix) of the 2 maps ---> use to generate noise alms
    - lmin_tauA, lmax_tauA: array of multipole ranges to use for matched filter (dim=nº of templates)
    - N: number of noise realizations to generate
    - map_cmb: original CMB realization to use as template
    Return: tau, sigma tau (list of shape: 3,N )
    '''
    #print(f'(lmin,lmax) = ({lmin_tau},{lmax_tau})')
    
    LMAX=np.max(lmax_tauA)  #maximum lmax of the 3 templates

    #map alms
    alm0=hp.map2alm(map0, lmax=LMAX, use_pixel_weights=True)  #lmax=of the output alm
    alm1=hp.map2alm(map1, lmax=LMAX, use_pixel_weights=True)
    alm_template_cmb=hp.map2alm(map_cmb, lmax=LMAX, use_pixel_weights=True)  
    alm_template2=alm_template_cmb


    #beam Cls 
    Cls_beam0=hp.anafast(map_beam0, lmax=LMAX, use_pixel_weights=True)*4*np.pi  # symmetrized beams, up to lmax
    Cls_beam1=hp.anafast(map_beam1, lmax=LMAX, use_pixel_weights=True)*4*np.pi 
    bl0_inv=1/np.sqrt(Cls_beam0) ### NOTA: CLs_beam need to be cut to lmin_tau, lmax_tau
    bl1_inv=1/np.sqrt(Cls_beam1)

    check_beam0=np.all(np.abs(Cls_beam0[:3] - 1) < 1e-2) #CHECK: beam norm
    check_beam1=np.all(np.abs(Cls_beam1[:3] - 1) < 1e-2)   
    if not check_beam0 or not check_beam1:
        print("ERROR: beam normalizations is off (first multiples differ > 1e-2 from 1")

    lmax_alm = hp.Alm.getlmax(len(alm0))  #CHECK: if alms and bl have the same lmax
    lmax_bl  = len(bl0_inv) - 1  # bl is indexed 0..lmax
    assert lmax_alm == lmax_bl, f"ERROR: lmax mismatch: alms have lmax={lmax_alm}, bl has lmax={lmax_bl}"

    #noise levels 
    nside=hp.get_nside(map0)
    pixArea_rad=hp.nside2pixarea(nside, degrees=False)
    sigmaN0_alms=np.sqrt(pixArea_rad)*sigmaN0
    sigmaN1_alms=np.sqrt(pixArea_rad)*sigmaN1

    Cl_noise0=(pixArea_rad*sigmaN0**2/Cls_beam0) #th. noise Cls, needed for subtraction
    Cl_noise1= -(pixArea_rad*sigmaN1**2/Cls_beam1)#th. noise Cls, needed for subtraction
    Clnoise_list=[Cl_noise0, Cl_noise1, 0.]  #noise Cls for subtraction (only for cross Cls with template 0 or 1)

    #set seed to generate independent noise realizations
    MS0 = np.random.SeedSequence(88)   # master seed 
    MS1 = np.random.SeedSequence(42)   
    s0 = MS0.spawn(N)  #sequence of N seeds (same when master seed is the same)
    s1 = MS1.spawn(N)  

    #result arrays 
    l=np.arange(0, LMAX+1)

    tauE0=np.empty(N, dtype=float)
    tauE1=np.empty(N, dtype=float)
    tauE2=np.empty(N, dtype=float)
    sigma_tauE0=np.empty(N, dtype=float)
    sigma_tauE1=np.empty(N, dtype=float)
    sigma_tauE2=np.empty(N, dtype=float)

    #LOOP OVER NOISE REALIZATIONS 
    nom_l_all = np.zeros((3,N, LMAX+1)) #arrays to store results for all noise realizations: [i][j]-->i=noise realizations, j=ell 
    denom_l_all = np.zeros((3,N, LMAX+1)) #each row is a different noise realization
    for i in range(N):
        almN0=get_noise_alm(sigmaN0_alms, LMAX, seed=s0[i]) #generate noise
        almN1=get_noise_alm(sigmaN1_alms, LMAX, seed=s1[i])

        alm0_n=alm0+almN0  #add noise to maps
        alm1_n=alm1+almN1

        alm0_n_dec=hp.almxfl(alm0_n, bl0_inv)  #deconvolve 
        alm1_n_dec=hp.almxfl(alm1_n, bl1_inv)   
        alm_data=alm0_n_dec-alm1_n_dec #get data map

        Cl_cov=hp.alm2cl(alm_data) #Cls of covariance = Cls of data (APPROX.)

        alm_templates=[alm0_n_dec, alm1_n_dec, alm_template_cmb] 
        
        ### DIFFERENT TEMPLATES: 
        w=(2*l+1)/Cl_cov
        for k in range(3):
            Cl_crossT=hp.alm2cl(alm_data, alm_templates[k])-Clnoise_list[k]   #cross Cls --> NOM
            Cl_T=hp.alm2cl(alm_templates[k])  #template Cls --> DENOM

            nom_l_all[k, i, :]=w*Cl_crossT  #nom_l_all[ii, :] corresponds to the i-th noise realization
            denom_l_all[k, i, :]=w*Cl_T


    #LOOP OVER TEMPLATES:
    tauE3=np.zeros((3,N))
    sigma_tauE3=np.zeros((3,N))
    for k in range(3):
        nom=np.sum(nom_l_all[k, :, lmin_tauA[k]:lmax_tauA[k]+1], axis=1)
        denom=np.sum(denom_l_all[k, :, lmin_tauA[k]:lmax_tauA[k]+1], axis=1)
        tauE3[k,:]=nom/denom*1e3  
        sigma_tauE3[k,:]=np.sqrt(1/denom)*1e3


    return tauE3, sigma_tauE3


# tau from matched filtering without noise, with 3 different templates.  
def matched_filter_noiseless_ell(map0, map1, map_beam0, map_beam1, map_cmb,lmin_tauA, lmax_tauA):  
    '''
    Args: 
    - map0, map1: maps at 2 frequencies --> to compute difference map (DATA)
    - map_beam0, map_beam1: beam maps of the 2 maps --> use to deconvolve noisy maps 
    - map_cmb: original CMB realization to use as template
    Return: tau, sigma tau (array of len=3)
    '''

    LMAX=np.max(lmax_tauA)
    LMIN=np.min(lmin_tauA)

    #map alms
    alm0=hp.map2alm(map0, lmax=LMAX, use_pixel_weights=True) 
    alm1=hp.map2alm(map1, lmax=LMAX, use_pixel_weights=True)
    alm_template_cmb=hp.map2alm(map_cmb, lmax=LMAX, use_pixel_weights=True)  
 
    #beam Cls 
    Cls_beam0=hp.anafast(map_beam0, lmax=LMAX, use_pixel_weights=True)*4*np.pi  # symmetrized beams, up to lmax
    Cls_beam1=hp.anafast(map_beam1, lmax=LMAX, use_pixel_weights=True)*4*np.pi 
    bl0_inv=1/np.sqrt(Cls_beam0) ### NOTA: CLs_beam need to be cut to lmin_tau, lmax_tau
    bl1_inv=1/np.sqrt(Cls_beam1)

    check_beam0=np.all(np.abs(Cls_beam0[:3] - 1) < 1e-2) #CHECK: beam norm
    check_beam1=np.all(np.abs(Cls_beam1[:3] - 1) < 1e-2)   
    if not check_beam0 or not check_beam1:
        print("ERROR: beam normalizations is off (first multiples differ > 1e-2 from 1")

    lmax_alm = hp.Alm.getlmax(len(alm0))  #CHECK: if alms and bl have the same lmax
    lmax_bl  = len(bl0_inv) - 1  # bl is indexed 0..lmax
    assert lmax_alm == lmax_bl, f"ERROR: lmax mismatch: alms have lmax={lmax_alm}, bl has lmax={lmax_bl}"
    print('lmax=', lmax_bl)

    ell=np.arange(0, LMAX+1)
    w=2*ell+1

    alm0_dec=hp.almxfl(alm0, bl0_inv)  #deconvolve 
    alm1_dec=hp.almxfl(alm1, bl1_inv)   
    alm_data=alm0_dec-alm1_dec #get data map

    #templates
    alm_templates=[alm0_dec, alm1_dec, alm_template_cmb]  

    #compute Cls: dim=LMAX (automatic)
    Cl_cov=hp.alm2cl(alm_data)   #Cls of covariance = Cls of data (APPROX.)

    #matched filter quantities
    l=np.arange(0, LMAX+1)
    w=(2*l+1)/Cl_cov

    tauE_arr=np.zeros(3)
    sigma_tauE_arr=np.zeros(3)

    for k in range(3):
        Cl_crossT=hp.alm2cl(alm_data, alm_templates[k])  #cross Cls --> NOM
        Cl_T=hp.alm2cl(alm_templates[k])  #template Cls --> DENOM

        nom_l=w*Cl_crossT  # this is from l=0 to lmax --> slice later
        denom_l=w*Cl_T
        nom=np.sum(nom_l[lmin_tauA[k]:lmax_tauA[k]+1])
        denom=np.sum(denom_l[lmin_tauA[k]:lmax_tauA[k]+1])
        tauE_arr[k]=nom/denom*1e3
        sigma_tauE_arr[k]=np.sqrt(1/denom)*1e3

    return tauE_arr, sigma_tauE_arr


# tau from matched filtering from N noise realizations, with 3 different templates.  
def matched_filter_noise(map0, map1, map_beam0, map_beam1, map_cmb, sigmaN0, sigmaN1, lmin_tau, lmax_tau, N):  ## CASE: template = map0
    '''
    Args: 
    - map0, map1: maps at 2 frequencies --> to compute difference map (DATA)
    - map_beam0, map_beam1: beam maps of the 2 maps --> use to deconvolve noisy maps 
    - sigmaN0, sigmaN1: noise sigma in CMB T-units (μK⋅pix) of the 2 maps ---> use to generate noise alms
    - lmin_tau, lmax_tau: multipole range to use for matched filter
    - N: number of noise realizations to generate
    - map_cmb: original CMB realization to use as template
    Return: tau, sigma tau (list of shape: 3,N )
    '''
    #print(f'(lmin,lmax) = ({lmin_tau},{lmax_tau})')
    
    
    #map alms
    alm0=hp.map2alm(map0, lmax=lmax_tau, use_pixel_weights=True)  #lmax=of the output alm
    alm1=hp.map2alm(map1, lmax=lmax_tau, use_pixel_weights=True)
    alm_template_cmb=hp.map2alm(map_cmb, lmax=lmax_tau, use_pixel_weights=True)  
    alm_template2=alm_template_cmb
    Cl_T2=hp.alm2cl(alm_template2)[lmin_tau:]   #lmax=of the input alm, lmax_out= of the output Cls

    #beam Cls 
    Cls_beam0=hp.anafast(map_beam0, lmax=lmax_tau, use_pixel_weights=True)*4*np.pi  # symmetrized beams, up to lmax
    Cls_beam1=hp.anafast(map_beam1, lmax=lmax_tau, use_pixel_weights=True)*4*np.pi 
    bl0_inv=1/np.sqrt(Cls_beam0) ### NOTA: CLs_beam need to be cut to lmin_tau, lmax_tau
    bl1_inv=1/np.sqrt(Cls_beam1)

    check_beam0=np.all(np.abs(Cls_beam0[:3] - 1) < 1e-2) #CHECK: beam norm
    check_beam1=np.all(np.abs(Cls_beam1[:3] - 1) < 1e-2)   
    if not check_beam0 or not check_beam1:
        print("ERROR: beam normalizations is off (first multiples differ > 1e-2 from 1")

    lmax_alm = hp.Alm.getlmax(len(alm0))  #CHECK: if alms and bl have the same lmax
    lmax_bl  = len(bl0_inv) - 1  # bl is indexed 0..lmax
    assert lmax_alm == lmax_bl, f"ERROR: lmax mismatch: alms have lmax={lmax_alm}, bl has lmax={lmax_bl}"

    #noise levels 
    nside=hp.get_nside(map0)
    pixArea_rad=hp.nside2pixarea(nside, degrees=False)
    sigmaN0_alms=np.sqrt(pixArea_rad)*sigmaN0
    sigmaN1_alms=np.sqrt(pixArea_rad)*sigmaN1

    Cl_noise0=(pixArea_rad*sigmaN0**2/Cls_beam0)[lmin_tau:] #th. noise Cls, needed for subtraction
    Cl_noise1= -(pixArea_rad*sigmaN1**2/Cls_beam1)[lmin_tau:] #th. noise Cls, needed for subtraction

    #set seed to generate independent noise realizations
    MS0 = np.random.SeedSequence(88)   # master seed 
    MS1 = np.random.SeedSequence(42)   
    s0 = MS0.spawn(N)  #sequence of N seeds (same when master seed is the same)
    s1 = MS1.spawn(N)  

    #result arrays 
    ell=np.arange(lmin_tau, lmax_tau+1)
    w=2*ell+1
    tauE0=np.empty(N, dtype=float)
    tauE1=np.empty(N, dtype=float)
    tauE2=np.empty(N, dtype=float)
    sigma_tauE0=np.empty(N, dtype=float)
    sigma_tauE1=np.empty(N, dtype=float)
    sigma_tauE2=np.empty(N, dtype=float)

    #LOOP OVER NOISE REALIZATIONS 
    for i in range(N):
        almN0=get_noise_alm(sigmaN0_alms, lmax_tau, seed=s0[i]) #generate noise
        almN1=get_noise_alm(sigmaN1_alms, lmax_tau, seed=s1[i])

        alm0_n=alm0+almN0  #add noise to maps
        alm1_n=alm1+almN1

        alm0_n_dec=hp.almxfl(alm0_n, bl0_inv)  #deconvolve 
        alm1_n_dec=hp.almxfl(alm1_n, bl1_inv)   
        alm_data=alm0_n_dec-alm1_n_dec #get data map

        Cl_cov=hp.alm2cl(alm_data)[lmin_tau:]   #Cls of covariance = Cls of data (APPROX.)

        ### DIFFERENT TEMPLATES: 
        alm_template0=alm0_n_dec  #templates
        alm_template1=alm1_n_dec
        Cl_crossT0=hp.alm2cl(alm_data, alm_template0)[lmin_tau:]-Cl_noise0     #noise subtraction
        Cl_crossT1=hp.alm2cl(alm_data, alm_template1)[lmin_tau:]-Cl_noise1    #noise subtraction
        Cl_crossT2=hp.alm2cl(alm_data, alm_template2)[lmin_tau:]    #NO noise subtraction!! since noise of diff map is not supposed to correlate

        Cl_T0=hp.alm2cl(alm_template0)[lmin_tau:]  #template Cls
        Cl_T1=hp.alm2cl(alm_template1)[lmin_tau:]  

        #MATCHED FILTER
        nom0=np.sum(w*Cl_crossT0/Cl_cov)  #with template 1 (map nu0)
        denom0=np.sum(w*Cl_T0/Cl_cov)
        tauE0[i]=nom0/denom0
        sigma_tauE0[i]=np.sqrt(1/denom0)


        nom1=np.sum(w*Cl_crossT1/Cl_cov)  #with template 2 (map nu1)
        denom1=np.sum(w*Cl_T1/Cl_cov)
        tauE1[i]=nom1/denom1
        sigma_tauE1[i]=np.sqrt(1/denom1)

        nom2=np.sum(w*Cl_crossT2/Cl_cov)  #with template 3 (CMB)
        denom2=np.sum(w*Cl_T2/Cl_cov)
        tauE2[i]=nom2/denom2
        sigma_tauE2[i]=np.sqrt(1/denom2)

    tauE_list=[tauE0*1e3, tauE1*1e3, tauE2*1e3]
    sigma_tauE_list=[sigma_tauE0*1e3, sigma_tauE1*1e3, sigma_tauE2*1e3]

    return tauE_list, sigma_tauE_list

# tau from matched filtering without noise, with 3 different templates.  
def matched_filter_noiseless(map0, map1, map_beam0, map_beam1, map_cmb, lmin_tau, lmax_tau):  
    '''
    Args: 
    - map0, map1: maps at 2 frequencies --> to compute difference map (DATA)
    - map_beam0, map_beam1: beam maps of the 2 maps --> use to deconvolve noisy maps 
    - map_cmb: original CMB realization to use as template
    Return: tau, sigma tau (array of len=3)
    '''

    #map alms
    alm0=hp.map2alm(map0, lmax=lmax_tau, use_pixel_weights=True) 
    alm1=hp.map2alm(map1, lmax=lmax_tau, use_pixel_weights=True)
    alm_template_cmb=hp.map2alm(map_cmb, lmax=lmax_tau, use_pixel_weights=True)  
 
    #beam Cls 
    Cls_beam0=hp.anafast(map_beam0, lmax=lmax_tau, use_pixel_weights=True)*4*np.pi  # symmetrized beams, up to lmax
    Cls_beam1=hp.anafast(map_beam1, lmax=lmax_tau, use_pixel_weights=True)*4*np.pi 
    bl0_inv=1/np.sqrt(Cls_beam0) ### NOTA: CLs_beam need to be cut to lmin_tau, lmax_tau
    bl1_inv=1/np.sqrt(Cls_beam1)

    check_beam0=np.all(np.abs(Cls_beam0[:3] - 1) < 1e-2) #CHECK: beam norm
    check_beam1=np.all(np.abs(Cls_beam1[:3] - 1) < 1e-2)   
    if not check_beam0 or not check_beam1:
        print("ERROR: beam normalizations is off (first multiples differ > 1e-2 from 1")

    lmax_alm = hp.Alm.getlmax(len(alm0))  #CHECK: if alms and bl have the same lmax
    lmax_bl  = len(bl0_inv) - 1  # bl is indexed 0..lmax
    assert lmax_alm == lmax_bl, f"ERROR: lmax mismatch: alms have lmax={lmax_alm}, bl has lmax={lmax_bl}"
    print('lmax=', lmax_bl)

    ell=np.arange(lmin_tau, lmax_tau+1)
    w=2*ell+1

    alm0_dec=hp.almxfl(alm0, bl0_inv)  #deconvolve 
    alm1_dec=hp.almxfl(alm1, bl1_inv)   
    alm_data=alm0_dec-alm1_dec #get data map

    Cl_cov=hp.alm2cl(alm_data)[lmin_tau:]   #Cls of covariance = Cls of data (APPROX.)

    alm_template0=alm0_dec  #templates
    alm_template1=alm1_dec
    alm_template2=alm_template_cmb

    Cl_crossT0=hp.alm2cl(alm_data, alm_template0)[lmin_tau:]     #noise subtraction
    Cl_crossT1=hp.alm2cl(alm_data, alm_template1)[lmin_tau:]    #noise subtraction
    Cl_crossT2=hp.alm2cl(alm_data, alm_template2)[lmin_tau:]    #noise subtraction

    Cl_T0=hp.alm2cl(alm_template0)[lmin_tau:]  #template Cls
    Cl_T1=hp.alm2cl(alm_template1)[lmin_tau:]  
    Cl_T2=hp.alm2cl(alm_template2)[lmin_tau:]  

    #MATCHED FILTER
    nom0=np.sum(w*Cl_crossT0/Cl_cov)  #with template 1
    denom0=np.sum(w*Cl_T0/Cl_cov)
    tauE0=nom0/denom0
    sigma_tauE0=np.sqrt(1/denom0)

    nom1=np.sum(w*Cl_crossT1/Cl_cov)  #with template 2
    denom1=np.sum(w*Cl_T1/Cl_cov)
    tauE1=nom1/denom1
    sigma_tauE1=np.sqrt(1/denom1)

    nom2=np.sum(w*Cl_crossT2/Cl_cov)  #with template 3
    denom2=np.sum(w*Cl_T2/Cl_cov)
    tauE2=nom2/denom2
    sigma_tauE2=np.sqrt(1/denom2)

    tauE_arr=np.array([tauE0*1e3, tauE1*1e3, tauE2*1e3])
    sigma_tauE_arr=np.array([sigma_tauE0*1e3, sigma_tauE1*1e3, sigma_tauE2*1e3])

    return tauE_arr, sigma_tauE_arr



# tau from matched filtering from N noise realizations, with 3 different templates.  
def matched_filter_statN_old(map0, map1, Cls_beam0, Cls_beam1, sigmaN0, sigmaN1, map_cmb, lmin_tau, lmax_tau, N):  ## CASE: template = map0
    print('func old')
    '''
    Args: 
    - map0, map1: maps at 2 frequencies --> to compute difference map (DATA)
    - Cls_beam0, Cls_beam1: beam Cls of the 2 maps --> use to deconvolve noisy maps 
    - sigmaN0, sigmaN1: noise sigma in CMB T-units (μK⋅pix) of the 2 maps ---> use to generate noise alms
    - lmin_tau, lmax_tau: multipole range to use for matched filter
    - N: number of noise realizations to generate
    - map_cmb: original CMB realization to use as template
    Return: tau, sigma tau (list of shape: 3,N )
    '''
    l=np.arange(lmin_tau, lmax_tau+1)

    alm0=hp.map2alm(map0, lmax=lmax_tau, use_pixel_weights=True) #alms of maps to get difference map
    alm1=hp.map2alm(map1, lmax=lmax_tau, use_pixel_weights=True)
    alm_template_cmb=hp.map2alm(map_cmb, lmax=lmax_tau, use_pixel_weights=True)  #template from original CMB map (not convolved, no noise)

    bl0_inv=1/np.sqrt(Cls_beam0[:lmax_tau+1]) ### NOTA: CLs_beam need to be cut to lmin_tau, lmax_tau
    bl1_inv=1/np.sqrt(Cls_beam1[:lmax_tau+1])

    lmax_alm = hp.Alm.getlmax(len(alm0))  ### check if alms and bl have the same lmax
    lmax_bl  = len(bl0_inv) - 1  # bl is indexed 0..lmax
    assert lmax_alm == lmax_bl, f"ERROR: lmax mismatch: alms have lmax={lmax_alm}, bl has lmax={lmax_bl}"

    #noise for cross Cls for different templates (deconvolved noise)
    nside=hp.get_nside(map0)
    pixArea_rad=hp.nside2pixarea(nside, degrees=False)

    sigmaN0_alms=np.sqrt(pixArea_rad)*sigmaN0
    sigmaN1_alms=np.sqrt(pixArea_rad)*sigmaN1

    Cl_noise0=pixArea_rad*sigmaN0**2/Cls_beam0[:lmax_tau+1]
    Cl_noise1=-pixArea_rad*sigmaN1**2/Cls_beam1[:lmax_tau+1]


    MS0 = np.random.SeedSequence(88)   # master seed
    s0 = MS0.spawn(N)  

    MS1 = np.random.SeedSequence(42)   # master seed
    s1 = MS1.spawn(N)  

    tauE0=np.empty(N, dtype=float)
    tauE1=np.empty(N, dtype=float)
    tauE2=np.empty(N, dtype=float)
    sigma_tauE0=np.empty(N, dtype=float)
    sigma_tauE1=np.empty(N, dtype=float)
    sigma_tauE2=np.empty(N, dtype=float)

    for i in range(N):
        almN0=get_noise_alm(sigmaN0_alms, lmax_tau, seed=s0[i]) #generate noise
        almN1=get_noise_alm(sigmaN1_alms, lmax_tau, seed=s1[i])

        alm0_n=alm0+almN0  #add noise to maps
        alm1_n=alm1+almN1

        alm0_n_dec=hp.almxfl(alm0_n, bl0_inv)  #deconvolve 
        alm1_n_dec=hp.almxfl(alm1_n, bl1_inv)   
        alm_data=alm0_n_dec-alm1_n_dec #get data map

        Cl_cov=hp.alm2cl(alm_data, lmax=lmax_tau)  #Cls of covariance = Cls of data (APPROX.)
        Cl_cov_cut=Cl_cov[lmin_tau:] 

        ### DIFFERENT TEMPLATES: 
        alm_template0=alm0_n_dec  #templates
        alm_template1=alm1_n_dec
        alm_template2=alm_template_cmb
        Cl_crossT0=hp.alm2cl(alm_data, alm_template0, lmax=lmax_tau)-Cl_noise0     #noise subtraction
        Cl_crossT1=hp.alm2cl(alm_data, alm_template1, lmax=lmax_tau)-Cl_noise1    #noise subtraction
        Cl_crossT2=hp.alm2cl(alm_data, alm_template2, lmax=lmax_tau)    #NO noise subtraction!! since noise of diff map is not supposed to correlate

        Cl_T0=hp.alm2cl(alm_template0, lmax=lmax_tau)  #template Cls
        Cl_T1=hp.alm2cl(alm_template1, lmax=lmax_tau)  
        Cl_T2=hp.alm2cl(alm_template2, lmax=lmax_tau)  

        #MATCHED FILTER
        nom0=np.sum((2*l+1)*Cl_crossT0[lmin_tau:]/Cl_cov_cut)  #with template 1 (map nu0)
        denom0=np.sum((2*l+1)*Cl_T0[lmin_tau:]/Cl_cov_cut)
        tauE0[i]=nom0/denom0
        sigma_tauE0[i]=np.sqrt(1/denom0)


        nom1=np.sum((2*l+1)*Cl_crossT1[lmin_tau:]/Cl_cov_cut)  #with template 2 (map nu1)
        denom1=np.sum((2*l+1)*Cl_T1[lmin_tau:]/Cl_cov_cut)
        tauE1[i]=nom1/denom1
        sigma_tauE1[i]=np.sqrt(1/denom1)

        nom2=np.sum((2*l+1)*Cl_crossT2[lmin_tau:]/Cl_cov_cut)  #with template 3 (CMB)
        denom2=np.sum((2*l+1)*Cl_T2[lmin_tau:]/Cl_cov_cut)
        tauE2[i]=nom2/denom2
        sigma_tauE2[i]=np.sqrt(1/denom2)

    tauE_list=[tauE0*1e3, tauE1*1e3, tauE2*1e3]
    sigma_tauE_list=[sigma_tauE0*1e3, sigma_tauE1*1e3, sigma_tauE2*1e3]

    return tauE_list, sigma_tauE_list


# tau from matched filtering without noise, with 3 different templates.  
def matched_filter_noiseless_old(map0, map1, Cls_beam0, Cls_beam1, map_cmb, lmin_tau, lmax_tau):  
    '''
    Args: 
    - map0, map1: maps at 2 frequencies --> to compute difference map (DATA)
    - Cls_beam0, Cls_beam1: beam Cls of the 2 maps --> use to deconvolve noisy maps 
    - sigmaN0, sigmaN1: noise sigma in CMB T-units (μK⋅pix) of the 2 maps ---> use to generate noise alms
    - lmin_tau, lmax_tau: multipole range to use for matched filter
    - N: number of noise realizations to generate
    - map_cmb: original CMB realization to use as template
    Return: tau, sigma tau (array of len=3)
    '''
    l=np.arange(lmin_tau, lmax_tau+1)

    alm0=hp.map2alm(map0, lmax=lmax_tau, use_pixel_weights=True) #alms of maps to get difference map
    alm1=hp.map2alm(map1, lmax=lmax_tau, use_pixel_weights=True)
    alm_template_cmb=hp.map2alm(map_cmb, lmax=lmax_tau, use_pixel_weights=True)  #template from ord', iginal CMB map (not convolved, no noise)

    bl0_inv=1/np.sqrt(Cls_beam0[:lmax_tau+1]) ### NOTA: CLs_beam need to be cut to lmin_tau, lmax_tau
    bl1_inv=1/np.sqrt(Cls_beam1[:lmax_tau+1])

    lmax_alm = hp.Alm.getlmax(len(alm0))  ### check if alms and bl have the same lmax
    lmax_bl  = len(bl0_inv) - 1  # bl is indexed 0..lmax
    assert lmax_alm == lmax_bl, f"ERROR: lmax mismatch: alms have lmax={lmax_alm}, bl has lmax={lmax_bl}"


    alm0_dec=hp.almxfl(alm0, bl0_inv)  #deconvolve 
    alm1_dec=hp.almxfl(alm1, bl1_inv)   
    alm_data=alm0_dec-alm1_dec #get data map

    Cl_cov=hp.alm2cl(alm_data, lmax=lmax_tau)  #Cls of covariance = Cls of data (APPROX.)
    Cl_cov_cut=Cl_cov[lmin_tau:] 

    alm_template0=alm0_dec  #templates
    alm_template1=alm1_dec
    alm_template2=alm_template_cmb

    Cl_crossT0=hp.alm2cl(alm_data, alm_template0, lmax=lmax_tau)     #noise subtraction
    Cl_crossT1=hp.alm2cl(alm_data, alm_template1, lmax=lmax_tau)    #noise subtraction
    Cl_crossT2=hp.alm2cl(alm_data, alm_template2, lmax=lmax_tau)    #noise subtraction

    Cl_T0=hp.alm2cl(alm_template0, lmax=lmax_tau)  #template Cls
    Cl_T1=hp.alm2cl(alm_template1, lmax=lmax_tau)  
    Cl_T2=hp.alm2cl(alm_template2, lmax=lmax_tau)  

    #MATCHED FILTER
    nom0=np.sum((2*l+1)*Cl_crossT0[lmin_tau:]/Cl_cov_cut)  #with template 1
    denom0=np.sum((2*l+1)*Cl_T0[lmin_tau:]/Cl_cov_cut)
    tauE0=nom0/denom0
    sigma_tauE0=np.sqrt(1/denom0)

    nom1=np.sum((2*l+1)*Cl_crossT1[lmin_tau:]/Cl_cov_cut)  #with template 2
    denom1=np.sum((2*l+1)*Cl_T1[lmin_tau:]/Cl_cov_cut)
    tauE1=nom1/denom1
    sigma_tauE1=np.sqrt(1/denom1)

    nom2=np.sum((2*l+1)*Cl_crossT2[lmin_tau:]/Cl_cov_cut)  #with template 3
    denom2=np.sum((2*l+1)*Cl_T2[lmin_tau:]/Cl_cov_cut)
    tauE2=nom2/denom2
    sigma_tauE2=np.sqrt(1/denom2)

    tauE_arr=np.array([tauE0*1e3, tauE1*1e3, tauE2*1e3])
    sigma_tauE_arr=np.array([sigma_tauE0*1e3, sigma_tauE1*1e3, sigma_tauE2*1e3])

    return tauE_arr, sigma_tauE_arr

