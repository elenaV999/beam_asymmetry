import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import functions_rot as frot

from scipy.interpolate import UnivariateSpline
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FormatStrFormatter, LogLocator


#Useful shortcuts: 
# CTRL + K,0  --->  collapse all functions
# CTRL + K,J  --->  expand all functions



############################
### Get beam quantities ####     
############################
def beam_radius(map_beam, vec_c):
    '''
    Compute radius including all non-zero pixels of beam map
    Args: 
    - beam map
    - vector to beam center
    Returns: 
    - angdistR: distance from beam center to include all pixels
    '''
    nside=hp.get_nside(map_beam)

    beam_idpix0 = np.where(map_beam != 0)  #get pixels where beam map is non-zero
    beam_idpix0 = np.array(beam_idpix0[0])
    v0=hp.pix2vec(nside, beam_idpix0) 
    v0=np.array(v0)
    v0=v0.T

    v1c_tile=np.tile(vec_c, (len(v0), 1))  #get distances of pixels from the center (ang. dist. in radians)
    angdist=hp.rotator.angdist(v1c_tile.T, v0.T) 
    angdistR=np.max(angdist)*1.01  # set beam radius x% away from max angdistance, to encircle all pixels

    return angdistR


def beam_rad_profile(map_beam, angdistR, plot=False): 

    nside=hp.get_nside(map_beam) #get Cls
    lmax_beam=3*nside-1
    l=np.arange(lmax_beam+1)
    Cls_beam=hp.anafast(map_beam, lmax=lmax_beam)
    bl=np.sqrt(4*np.pi*Cls_beam)

    theta_arr=np.linspace(0,angdistR*1.1, 100)
    beam_profile=hp.bl2beam(bl, theta_arr)

    if plot==True: 
        plt.figure(figsize=(8,5))
        plt.plot(theta_arr, beam_profile, color='blue')
        plt.hlines(min(theta_arr), max(theta_arr), 0., color='gray', ls='--', alpha=0.6)
        plt.yscale('log')
        plt.xlabel(r'$\theta$', fontsize=20)
        plt.ylabel(r'$\frac{|B_2-B_1|}{B_1}$', fontsize=22)
        plt.title(r'beam radial profile from $4\pi C_\ell$s', fontsize=16)
        plt.show()


    return theta_arr, beam_profile


####################
### Read beams  ####     
####################

def read_beam(frequency, printtext=False):
    '''
    Read beam maps from FITS files + normalize beam
    Arg: 
    - beam frequency
    Return: 
    - beam map normalized to 1, cast to float64
    - beam center vector
    '''
    beam_dict={'30GHz':0, '70GHz':1, '143GHz':2}  

    if beam_dict[frequency]==0:  #3446 non-zero pixels  . (min, max) = (3.3462144983786857e-06, 3.023334264755249)  // cut at 1e-4 --> 0.00018 cut fraction, reduce to 2062 pixels
        fbeam='beams_030_2247339.fits'
        cpix=2247339
        xsize_beam=2800

    if beam_dict[frequency]==1:  #717 non-zero pixels (don't cut)  -  (min, max) = (9.00245358934626e-05, 18.002927780151367)
        fbeam='beams_070_2247339.fits'  #effective beam at theta=40, phic60
        cpix=2247339
        xsize_beam=1200


    if beam_dict[frequency]==2:  #10646 (need to cut) -  (min, max) = (-0.0005849457229487598, 59.50037384033203) // cut at 1e-4 --> 0.000186 cut fraction 
        fbeam='beams_143_8992085.fits'
        cpix=8992085
        xsize_beam=2000

    fbeam_dir='/home/evanetti/BEAMS/code_input/'+fbeam

    map_beam_ini=hp.fitsfunc.read_map(fbeam_dir, field=0)  #initial beam map, with less precise normalization
    map_beam_ini = np.asarray(map_beam_ini, dtype=np.float64)

    nside=hp.get_nside(map_beam_ini)
    #res_deg = hp.nside2resol(nside, arcmin=True)
    #print(f"Resolution: {res_deg:.2f} arcmin")
    if printtext==True:
        print('Beam file: ', fbeam)
        print('Nside = ', nside)


    # Beam center
    angc=hp.pixelfunc.pix2ang(nside, ipix=cpix, lonlat=True) #reurns phi, theta(lat)
    v1c=hp.pixelfunc.pix2vec(nside, ipix=cpix)
    v1c=np.array(v1c)
    if printtext==True:
        print('\nBeam center:')
        print('- center pixel idx = ', cpix) 
        print('- center lonlat (phi,theta) = ',angc)
        print('- center vector =', v1c)

    # Beam normalization
    pixArea_rad = hp.nside2pixarea(nside, degrees=False)

    norm=np.sum(map_beam_ini)*pixArea_rad
    print('Normalization factor = ', norm)
    map_beam=map_beam_ini/norm
    I=np.sum(map_beam)*pixArea_rad
    print('Normalized beam integral = ', I)

    return map_beam, v1c


def plot_beam(map_beam, v1c):

    #compute Cls
    nside=hp.get_nside(map_beam)
    lmax_beam=3*nside-1
    l=np.arange(lmax_beam+1)
    Cls_beam=hp.anafast(map_beam, lmax=lmax_beam)
    for i in range(5):
        print(f'l={l[i]}\t 4pi*Cl=: {Cls_beam[i]*4*np.pi} \t Cl = {Cls_beam[i]}')

    #compute radial profile
    angdistR=beam_radius(map_beam, v1c)
    theta_arr, beam_profile = beam_rad_profile(map_beam, angdistR, plot=False)
    print('max map_beam = ', np.max(map_beam), '\tmax rad profile = ', np.max(beam_profile))

    #PLOT
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 3 subplots now

    angc_=hp.pixelfunc.vec2ang(np.array(v1c), lonlat=True)
    phic_=angc_[0][0]
    thetac_=angc_[1][0]

    map_beam_masked = np.where(map_beam==0, hp.UNSEEN, map_beam)
    hp.visufunc.gnomview(map_beam_masked, rot=[phic_, thetac_], reso=0.1, xsize=1200, norm='log', title='PLA beam map', return_projected_map=True , sub=(1, 3, 1))
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    axes[0].set_xticks([])  # remove x-axis ticks
    axes[0].set_yticks([])  # remove y-axis ticks


    axes[1].loglog(l, Cls_beam/Cls_beam[0], color='b')
    axes[1].set_title("Beam Cls", fontsize=15)
    axes[1].set_xlabel(r'$\ell$', fontsize=14)
    axes[1].set_ylabel(r'$4\pi C_\ell$', fontsize=14)

    axes[2].plot(theta_arr, beam_profile, color='red')
    axes[2].set_yscale('log')
    axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[2].set_title("Beam radial profile", fontsize=15)
    axes[2].set_xlabel(r"$\theta$", fontsize=14)
    axes[2].set_ylabel(r'$B(\theta)$', fontsize=14)

    plt.tight_layout()
    plt.show()

    #Plot beam map
    

    return 1


def compare_beams(v1c, v2c, map_beam1, map_beam2):

    #needed quantities
    nside_beam=hp.get_nside(map_beam1)
    pixArea = hp.nside2pixarea(nside_beam, degrees=False) 

    beam_idpix1 = np.where(map_beam1 != 0) 
    beam_idpix2 = np.where(map_beam2 != 0) 

    angc1=hp.pixelfunc.vec2ang(np.array(v1c), lonlat=True)
    phic1=angc1[0][0]
    thetac1=angc1[1][0]

    angc2=hp.pixelfunc.vec2ang(np.array(v2c), lonlat=True)
    phic2=angc2[0][0]
    thetac2=angc2[1][0]
    
    #max value: 
    max1=np.max(map_beam1)
    max2=np.max(map_beam2)
    print(f'Max value: {max1} original map\t{max2} rotated map\t difference = {max2-max1}\trel difference = {(max2-max1)/max1*100}%\n')
    
    
    #beam moments 
    for n in range(1,6):
        In1=sum( map_beam1[beam_idpix1]**n)*pixArea
        In2=sum( map_beam2[beam_idpix2]**n)*pixArea
        print(f'{n}-order moment: In1={In1}\tIn2={In2}\tdifference = {In1-In2}\trel difference = {(In1-In2)/In1*100} %')

    #Cls  
    lmax_beam=3*nside_beam-1
    l=np.arange(lmax_beam+1)
    Cl_beam1=hp.anafast(map_beam1, lmax=lmax_beam)
    Cl_beam2=hp.anafast(map_beam2, lmax=lmax_beam)
    diff_Cls=np.abs(Cl_beam2-Cl_beam1)
    diff_rel_Cls = np.abs(Cl_beam2-Cl_beam1)/Cl_beam1
    print('\nCls factional difference at l=0', diff_rel_Cls[0] )

    #Radial profiles: 
    angdistR=beam_radius(map_beam1, v1c)
    theta_arr, beam_profile1 = beam_rad_profile(map_beam1, angdistR, plot=False)
    theta_arr, beam_profile2 = beam_rad_profile(map_beam2, angdistR, plot=False)

    diff_profile=np.abs(beam_profile2-beam_profile1)
    diff_rel_profile = np.abs(beam_profile2-beam_profile1)/beam_profile1

    # Plot Cls and radial profiles
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex='col', gridspec_kw={'hspace': 0.1})

    axes[0,0].loglog(l, Cl_beam1, color='b', label='beam 1')
    axes[0,0].loglog(l, Cl_beam2, color='r', label='beam 2')
    axes[0,0].tick_params(labelbottom=False)
    axes[0,0].set_ylabel(r'$C_\ell$', fontsize=14)
    axes[0,0].legend(fontsize=16)
    axes[0,0].set_title("Beam Cls", fontsize=16)

    axes[1,0].plot(l, diff_rel_Cls, color='k', label='fractional difference')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].set_xlabel(r'$\ell$', fontsize=14)
    axes[1,0].set_ylabel(r'$\frac{|C_\ell^2-C_\ell^1|}{C_\ell^1}$', fontsize=14)
    axes[1,0].legend(fontsize=16)

    axes[0,1].plot(theta_arr, beam_profile1, color='b', label='beam 1')
    axes[0,1].plot(theta_arr, beam_profile2, color='r', label='beam 2')
    axes[0,1].set_yscale('log')
    axes[0,1].tick_params(labelbottom=False)
    axes[0,1].set_ylabel(r'$B(\theta)$', fontsize=14)
    axes[0,1].legend(fontsize=16)
    axes[0,1].set_title("Beam radial profile", fontsize=16)

    axes[1,1].plot(theta_arr, diff_rel_profile, color='k', label='fractional difference')
    axes[1,1].set_yscale('log')
    axes[1,1].set_xlabel(r'$\theta$', fontsize=14)
    axes[1,1].set_ylabel(r'$\frac{|B_2-B_1|}{B_1}$', fontsize=14)
    axes[1,1].legend(fontsize=16)

    plt.show()


    #Beam maps 
    # 30 GHz beam: xsize=2800
    # 70 GHz beam: xsize=1200
    # 143 GHz beam: xsize=
    fig = plt.figure(figsize=(12, 4))
    hp.mollview(map_beam1, title="Beam locations" , norm='hist', unit=r'$\mu K$',sub=(1, 3, 1))
    hp.projplot(phic1, thetac1 , 'ro', markersize=5, lonlat=True)
    hp.projplot(phic2, thetac2, 'wo', markersize=5, lonlat=True)

    map_beam1_masked = np.where(map_beam1==0, hp.UNSEEN, map_beam1)
    map_beam1_grid = hp.visufunc.gnomview(map_beam1_masked, rot=[phic1, thetac1], reso=0.1, xsize=1200, norm='hist', title='Beam 1', return_projected_map=True , sub=(1, 3, 2))
    #hp.projplot(phic1, thetac1, 'ro', markersize=5, lonlat=True) 

    map_beam2_masked = np.where(map_beam2==0, hp.UNSEEN, map_beam2)
    map_beam2_grid = hp.visufunc.gnomview(map_beam2_masked, rot=[phic2, thetac2], reso=0.1, xsize=1200, norm='hist', title='Beam 2', return_projected_map=True, sub=(1, 3, 3)) #min=1.0e-5, max=18. 
    #hp.projplot(phic2, thetac2, 'wo', markersize=5, lonlat=True)

    plt.show()


    # Map difference - compute only if the 2 beams are at the same position
    if np.allclose(v1c, v2c):

        diff = map_beam2_grid-map_beam1_grid
        diff_rel = (map_beam2_grid-map_beam1_grid)/map_beam1_grid
        print('difference map (min,max) = ', np.min(diff),np.max(diff) )
        print(f'relative difference map (min,max) =  {np.min(diff_rel)*100} % , {np.max(diff_rel)*100}%' )

        linthresh = 1e-3  # Values between -1e-2 and 1e-2 will be shown linearly
        linscale = 1.0    # Controls the size of the linear range

        plt.figure(figsize=(5, 5))

        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        axs[0].imshow(diff,origin='lower', cmap='viridis', norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=np.min(diff), vmax=np.max(diff)))
        axs[0].set_title('Difference map', fontsize=16)

        axs[1].imshow(diff_rel,origin='lower', cmap='viridis', norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=np.min(diff_rel), vmax=np.max(diff_rel)))
        axs[1].set_title('Relative difference map', fontsize=16)

        #cbar = plt.colorbar()
        #cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        
        #cbar.locator = LogLocator(numticks=10)
        #cbar.update_ticks()

        plt.show()


    return 1


#############################
#### Get symmetric beam  ####
#############################

def symmetrize_beam(map_beam, vec_c, plot=False):
    '''
    Build symmetric beam map. 
    Args: beam map, vector to beam center
    Plot (optional): beam radial profile + interpolation to all pixels + symmetric beam map
    Returns: 
    '''
    angdistR=beam_radius(map_beam, vec_c)
    theta_arr, beam_profile = beam_rad_profile(map_beam, angdistR, plot=False) #get beam radial profile
    interpolator_spline = UnivariateSpline(theta_arr, beam_profile, s=0, k=3) #build interpolator

    #get ang dist of each pixel from center --> points to interpolate at
    nside=hp.get_nside(map_beam)
    idpix_disk=hp.query_disc(nside=nside, vec=vec_c, radius=angdistR, inclusive=False, nest=False, buff=None) 
    print(len(idpix_disk), ' pixels inside defined beam radius')
    vec_disk=np.array(hp.pix2vec(nside, idpix_disk))
    vc_tile=np.tile(vec_c, (len(idpix_disk), 1))  
    angdist_disk=hp.rotator.angdist(vc_tile.T, vec_disk) #angular distance (in radians) of each pixel from center pixel

    #interpolate to beam radial profile
    beam_interp_symm = interpolator_spline(angdist_disk)
    max_symm=np.max(beam_interp_symm)
    print('symmetric beam max: ', max_symm)
    print('original beam max: ', np.max(map_beam))

    #create symmetric beam map
    npix=12*nside**2
    map_beam_symm=np.zeros(npix)
    map_beam_symm[idpix_disk]=beam_interp_symm 

    #PLOTS
    if plot==True :#check inyterpolation 
        fig = plt.figure(figsize=(8, 5))
        plt.plot(theta_arr, beam_profile , color='g', zorder=0 , label='beam radial profile')
        plt.scatter(angdist_disk, beam_interp_symm , color='r', s=2.5 , label='interpolated values')
        plt.yscale('symlog')
        plt.xlabel(r'$\theta$', fontsize=18)
        plt.ylabel(r'$B(\theta)$', fontsize=18)
        plt.title('Beam radial profile', fontsize=15)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(10, 5))
        angc=hp.pixelfunc.vec2ang(np.array(vec_c), lonlat=True)
        phic=angc[0][0]
        thetac=angc[1][0]
        min_val=np.min(map_beam[map_beam != 0])
        map_beam_masked = np.where(map_beam==0, hp.UNSEEN, map_beam)
        map_beam_symm_masked = np.where(map_beam_symm==0, hp.UNSEEN, map_beam_symm)
        hp.visufunc.gnomview(map_beam_symm_masked, rot=[phic, thetac], reso=0.1, xsize=1200, norm='hist', title='Symmetric beam', return_projected_map=True, sub=(1, 2, 1), min=min_val) # max=np.max(map_beam_pole)
        hp.visufunc.gnomview(map_beam_masked, rot=[phic, thetac], reso=0.1, xsize=1200, norm='hist', title='Original beam', return_projected_map=True, sub=(1, 2, 2)) 
        plt.show()

    return map_beam_symm
