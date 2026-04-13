import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


#Useful shortcuts: 
# CTRL + K,0  --->  collapse all functions
# CTRL + K,J  --->  expand all functions



def plot_alm_hist(map_beam):
    '''
    Plot 3 histograms with Re(alm), Im(alm), |alm| 
    '''

    #get alms from beam map
    nside=hp.get_nside(map_beam)
    lmax=3*nside-1
    alm_arr=hp.map2alm(map_beam, lmax=lmax)

    #get alm abs, Re, Im
    alm_abs = np.abs(alm_arr)

    almRe=alm_arr.real
    almRe_pos = almRe[almRe > 0]
    almRe_neg = -almRe[almRe < 0]  # absolute value for log bins

    almIm=alm_arr.imag
    almIm = almIm[almIm != 0]  # remove zeros
    almIm_pos = almIm[almIm > 0]
    almIm_neg = -almIm[almIm < 0]

    print('Re(alm)')
    print('positive (min,max) = ', np.min(almRe_pos), np.max(almRe_pos))
    print('- negative (min,max) = ', np.min(almRe_neg), np.max(almRe_neg))
    print('\nIm(alm):')
    print('positive (min,max) = ', np.min(almIm_pos), np.max(almIm_pos))
    print('- negative (min,max) = ', np.min(almIm_neg), np.max(almIm_neg))
    print('\n|alm|')
    print('(min,max) = ', np.min(alm_abs), np.max(alm_abs))

    #define histogram bins 
    def log_bins(*arrays, nbins=80):
        data = np.concatenate(arrays)
        xmin = data.min()
        xmax = data.max()
        return np.logspace(np.log10(xmin), np.log10(xmax), nbins)

    bins_re = log_bins(almRe_pos, almRe_neg)
    bins_im = log_bins(almIm_pos, almIm_neg)
    bins_abs = log_bins(alm_abs)


    #plot 
    fig, ax = plt.subplots(1, 3, figsize=(15,4))

    # ---- Re(alm) ----
    ax[0].hist(almRe_pos, bins=bins_re, histtype='step', label='positive values', linewidth=1.5, color='red')
    ax[0].hist(almRe_neg, bins=bins_re, histtype='step', label='negative values', linewidth=1.5, color='dodgerblue')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title("alms - real part", fontsize=15)
    ax[0].set_xlabel("Re($a_{\ell m}$)", fontsize=18)
    ax[0].set_ylabel("N", fontsize=15)
    ax[0].legend(fontsize=15)
    ax[0].grid(True, which='both', alpha=0.3)

    # ---- Im(alm) ----
    ax[1].hist(almIm_pos, bins=bins_im, histtype='step', label='positive values', linewidth=1.5, color='red')
    ax[1].hist(almIm_neg, bins=bins_im, histtype='step', label='negative values', linewidth=1.5, color='dodgerblue')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title("alms - imaginary part", fontsize=15)
    ax[1].set_xlabel("Im($a_{\ell m}$)", fontsize=18)
    ax[1].set_ylabel("N", fontsize=15)
    ax[1].legend(fontsize=15)
    ax[1].grid(True, which='both', alpha=0.2)

    # ---- abs ----
    ax[2].hist(alm_abs, bins=bins_abs, histtype='step', linewidth=1.5, color='k')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_title("alm - absolute value", fontsize=15)
    ax[2].set_xlabel("$|a_{\ell m}|$", fontsize=18)
    ax[2].set_ylabel("N", fontsize=15)
    ax[2].grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    plt.show()

    return 1


def plot_alm_heatmap(map_beam):
    '''
    Heatmap of |alms| in function of (l,m)
    '''

    #get alms from beam map
    nside=hp.get_nside(map_beam)
    lmax=3*nside-1
    alm_arr=hp.map2alm(map_beam, lmax=lmax)
    alm_abs=np.abs(alm_arr)

    #mask values below a certain threshold
    thresh=1e-12
    alm_abs_masked = np.ma.masked_less(alm_abs, thresh)
    alm_abs_masked_log = np.ma.log10(alm_abs_masked)

    #define arrays for (l,m) indeces
    l_arr = np.arange(0, lmax+1)
    m_arr = np.arange(0, lmax+1) 

    #define plot grid
    alm_grid = np.full((len(l_arr), len(m_arr)), np.nan) 
    for i, l in enumerate(l_arr):
        for j, m in enumerate(m_arr):     
            if abs(m) > l: # skip invalid harmonic pairs
                continue
            
            idx = hp.Alm.getidx(lmax, int(l), int(m))
            alm_val = alm_abs_masked_log[idx]
    
            alm_grid[i, j] = alm_val

    ## PLOT
    '''
    plt.figure(figsize=(8, 6))

    extent = [ m_arr.min() - 0.5,  m_arr.max() + 0.5, l_arr.min() - 0.5, l_arr.max() + 0.5 ]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    im = plt.imshow( alm_grid, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap=cmap)
    plt.xlabel("$m$", fontsize=18)
    plt.ylabel("$\ell$", fontsize=18)
    plt.title("spherical harmonic coefficients heatmap")

    cbar = plt.colorbar(im)
    
    cbar.set_label("$\log_{10} |a_{\ell m}|$")

    plt.tight_layout()
    #plt.savefig("/home/evanetti/BEAMS/paper_figures/alms_heatmap.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    '''

    return alm_grid



'''
def plot_alm_heatmap(map_beam, opt=='abs, 
'alm_arr, lmax, l_arr, m_arr,
                    quantity="abs",
                    log_scale=True):
    """
    Plot heatmap of alm coefficients 
    Args:
    - alm_arr: 1D healpix-format array with the alms
    lmax : int
        Maximum multipole.
    l_arr : array-like
        Array of l values to visualize.
    m_arr : array-like
        Array of m values to visualize.
    quantity : str
        "abs", "real", or "imag".
    log_scale : bool
        If True and quantity == "abs", apply log10 scaling.
    """


    #define arrays for (l,m) indeces
    l_arr = np.arange(0, lmax+1)
    m_arr = np.arange(0, lmax+1)


    # Output grid
    alm_grid = np.full((len(l_arr), len(m_arr)), np.nan)

    for i, l in enumerate(l_arr):
        for j, m in enumerate(m_arr):     
            if abs(m) > l: # skip invalid harmonic pairs
                continue
            
            idx = hp.Alm.getidx(lmax, int(l), int(m))
            alm = alm_arr[idx]

            almRe=np.real(alm)
            almIm=np.imag(alm)

            if opt == 'abs':
                alm_ = np.abs(alm)
                alm_log = np.log10(alm_ + 1e-10)
            elif opt == "real":
                alm = np.real(alm)
            elif opt == "imag":
                alm = np.imag(alm)
            else:
                raise ValueError("quantity must be 'abs', 'real', or 'imag'")


            masked = np.ma.masked_where(data < threshold, data)

            alm_grid[i, j] = almRe

    plt.figure(figsize=(8, 6))

    extent = [
        m_arr.min() - 0.5,
        m_arr.max() + 0.5,
        l_arr.min() - 0.5,
        l_arr.max() + 0.5,
    ]

    im = plt.imshow(
        alm_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        cmap="viridis"
    )

    plt.xlabel("$m$", fontsize=18)
    plt.ylabel("$\ell$", fontsize=18)
    plt.title("spherical harmonic coefficients heatmap")

    cbar = plt.colorbar(im)

    if quantity == "abs" and log_scale:
        cbar.set_label("$\log_{10} |a_{\ell m}|$")
    else:
        cbar.set_label(quantity)

    plt.tight_layout()
    plt.savefig("/home/evanetti/BEAMS/paper_figures/alms_heatmap.pdf", format="pdf", bbox_inches="tight")
    plt.show()
'''



