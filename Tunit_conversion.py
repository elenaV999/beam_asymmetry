# Radio Astronomy Unit Conversions
import numpy as np

def planckcorr(nu_in):
    """
    Planck correction factor to convert between CMB and RJ temperature scales.

    Parameters:
    nu_in : frequency in GHz

    Returns:
    correction factor (dimensionless)
    """
    c = 299792458.      # speed of light [m/s]
    k = 1.3806488e-23   # Boltzmann constant [J/K]
    h = 6.62606957e-34  # Planck constant [J⋅s]
    T_cmb = 2.725       # CMB temperature [K]

    nu = nu_in * 1e9    # convert GHz to Hz
    x = h*nu/(k*T_cmb)  # dimensionless frequency

    return x**2*np.exp(x)/(np.exp(x) - 1.)**2

def toJy(nu, beam):
    """
    Convert from Kelvin to Jansky using Rayleigh-Jeans approximation.

    Parameters:
    nu   : frequency in GHz
    beam : beam solid angle in steradians

    Returns:
    conversion factor [Jy/K]
    """
    c = 299792458.      # speed of light [m/s]
    k = 1.3806488e-23   # Boltzmann constant [J/K]

    nu_hz = nu * 1e9    # convert GHz to Hz
    jy = 2 * k * nu_hz**2 / c**2 * beam * 1e26

    return jy

def Jy2K(nu, beam):
    """
    Convert from Jansky to Kelvin using Rayleigh-Jeans approximation.

    Parameters:
    nu   : frequency in GHz
    beam : beam solid angle in steradians

    Returns:
    conversion factor [K/Jy]
    """
    c = 299792458.      # speed of light [m/s]
    k = 1.3806488e-23   # Boltzmann constant [J/K]

    nu_hz = nu * 1e9    # convert GHz to Hz
    T = c**2 / (2*k*nu_hz**2*beam*1e26)  # inverse of toJy conversion

    return T

def convert_to_mK(mapinfo):
    """
    Convert HEALPix map to millikelvin units.

    Parameters:
    mapinfo : dict with keys 'frequency' (GHz), 'nside', 'units'

    Returns:
    multiplication factor to convert map values to mK

    Supported units:
    - 'K', 'mK', 'mK_RJ': Rayleigh-Jeans temperature scales
    - 'mKCMB', 'KCMB': CMB brightness temperature
    - 'MJysr': intensity units (MJy/sr)
    """
    nu = mapinfo['frequency']  # in GHz
    nside = mapinfo['nside']
    units = mapinfo['units']

    # HEALPix pixel solid angle
    pixbeam = 4.*np.pi/(12.*nside**2)  # steradians

    # Conversion factors to get mK
    conversions = {
        'K': 1e3,                                           # K → mK_RJ
        'mK_RJ': 1.,                                        # already mK_RJ
        'mK': 1.,                                           # already mK_RJ
        'mKCMB': planckcorr(nu),                           # CMB → mK_RJ
        'KCMB': planckcorr(nu)*1e3,                        # K_CMB → mK_RJ
        'MJysr': 1e6*pixbeam*Jy2K(nu, pixbeam)*1e3        # MJy/sr → mK_RJ
    }

    return conversions[units]

# Example usage
if __name__ == "__main__":
    # Example: Convert a Planck 143 GHz map from CMB to RJ temperature
    mapinfo = {
        'frequency': 143.0,  # GHz
        'nside': 2048,       # HEALPix resolution
        'units': 'mKCMB'     # Input units
    }

    factor = convert_to_mK(mapinfo)
    print(f"To convert 143 GHz CMB map to mK: multiply by {factor:.3f}")
