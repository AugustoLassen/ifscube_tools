__version__ = 0.1

import numpy as np
from astropy import constants as const
from scipy.stats import norm, kurtosis

c = const.c.to("km/s").value

def FWHM_MUSE(lmb, out_type="fwhm", out_unit="AA"):
    """
    Calculate the spectral FWHM of MUSE instrument for a given wavelength or
    wavelength ndarray, based on the Guerou+17 parametrization of MUSE LSF.

    Parameters:
    ----------
    lmb : float or ndarray
        The input wavelength(s) in Angstroms. Can be a single float value or a NumPy array.
        
    out_type : str, optional, default: 'fwhm'
        Determines the type of output:
        - 'fwhm': Returns the Full Width at Half Maximum (FWHM).
        - 'sig': Returns the velocity dispersion.
        
    out_unit : str, optional, default: 'AA'
        The unit of the output:
        - 'AA': Output in Angstroms (for 'fwhm') or velocity dispersion in Angstroms (for 'sig').
        - 'kms': Output in km/s.
    
    Returns:
    -------
    float or ndarray
        The computed FWHM or velocity dispersion for the input wavelength(s), in the desired units.

    Reference:
    ----------         
    https://ui.adsabs.harvard.edu/abs/2017A%26A...608A...5G/abstract
    Guerou et. al 2017
    """
    assert (out_type == "fwhm") or (out_type == "sig"), \
        f"Unrecognized option for out_type: {out_type}"
    assert (out_unit == "AA") or (out_unit == "kms"), \
        f"Unrecognized option for out_unit: {out_unit}"
    
    cte = 2.*np.sqrt(2.*np.log(2.))
    FWHM_MUSE = 5.866e-08*np.power(lmb, 2) -9.187e-04*lmb + 6.04

    if out_type == "fwhm":
        if out_unit == "AA": x = FWHM_MUSE
        else: x = c * np.divide(FWHM_MUSE, lmb)
    else:
        x = np.divide(FWHM_MUSE, cte)
        if out_unit == "AA": pass
        else: x *= np.divide(c, lmb)
    
    return x

def get_kurtosis(sp, lmb, peak_lmd, FWHM, flux):
    '''
    Calculate the kurtosis of the spectral line by defining the window such that the Gaussian fit of the line has kurtosis = 0.
    
    Parameters:
    ----------
    sp: array
        The observed spectrum including a surrounding window.
        
    lmb: array
        The wavelengths corresponding to the spectrum in Ångströms.
        
    peak_lmd: float
        The wavelength of the peak of the Gaussian fit.
        
    FWHM: float
        The FWHM of the Gaussian fit.
    
    flux: float
        The flux of the Gaussian fit.

    Returns
    -------
    float
        The kurtosis of the observed spectrum.

    '''
    
    #Starting with a window 2*FWHM
    window_right = peak_lmd + 2*FWHM
    window_left = peak_lmd - 2*FWHM
    
    #Arbitrary value
    kurt_gauss = 10
    n = 0
    
    
    while abs(kurt_gauss) > 0.1: 
        n += 1
        
        
        window_right += 0.1
        window_left -= 0.1
        
        x = np.arange(window_left, window_right, 0.2)
        gauss_OIII_ = norm.pdf(x, peak_lmd, FWHM/2.355) * flux
    
        kurtosis_gauss = kurtosis(gauss_OIII_)
        
        ind_min = np.argmin(abs(lmb - window_left))
        ind_max = np.argmin(abs(lmb - window_right))
        
        kurt = kurtosis(sp[ind_min:ind_max])
        
        # print(kurtosis_gauss)
        
        kurt_gauss = kurtosis_gauss
        
        if n == 50:
            break
    return(kurt)