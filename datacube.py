__version__ = 0.1

import os
import numpy as np
import pyneb as pn
import pandas as pd

from . import utils
from tqdm import tqdm
from astropy.io import fits
from astropy import constants as const
from ifscube.stats import line_flux_error
from uncertainties import ufloat, umath, unumpy

c = const.c.to("km/s").value

class Datacube:
    def __init__(self, fpath, **kwargs):
        if kwargs:
            self._load(**kwargs)

        ### Validating provided filepath
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")
        self.filepath = fpath

        ### Load data cube assuming IFSCUBE standard nomenclature for keyword headers
        self.keynames = self._load_cube()

    def _load(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _load_cube(self):
        ### Mapping BITPIX values to NumPy data types
        bitpix_to_dtype = {8: np.uint8, 16: np.int16,
                           32: np.int32, 64: np.int64,
                           -32: np.float32, -64: np.float64}

        ### Headers will be stored together
        self.headers = {} 
        with fits.open(self.filepath, "readonly") as hdul:
            for hdu in hdul:
                ext = hdu.name.lower()
                self.headers[ext] = hdul[hdu.name].header
                dtype = bitpix_to_dtype.get(hdu.header.get("BITPIX", -32), np.float32)
                
                if hdu.data is not None:
                    if hdu.is_image: setattr(self, ext, hdu.data.astype(dtype, copy=False))
                    else: setattr(self, ext, hdu.data)
        
        ### Making sure hdulist is cleaned from memory
        del hdul 

        ### Handle potential missing extensions
        keynames = list(self.headers.keys())

        for unc_keys in ["DISP", "MC_RUNS"]:
            if not np.isin(unc_keys.lower(), keynames): setattr(self, unc_keys.lower(), None)

        if np.isin("mask2d", keynames): self.mask2d = self.mask2d.astype(bool)
        else: self.mask2d = np.zeros(self.fitspec.shape[1:], dtype=bool)

        return keynames
    
    def get_gaussian_moments(self, line):
        ### Validate emission line alias
        assert np.isin(line, self.featwl.feature), \
            f"Line {line} was not found among emission line aliases"

        A, v, sig_v = self.solution[self.parnames.component == line, :, :]

        return A, v, sig_v

    def get_gaussian_lineflux(self, line, dust_corr=False):
        ### Validate emission line alias
        assert np.isin(line, self.featwl.feature), \
            f"Line {line} was not found among emission line aliases"
        
        if dust_corr:
            if not hasattr(self, "ebv"):
                message="""
                E(B-V) map not computed. Call get_ebv_map() first, or set dust_corr=False"""
                raise ValueError(message)

        ### Load emission line fitted parameters
        A, v, sig_v = self.solution[self.parnames.component == line, :, :]
        l0 = self.featwl.rest_wavelength[self.featwl.feature == line][0]

        flux = A * np.sqrt(2.*np.pi) * np.divide(sig_v, c) * l0 * (1. + np.divide(v, c))

        ### Check if errors are available
        if self.disp is None: flux_err = 0.0
        else:
            flux_err = np.empty_like(flux)
            A_err, _, _ = self.disp[self.parnames.component == line, :, :]

            m = np.logical_and(~self.mask2d, A_err != 0.)
            flux_err[m] = line_flux_error(flux[m], utils.FWHM_MUSE(l0, out_type="fwhm", out_unit="AA"),
                                          np.ediff1d(self.restwave).mean(), A[m], A_err[m])
            flux_err[~m] = 0.0

        if dust_corr:
            ## Instantiate reddening correction PyNeb object
            rc = pn.RedCorr()
            rc.law, rc.R_V = self.reddening_law, self.Rv

            ebv, uebv = self.ebv.copy(), self.uebv.copy()
            ebv[np.isnan(ebv)], uebv[np.isnan(uebv)] = 0., 0.

            ### Apply dust correction
            E_BV = unumpy.uarray(ebv, uebv)
            corr_term = np.power(10., 0.4 * E_BV * rc.X(l0))

            f = unumpy.uarray(flux, flux_err) * corr_term
            flux = unumpy.nominal_values(f)
            flux_err = unumpy.std_devs(f)

        return flux, flux_err

    def emlines_at_xy(self, x, y, dust_corr=False):
        """
        Function that returns a table containing the main properties
        of fitted emission line features at a given position (x,y)
        """
        d = {"line": list(self.featwl.feature),
             "wave": self.featwl.rest_wavelength.astype(np.float64),
             "flux":[], "flux_err":[], "A": [], "A_err": [],
             "v": [], "v_err": [], "sig": [], "sig_err": []}

        for i, lname in enumerate(self.featwl.feature):
            f, ferr = self.get_gaussian_lineflux(lname, dust_corr=dust_corr)
            
            A, v, sig = self.solution[self.parnames.component == lname, :, :]
            if self.disp is None:
                ferr = np.zeros_like(f)
                A_err = np.zeros_like(A)
                v_err = np.zeros_like(v)
                sig_err = np.zeros_like(sig)
            else:
                A_err, v_err, sig_err = self.disp[self.parnames.component == lname, :, :]

            array_list = [f[y,x], ferr[y,x], A[y,x], A_err[y,x],
                          v[y,x], v_err[y,x], sig[y,x], sig_err[y,x]]
            
            for i, key in enumerate(list(d.keys())[2:]):
                d[key].append(array_list[i])
            
        return pd.DataFrame(d)
    
    def get_snr(self):
        """
        Calculate the signal-to-noise ratio (SNR) for each spectral feature
        in the data cube.

        The SNR is computed as the ratio between the fitted line flux and the 
        associated error (flux / flux_err). This error is derived from the
        Gaussian fit—typically available only if MCMC iterations were performed.
        If all error values are zero (i.e., no uncertainty estimate available),
        an exception is raised as SNR cannot be reliably computed.

        Returns
        -------
        dict
            A dictionary where keys are feature names and values are SNR values.

        Raises
        ------
        ValueError
            If all flux error values (`ferr`) are zero for any spectral feature.
        """
        snr = {}
        for lname in self.featwl.feature:
            f, ferr = self.get_gaussian_lineflux(lname)

            if np.all(ferr == 0):
                raise ValueError(
                    f"Cannot compute SNR for '{lname}': all flux errors are zero. "
                    "Ensure MCMC was performed to estimate uncertainties.")
            
            snr[lname] = np.divide(f, ferr, where=ferr != 0., out=np.zeros_like(f))

        return snr
    
    def get_ebv_map(self, ha_alias="ha", hb_alias="hb",
                    snr_min_ha=3., snr_min_hb=3., reddening_law="CCM89",
                    Rv=3.1, balmer_dec=2.863):
        
        ### Instantiate reddening correction PyNeb object
        rc = pn.RedCorr()
        rc.law, rc.R_V = reddening_law, Rv
                
        ### Start loading the snr of the emission lines
        snr = self.get_snr()
        snr_ha, snr_hb = snr[ha_alias], snr[hb_alias]
        
        ### Initialize output arrays
        # ebv = np.full(snr_ha.shape, np.nan)
        # uebv = np.full_like(ebv, np.nan)
        ebv = np.full(snr_ha.shape, 0.)
        uebv = np.full_like(ebv, 0.)

        ### Define mask based on minimum SNR
        mask_snr = np.logical_or(snr_ha < snr_min_ha, snr_hb < snr_min_hb)

        ### Line info
        ha, err_ha = self.get_gaussian_lineflux(ha_alias, dust_corr=False)
        hb, err_hb = self.get_gaussian_lineflux(hb_alias, dust_corr=False)
        obs_ratio = np.divide(ha, hb, where=~mask_snr, out=np.zeros_like(ha))

        wave1 = self.featwl.rest_wavelength[self.featwl.feature == ha_alias][0]
        wave2 = self.featwl.rest_wavelength[self.featwl.feature == hb_alias][0]

        ### Apply dust correction iterating only over spaxels with sufficient signal
        valid_y, valid_x = np.where(~mask_snr)
        for iy,ix in tqdm(zip(valid_y, valid_x), total=valid_x.size,
                          desc="Calculating color excess for unmasked spaxels"):
            
            ### Apply correction
            rc.setCorr(obs_over_theo = obs_ratio[iy,ix]/balmer_dec, wave1=wave1, wave2=wave2)
            if rc.E_BV >= 0.:    
                ebv[iy, ix] = rc.E_BV.ravel()[0]
            
                ### Estimating the uncertainty
                cte = 2.5/np.log(10.)
                k_ha, k_hb = rc.X(wave1), rc.X(wave2)
                t1, t2 = np.divide(err_hb[iy,ix], hb[iy,ix]), np.divide(err_ha[iy,ix], ha[iy,ix])

                uebv[iy, ix] = (cte/(k_hb - k_ha)) * np.sqrt(t1**2. + t2**2.)

            else:
                ebv[iy, ix], uebv[iy, ix] = 0.0, 0.0

        self.ebv, self.uebv = ebv, uebv
        self.reddening_law = rc.law
        self.balmer_dec = balmer_dec
        self.Rv = rc.R_V
        
        return None
    
    def rc_emlines_at_xy(self, ix, iy):
        if not hasattr(self, "ebv"):
            raise ValueError("E(B-V) map not computed. Call get_ebv_map() first.")
        
        df = self.emlines_at_xy(ix, iy)
        df_rc = df.copy()

        ### Use derived color excess value at this position. Instantiate PyNeb object
        rc = pn.RedCorr()
        rc.law, rc.R_V = self.reddening_law, self.Rv
        E_BV = ufloat(self.ebv[iy, ix], self.uebv[iy, ix])

        ### Initialize corrected values
        rc_flux, rc_flux_err = [], []

        ### Iterate over each line feature
        flux, flux_err, wave = df[["flux", "flux_err", "wave"]].to_numpy().T
        for j in range(flux.size):
            # Calculate the correction term
            corr_term = 10.0 ** (0.4 * E_BV * rc.X(wave[j]))

            # Apply the correction, propagating the uncertainty
            rc_f = ufloat(flux[j], flux_err[j]) * corr_term
            rc_flux.append(rc_f.nominal_value)
            rc_flux_err.append(rc_f.std_dev)

        df_rc["flux"] = rc_flux
        df_rc["flux_err"] = rc_flux_err

        del df
        return df_rc