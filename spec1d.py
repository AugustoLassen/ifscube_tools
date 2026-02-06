__version__ = 0.1

import os
import utils
import numpy as np
import pyneb as pn
import pandas as pd

from tqdm import tqdm
from astropy.io import fits
from ifscube.stats import line_flux_error
from uncertainties import ufloat, umath, unumpy

class Spec1D:
    def __init__(self, fpath, mc_ext=None, flux_type="gaussian", **kwargs):
        self._load(**kwargs)

        ### Load data cube assuming IFSCUBE standard
        ### nomenclature for keyword headers
        with fits.open(os.path.abspath(fpath), "readonly") as hdul:
            self.lmb = hdul["RESTWAVE"].data
            self.spectrum = hdul["FITSPEC"].data
            self.variance = hdul["VAR"].data
            self.cont = hdul["FITCONT"].data
            self.stellar = hdul["STELLAR"].data
            self.model_spectrum = hdul["MODEL"].data
            
            self.par = hdul["PARNAMES"].data
            self.lines = hdul["FEATWL"].data
            self.cfg = hdul["FITCONFIG"].data
                        
            if mc_ext is not None:
                try: self.markov_chains = hdul[mc_ext].data
                except:
                    print(f"Extension containing Markov Chains {mc_ext} could not be found")
                    print("and therefore will be ignored")
                    self.markov_chains = None
            else: self.markov_chains = None

            self.solution = self.get_sol(hdul["SOLUTION"].data)
            del hdul

        ### Table with line fluxes and corresponding errors
        self.fluxes = self.get_fluxes(flux_type)

        ### Signal to noise in the featureless continuum
        self.snr_fc = self.calculate_SNR_FC()

    def _load(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        ### Attribute standard values for some parameters
        # Instantiate reddening correction PyNeb object
        self.rc = pn.RedCorr()
        self.rc.law = kwargs.get("reddening_law", "CCM89")
        self.rc.R_V = kwargs.get("Rv", 3.1)
        
        self.FC_wave1 = kwargs.get("FC_wave1", 4730.)
        self.FC_wave2 = kwargs.get("FC_wave2", 4780.)
        
        # Theoretical Balmer decrement assuming case B of recombination
        self.Balmer_decrement = kwargs.get("Balmer_decrement", 2.863)

    def _vdisp_correction(self, sig, sig_err, sig_instr):
        if sig <= sig_instr:
            scorr, scorr_err = 0.0, 0.0
        else:
            SIG = ufloat(sig, sig_err)
            SIGCORR = umath.sqrt(np.subtract(SIG**2., sig_instr**2.))
            scorr, scorr_err = SIGCORR.n, SIGCORR.std_dev
        
        return scorr, scorr_err

    def get_sol(self, sol):
        sdict = {"line": [], "l0": []}

        keys = ["amplitude", "velocity", "sigma", "sigma_corr"]
        for k in keys:
            sdict[k], sdict[f"{k}_err"] = [], []
            
        for line, wave in zip(self.lines.feature, self.lines.rest_wavelength):
            sdict["line"].append(line)
            sdict["l0"].append(wave)

            amp, v, sigma = sol[self.par.component == line]
            sdict["amplitude"].append(amp)
            sdict["velocity"].append(v)
            sdict["sigma"].append(sigma)

            # Add the errors as well
            inst_broadening = utils.FWHM_MUSE(wave, out_type="sig", out_unit="kms")

            if self.markov_chains is None:
                sdict["amplitude_err"], sdict["velocity_err"], sdict["sigma_err"] = 0.0, 0.0, 0.0
                
                # Include velocity dispersion corrected for instrumental broadening            
                sig_corr, sig_corr_err = \
                    self._vdisp_correction(sigma, 0.0, inst_broadening)
            else:
                chains = self.markov_chains[:, self.par.component == line]
                amp_chains, v_chains, sig_chains = chains[:, 0], chains[:, 1], chains[:, 2]
                for ch, k in zip([amp_chains, v_chains, sig_chains],
                                 ["amplitude","velocity","sigma"]):
                    mu = sdict[k]
                    q1, q2 = np.percentile(ch, (15.9, 84.1))
                    sdict[f"{k}_err"].append(np.mean([q2 - mu, mu - q1]))

                # Include velocity dispersion corrected for instrumental broadening            
                sig_corr, sig_corr_err = \
                    self._vdisp_correction(sigma, sdict["sigma_err"][-1], inst_broadening)

            sdict["sigma_corr"].append(sig_corr)
            sdict["sigma_corr_err"].append(sig_corr_err)

        return pd.DataFrame(sdict)

    def get_fluxes(self, flux_type):
        assert (flux_type == "gaussian") or (flux_type == "direct"), \
            f"Unrecognized option for flux_type: {flux_type}"
        
        if flux_type == "direct":
            print("under development")
        if flux_type == "gaussian":
            fdict = {}
            keys = ["line", "l0", "flux", "flux_err"]
            for k in keys: fdict[k] = []

            for i,stab in self.solution.iterrows():
                fdict["line"].append(stab["line"])
                fdict["l0"].append(stab["l0"])

                # Get flux
                flux = stab["amplitude"] * np.sqrt(2.*np.pi) * np.divide(stab["sigma"], c) \
                    *stab["l0"] * (1. + np.divide(stab["velocity"], c))
                fdict["flux"].append(flux)

                # Get flux error
                if self.markov_chains is None:
                    flux_err = 0.0
                else:                    
                    fwhm = utils.FWHM_MUSE(stab["l0"], out_type="fwhm", out_unit="AA")
                    spectral_sampling = np.median(np.ediff1d(self.lmb))
                    flux_err = line_flux_error(flux, fwhm, spectral_sampling,
                                               stab["amplitude"], stab["amplitude_err"])
                    
                fdict["flux_err"].append(flux_err)
        return pd.DataFrame(fdict)
    
    def get_ebv(self, ha_alias="ha", hb_alias="hb"):
        df = self.fluxes.copy()
        ha, hb = df.loc[df["line"] == ha_alias], df.loc[df["line"] == hb_alias]

        # Estimate E(B-V)
        obs = (ha["flux"].values / hb["flux"].values)[0]
        self.rc.setCorr(obs_over_theo = obs/self.Balmer_decrement,
                        wave1=ha["l0"].values[0], wave2=hb["l0"].values[0])
        
        if self.rc.E_BV <= 0.:
            self.rc.E_BV = 0.0
            self.unc_ebv = 0.0
        else:
            # Estimate the uncertainty on the colour excess
            cte = 2.5/np.log(10.)
            k_ha, k_hb = self.rc.X(ha["l0"].values[0]), self.rc.X(hb["l0"].values[0])
            t1, t2 = (hb["flux_err"]/hb["flux"]).values[0], (ha["flux_err"]/ha["flux"]).values[0]

            self.unc_ebv = (cte/(k_hb - k_ha)) * np.sqrt(t1**2. + t2**2.)
        
        return self.rc.E_BV.ravel()[0], self.unc_ebv

    def get_rc_fluxes(self, f2corr=None):
        if f2corr is None: f2corr = self.fluxes.copy()
        else: pass

        rc_flux, rc_flux_err = [], []
        for i,ftab in f2corr.iterrows():
            rc_term = ufloat(self.rc.E_BV.ravel()[0], self.unc_ebv)
            F = ufloat(ftab["flux"], ftab["flux_err"])
            rcF = F*10.**(0.4*self.rc.X(ftab["l0"])*rc_term)

            rc_flux.append(rcF.n)
            rc_flux_err.append(rcF.std_dev)

        df = f2corr.drop(columns=["flux", "flux_err"])
        df.insert(loc=2, column="flux", value=rc_flux)
        df.insert(loc=3, column="flux_err", value=rc_flux_err)
        return df
    
    def calculate_SNR_FC(self):
        # Make wavelength boolean mask based on featureless continuum window
        wmask = (self.lmb >= self.FC_wave1) & (self.lmb <= self.FC_wave2)

        # Effectively calculate the SNR on the provided featureless continuum
        S = np.nanmean(self.spectrum[wmask])
        N = np.nanstd(self.spectrum[wmask])
        return S / N