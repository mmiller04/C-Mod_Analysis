# Routine written by Francesco Sciortino
# Modified by Andrés Miller to remove use of geqdsk object

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_p, e as q_electron, Boltzmann as k_B
from scipy.interpolate import interp1d
import pickle as pkl, os
import cmod_tools

from scipy.constants import Boltzmann as kB, e as q_electron
import aurora

# from this repo
import data_access


# PFS this is the wrapper for the 2-point model
def Teu_2pt_model(shot,tmin,tmax, lambdaq_opt=1, rhop_vec=None, ne=None, Te=None):
    '''
    Get 2-point model prediction for Te at the LCFS.

    Parameters
    ----------
    shot : C-Mod shot number
    tmin, tmax : time window in seconds
    lambdaq_opt: int, choice of scaling to use 


    # for the time being, this functionality is removed - if want to use profile information, can add back in later
    rhop_vec : 1D arr, sqrt(norm.pol.flux) vector for ne and Te profiles, only used if pressure_opt=2    
    ne : 1D arr, electron density profile in units of 1e20m^-3, only used if pressure_opt=2
    Te : 1D arr, electron temperature profile in units of keV, only used if pressure_opt=2

    Returns
    -------
    Tu_eV : 2-point model prediction for the electron temperature at the LCFS
    '''
    # if geqdsk is None:
    #     geqdsk = cmod_tools.get_geqdsk_cmod(
    #                 shot, time*1e3, gfiles_loc = '/home/sciortino/EFIT/lya_gfiles/')

    time = (tmax+tmin)/2.
    
    P_rad_main = cmod_tools.get_CMOD_var(var='P_rad_main',shot=shot, tmin=tmin, tmax=tmax, plot=False)
    P_rad_diode = cmod_tools.get_CMOD_var(var='P_rad_diode',shot=shot, tmin=tmin, tmax=tmax, plot=False)
    P_RF = cmod_tools.get_CMOD_var(var='P_RF',shot=shot, tmin=tmin, tmax=tmax, plot=False)
    P_oh = cmod_tools.get_CMOD_var(var='P_oh',shot=shot, tmin=tmin, tmax=tmax, plot=False)
    q95 = cmod_tools.get_CMOD_var(var='q95',shot=shot, tmin=tmin, tmax=tmax, plot=False)

    eff = 0.8 # value suggested by paper by Bonoli
    P_rad = P_rad_main if P_rad_main is not None else P_rad_diode # sometimes P_rad main will be missing
    Psol = eff *P_RF + P_oh - P_rad

    if Psol<0:
        print('Inaccuracies in Psol determination do not allow a precise 2-point model prediction. Set Te_sep=60 eV')
        return 60.


    # alternate ways of calculating pressure for Brunner scaling - can be added back in later if desired

    # if pressure_opt==1:
    #     # Use W_mhd rather than plasma pressure:
    #     vol =  geqdsk['fluxSurfaces']['geo']['vol']
    #     W_mhd = cmod_tools.get_CMOD_var(var='Wmhd',shot=shot, tmin=tmin, tmax=tmax, plot=False)
    #     p_Pa_vol_avg = 2./3. * W_mhd / vol[-1]

    # elif pressure_opt==2:
    #     # find volume average within LCF from ne and Te profiles
    #     p_Pa = 2 * (ne*1e20) * (Te*1e3*q_electron)  # taking pe=pi
    #     indLCFS = np.argmin(np.abs(rhop_vec-1.0))
    #     p_Pa_vol_avg = aurora.vol_average(p_Pa[:indLCFS], rhop_vec[:indLCFS], geqdsk=geqdsk)[-1]

    # use tor beta to extract vol-average pressure
    BTaxis = np.abs(cmod_tools.get_CMOD_var(var='Bt',shot=shot, tmin=tmin, tmax=tmax, plot=False))
    betat = cmod_tools.get_CMOD_var(var='betat',shot=shot, tmin=tmin, tmax=tmax, plot=False)
    p_Pa_vol_avg = (betat/100)*BTaxis**2.0/(2.0*4.0*np.pi*1e-7)   # formula used by D.Brunner


    # B fields at the LFS LCFS midplane - need equilibrium information
    try: # EFIT20 only exists for shots from certain years
        e = data_access.CModEFITTree(int(shot), tree='EFIT20', length_unit='m')
    except:
        e = data_access.CModEFITTree(int(shot), tree='analysis', length_unit='m')
    Rlcfs = e.rho2rho('psinorm', 'Rmid', 1, time)
    gfilename = '/home/millerma/lya/gfiles/' + f'g{shot}.{str(int(time*1e3)).zfill(5)}'

    Bt = np.abs(e.rz2BT(Rlcfs, 0, time)) 
    Bp = np.abs(e.rz2BZ(Rlcfs, 0, time)) 

    if lambdaq_opt==1:
        # now get 2-point model prediction
        mod = two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0,  # dummy value of ne at the sepatrix, not used for Tu_eV calculation
            lam_q_model='Brunner'
        )
    if lambdaq_opt==2:
        # now get 2-point model prediction
        mod = two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0,  # dummy value of ne at the sepatrix, not used for Tu_eV calculation
            lam_q_model='Eich'
        )
    elif lambdaq_opt==3:
        # use lambdaT calculated from TS points to get lambda_q using lambda_q = 2/7*lambda_T
        mod = two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0, 
            lam_q_model='lam_T'
        )

    return mod.Tu_eV, mod.lam_q_mm



class two_point_model:
    '''
    2-point model results, all using SI units in outputs (inputs have other units as indicated)
    Refs: 
    - H J Sun et al 2017 Plasma Phys. Control. Fusion 59 105010 
    - Eich NF 2013
    - A. Kuang PhD thesis

    This should be converted from being a class to being a function at some point, but that may break a bunch 
    of dependencies, so.... it might just stay as it is. See the bottom of this script for an example on how to 
    run this. 

    Parameters
    ----------
    R0_m : float, major radius on axis
    a0_m : float, minor radius
    P_sol_MW : float, power going into the SOL, in MW.
    B_p : float, poloidal field in T
    B_t : float, toroidal field in T
    q95 : float
    p_Pa_vol_avg : float, pressure in Pa units to use for the Brunner scaling.
    nu_m3 : float, upstream density in [m^-3], i.e. ne_sep.
    '''
    def __init__(self, R0_m, a0_m, P_sol_MW, B_p, B_t, q95, p_Pa_vol_avg, nu_m3, lam_q_model='Brunner'):
        
        self.R0 = R0_m  # m
        self.a0 = a0_m  # m
        self.P_sol_MW = P_sol_MW   # in MW
        self.B_p = B_p # T
        self.B_t = B_t # T
        self.q95 = q95

        # volume-averaged plasma pressure for Brunner scaling
        self.p_Pa_vol_avg = p_Pa_vol_avg

        # upstream (separatrix) density
        self.nu_m3 = nu_m3
        
        self.R_lcfs = self.R0+self.a0
        self.eps = self.a0/self.R0
        self.L_par = np.pi *self.R_lcfs * self.q95

        # coefficients for heat conduction by electrons or H ions
        self.k0_e = 2000.  # W m^{-1} eV^{7/2}
        self.k0_i = 60.    # W m^{-1} eV^{7/2}
        self.gamma = 7  # sheat heat flux transmission coeff (Stangeby tutorial)

        # lam_q in mm units from Eich NF 2013. This is only appropriate in H-mode
        self.lam_q_mm_eich = 1.35 * self.P_sol_MW**(-0.02)* self.R_lcfs**(0.04)* self.B_p**(-0.92) * self.eps**0.42

        # lam_q in mm units from Brunner NF 2018. This should be valid across all confinement regimes of C-Mod
        Cf = 0.08
        self.lam_q_mm_brunner = (Cf/self.p_Pa_vol_avg)**0.5 *1e3

        if lam_q_model == 'Brunner':
            self.lam_q_mm = self.lam_q_mm_brunner
        elif lam_q_model == 'Eich':
            self.lam_q_mm = self.lam_q_mm_eich
        else:
            raise ValueError('Undefined option for lambda_q model')


        # Parallel heat flux in MW/m^2.
        # Assumes all Psol via the outboard midplane, half transported by electrons (hence the factor of 0.5 in the front).
        # See Eq.(B.2) in Adam Kuang's thesis (Appendix B).
        self.q_par_MW_m2 = 0.5*self.P_sol_MW / (2.*np.pi*self.R_lcfs* (self.lam_q_mm*1e-3))*\
                           np.hypot(self.B_t,self.B_p)/self.B_p

        # Upstream temperature (LCFS) in eV. See Eq.(B.2) in Adam Kuang's thesis (Appendix B).
        # Note: k0_i gives much larger Tu than k0_e (the latter is right because electrons conduct)
        self.Tu_eV = ((7./2.) * (self.q_par_MW_m2*1e6) * self.L_par/(2.*self.k0_e))**(2./7.)

        # Upsteam temperature (LCFS) in K
        self.Tu_K = self.Tu_eV * q_electron/k_B

        # downstream density (rough...) - source?
        self.nt_m3= (self.nu_m3**3/((self.q_par_MW_m2*1e6)**2)) *\
                    (7.*self.q_par_MW_m2*self.L_par/(2.*self.k0_e))**(6./7.)*\
                    (self.gamma**2 * q_electron**2)/(4.*(2.*m_p))

        print('lambda_{q} (mm)'+' = {:.2f}'.format(self.lam_q_mm))
        print('T_{e,sep} (eV)'+' = {:.1f}'.format(self.Tu_eV))

