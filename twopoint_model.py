# Routine written by Francesco Sciortino

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import m_p, e as q_electron, Boltzmann as k_B
from scipy.interpolate import interp1d
import pickle as pkl, os

from scipy.constants import Boltzmann as kB, e as q_electron
import aurora

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

