# Routines written by Francesco Sciortino and Andrés Miller

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

# more external dependencies
import sys
sys.path.append('/home/millerma')
import pysepest.pysepest as pysepest


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


def shift_profs(time_vec, r_vec, Te, Te_LCFS=75.0):
    '''
    Shift in x direction to match chosen temperature (in eV) at LCFS.
    '''
    x_of_TeSep =  np.zeros(len(time_vec))

    for ti, tt in enumerate(time_vec):

        x_of_TeSep[ti] = interp1d(Te[ti,:], r_vec, bounds_error=False,fill_value='extrapolate')(Te_LCFS)
        shift = 1 - x_of_TeSep[ti]

        if np.abs(shift > 0.05):
            print('Cannot determine accurate shift')
            x_of_TeSep = 1 # no shift - probe data probably not good

    return x_of_TeSep


def find_separatrix(res, fit_type='log_linear', delta_T_threshold=None, plot=False, plot_ind=None):

    
    '''

    Routine to find separatrix using lambda_T rather than lambda_q from a scaling

    '''

    # begin by deciding to exclude points below certain values #
    Te_gt_floor = (res['Te_raw'] > 20)
    ne_gt_floor = (res['ne_raw'] > 1e12)
    #pe_gt_floor = (res['pe_raw'] > 1.602)

    # assing the values from the dictionary to more easily read variables
    Te = res['Te_raw'][Te_gt_floor]
    Te_unc = res['Te_raw_unc'][Te_gt_floor]
    R_Te = res['Te_R'][Te_gt_floor] # have individual coordinates for ne and Te in case one point was filtered for one and not the other

    ne = res['ne_raw'][ne_gt_floor]/1e14
    ne_unc = res['ne_raw_unc'][ne_gt_floor]/1e14
    R_ne = res['ne_R'][ne_gt_floor] # same as above

    R_brun = res['R_sep'] # this is the R-coordinate of the separatrix according to EFIT (magnetics)

    # will ignore next block of code for now, later we will:
    #    -also fit the pressure, pe
    #    -add the option to take in a previously computed fit (denoted by the _prof variables)

    '''

    pe = res['pe_raw'][pe_gt_floor]
    pe_unc =  res['pe_raw_unc'][pe_gt_floor]
     pe_unc = (res['pe_raw'][db]*np.sqrt((res['Te_raw_unc'][db]/res['Te_raw'][db])**2 + (res['ne_raw_unc'][db]/res['ne_raw'][db])**2))[pe_gt_floor] 
    R_pe = res['pe_R'][pe_gt_floor]

    R_fit = res['R_prof']
    Te_fit = res['Te_prof']
    ne_fit = res['ne_prof']/1e14 # cm^{-3} --> 10^{20} m^{-3} 
    pe_fit = res['pe_prof']

    grad_Te_fit = res['grad_Te_prof']
    grad_ne_fit = res['grad_ne_prof']/1e14 # cm^{-4} --> 10^{20} m^{-4} 
    grad_pe_fit = res['grad_pe_prof']

    # need uncertainties as well
    Te_fit_unc = res['Te_prof_unc']
    ne_fit_unc = res['ne_prof_unc']/1e14 # cm^{-3} --> 10^{20} m^{-3} 
    pe_fit_unc = res['pe_prof_unc']

    grad_Te_fit_unc = res['grad_Te_prof_unc']
    grad_ne_fit_unc = res['grad_ne_prof_unc']/1e14 # cm^{-4} --> 10^{20} m^{-4} 
    grad_pe_fit_unc = res['grad_pe_prof_unc']

    Te_brun = postres['Te_ped'][db,2]*1e3 # keV --> eV
    ne_brun = postres['ne_ped'][db,2]
    pe_brun = postres['pe_ped'][db,2]

    '''

    #######################
        
    ### FIT DEFINITIONS ###

    #######################

    def linear_fit_lstsq(x, ylog, ylog_err):

        # Linear least squares fit


        Alog_mat = np.vstack([x, np.ones(len(x))]).T
        Wlog_mat = np.sqrt(np.diag(np.abs(1/ylog_err)))
        Aw = np.dot(Wlog_mat, Alog_mat)
        yw = np.dot(ylog, Wlog_mat)

        popt_log = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        
        # found these lines online but I don't really trust them
        rss = np.sum((np.exp(ylog) - np.exp(popt_log[1] + x*popt_log[0]))**2)
        cov_mat = np.linalg.inv(np.dot(Alog_mat.T, Alog_mat))*rss
        lam_unc = np.sqrt(cov_mat[0,0])

        lam = -1/popt_log[0]

        # calculate error in y-intercept accoriding to formula from wikipedia - not sure how we'd include the yerr but whatever
        b_unc = np.sqrt(np.sum((ylog - (popt_log[1] + x*popt_log[0]))**2)/(len(ylog)-2) / np.sum((x - np.mean(x))**2))
        lam_unc = lam*(b_unc/popt_log[0])**2

        return lam, lam_unc, popt_log


    # this is where the rest of the fitting functions will go:
    # to code these up I would recommend using the scipy.optimize.curve_fit function

    def linear_fit():

        # do the fitting and calculate lam = -y / grad y
    
        return lam, lam_unc, popt

    
    def log_linear_fit():

        # do the fitting and calculate lam = -y / grad y
    
        return lam, lam_unc, popt


    def quadratic_fit():
   
        # do the fitting and calculate lam = -y / grad y
 
        return lam, lam_unc, popt

    
    def log_quadratic_fit():
   
        # do the fitting and calculate lam = -y / grad y
 
        return lam, lam_unc, popt
    
    # may want to add more fitting functions later, like a generalized polynomial or something else



    #####################

    ### BEGIN FITTING ###

    #####################


    ### Begin with Te fit ###

    # choose which fit_type you would like to use
    if fit_type == 'log_linear':

        # define fit interval
        fit_interval = 5e-3
        fit_range = [R_brun-fit_interval, R_brun+fit_interval]
        xx = np.linspace(fit_range[0], fit_range[1], 100)

        Te_mask_fit = (R_Te > fit_range[0]) & (R_Te < fit_range[1])
        
        # this takes the log of the data, but this could be done in the fitting function instead
        xlog_tofit = R_Te[Te_mask_fit]
        ylog_tofit = np.log(Te[Te_mask_fit])
        yerrlog_tofit = Te_unc[Te_mask_fit]/Te[Te_mask_fit]


        # try to fit the data - if it fails for some reason, return nans
        try:
   
            lam_T, lam_T_unc, popt_log = linear_fit_lstsq(xlog_tofit, ylog_tofit, yerrlog_tofit)    

            # popt_log should be the coefficients of the line fit to the log of the data
            # will need to transform it back to Te by using exp(b + a*x)
            yfit = np.exp(popt_log[1] + xx*popt_log[0])
            dyfit = np.gradient(yfit, xx)

        except:
            print('Could not fit edge Te to an exponential')
            R_sep = np.nan
            Te_sep = np.nan
            lambda_Te = np.nan
            lambda_Te_unc = np.nan


    # do an elif for the other fits
    # here you should do any manipulation needed to the fitting interval as above and then make a call to the
    # functions you defined above

    elif fit_type == 'quadratic':
        
        fit_interval = 5e-3
        fit_range = [R_brun-fit_interval, R_brun+fit_interval]
        xx = np.linspace(fit_range[0], fit_range[1], 100)

        Te_mask_fit = (R_Te > fit_range[0]) & (R_Te < fit_range[1])
        
        x_tofit = R_Te[Te_mask_fit]
        y_tofit = Te[Te_mask_fit]
        yerr_tofit = Te_unc[Te_mask_fit]

        # this is where you'll want to use the "cleaned" data to pass it into the fit
        lam_T, lam_T_unc, popt = some_other_fit(x_tofit, y_tofit, yerr_tofit)

    # can ignore this for now
    elif fit_type == 'tanh':

        xx = R_fit
        yfit = Te_fit
        dyfit = grad_Te_fit

    else:
        print('Fit type not recognized')


    # now, using lambda_T that you've computed, find the separatrix
    # this will use a module called pysepest which performs the calculation from the lambda_T
    # and all the other parameters from the shot that you've provided in the res dictionary
    # if the algorithm doesn't converge, return nans - may need to dig into why this is happening later

    try:
        data_at_sep = single_pysepest(res, xx, yfit, dyfit)
        print('Found separatrix at {:.3f} m with Te = {:.1f} eV'.format(data_at_sep['x_sep'], data_at_sep['Te']))
    except:
        print('2-point model separatrix estimation failed using Te fit')
        return np.nan, np.nan
        #return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan # real elegant, I know

    # can choose to continue to fit iteratively until threshold is met
    # this next block of code allows one to fit iteratively until a certain criterion is met
    # essentially fit once, find the separatrix, fit again around the new separatrix and so on and so forth
    # for now, we'll ignore this and see if it makes sense to include later

    if delta_T_threshold is not None:

        # use delta_T_threshold = 200 to generate lambda_p scaling
        # use delta_T_threshold = 2 for the rest (esp. nesep vs tesep plots)    

        try:
            old_R = res['R_sep'][db]
            new_R = data_at_sep['x_sep']

            temp_diff = 200 # placeholder to enter the loop
            loop = 0 # limit on number of loops to use

            alpha = 0 # relaxation for iteration so that jumps are not too large
            ped_factor = 1 # factor to determine how much of interval inward of the "separatrix" should be kept
            while temp_diff > delta_T_threshold and ~np.isnan(data_at_sep['x_sep']) and loop < 10:

                fit_interval = 5e-3 # hard coded to be EFIT interval - could choose something different
                fit_range_new = [alpha*old_R + (1-alpha)*new_R - fit_interval*ped_factor, alpha*old_R + (1-alpha)*new_R + fit_interval]

                xx_new = np.linspace(fit_range_new[0], fit_range_new[1], 100)

                Te_mask_fit_new = (R_Te > fit_range_new[0]) & (R_Te < fit_range_new[1])
            
                xlog_tofit_new = R_Te[Te_mask_fit_new]
                ylog_tofit_new = np.log(Te[Te_mask_fit_new])
                yerrlog_tofit_new = Te_unc[Te_mask_fit_new]/Te[Te_mask_fit_new]

                lam_T_new, lam_T_unc_new, popt_log_new = linear_fit_lstsq(xlog_tofit_new, ylog_tofit_new, yerrlog_tofit_new)    

                yfit_new = np.exp(popt_log_new[1] + xx*popt_log_new[0])
                dyfit_new = np.gradient(yfit_new, xx_new)

                # data_at_sep_new = single_pysepest(res, db, data_at_sep['x_sep'], xx_new, yfit_new, dyfit_new)
                data_at_sep_new = single_pysepest(res, db, alpha*old_R + (1-alpha)*new_R, xx_new, yfit_new, dyfit_new)

                temp_diff = np.abs(data_at_sep['Te'] - data_at_sep_new['Te'])

                data_at_sep = data_at_sep_new
                lam_T = lam_T_new
                lam_T_unc = lam_T_unc_new
                fit_range = fit_range_new
                xx = xx_new

                old_R = alpha*old_R + (1-alpha)*new_R
                new_R = data_at_sep_new['x_sep']

                loop += 1

                # print(db, temp_diff)

        except:
            print('During loop, could not fit edge Te to an exponential for ind {}'.format(db))
            R_sep = np.nan
            Te_sep = np.nan
            lambda_Te = np.nan
            lambda_Te_unc = np.nan

    # save the data if it's physical (shouldn't really be more than 200 eV and can't be negative)

    if (data_at_sep['Te'] < 200) & (data_at_sep['Te'] > 0):

        R_sep = data_at_sep['x_sep']
        Te_sep = data_at_sep['Te']

        # interpolate the lambda_T profile we get out to find the actual value at the separatrix

        if fit_type == 'log_linear':

            lambda_Te = lam_T
            lambda_Te_unc = lam_T_unc
    
        # will also want to add any other fit_types to here using the syntax for the tanh since
        # the lambda_T will be a whole array if you do other types of fits


        elif fit_type == 'tanh':
                
            lam_T_fit = -yfit/dyfit
            lam_T_unc_fit = -yfit/dyfit*np.sqrt((Te_fit_unc/Te_fit)**2 + (grad_Te_fit_unc/grad_Te_fit)**2)

            lambda_Te = interp1d(xx, lam_T_fit)(data_at_sep['x_sep'])
            lambda_Te_unc = interp1d(xx, lam_T_unc_fit)(data_at_sep['x_sep'])

    else:
        print('Separatrix estimation failed somewhere for index {}'.format(db))

        R_sep = np.nan
        Te_sep = np.nan
        lambda_Te = np.nan
        lambda_Te_unc = np.nan


    ### Proceed with ne fit ###

    # now that we've identified the separatrix, use that for ne and pe
    # for now, this part won't matter too much since we're only returning Te,sep and R,sep
    # we might later choose to also return nesep from here, so might as well compute it

    # ne - just like Te
    if fit_type == 'log_linear':

        fit_interval = 5e-3
        fit_range = [R_brun-fit_interval, R_brun+fit_interval]
        xx = np.linspace(fit_range[0], fit_range[1], 100)

        ne_mask_fit = (R_ne > fit_range[0]) & (R_ne < fit_range[1])

        xlog_tofit = R_ne[ne_mask_fit]
        ylog_tofit = np.log(ne[ne_mask_fit])
        yerrlog_tofit = ne_unc[ne_mask_fit]/ne[ne_mask_fit]

        try:

            lam_n, lam_n_unc, ne_popt_log = linear_fit_lstsq(xlog_tofit, ylog_tofit, yerrlog_tofit)    
            ne_sep = np.exp(ne_popt_log[1] + data_at_sep['x_sep']*ne_popt_log[0])

        except:
            print('Could not fit edge ne to an exponential')
            ne_sep = np.nan
            lambda_ne = np.nan
            lambda_ne_unc = np.nan

    elif fit_type == 'tanh':

        # just do a simple interpolation of the fits
        xx = R_fit
        yfit = ne_fit
        dyfit = grad_ne_fit

        lam_n_unc_fit = -yfit/dyfit*np.sqrt((ne_fit_unc/ne_fit)**2 + (grad_ne_fit_unc/grad_ne_fit)**2)

        ne_sep = interp1d(xx, yfit)(data_at_sep['x_sep'])
        lam_n = interp1d(xx, -yfit/dyfit)(data_at_sep['x_sep'])
        lam_n_unc = interp1d(xx, lam_n_unc_fit)(data_at_sep['x_sep'])

    else:
        print('Fit type not recognized')

    # check for physicality of result - should probably add some stuff in about gradients here
    if (ne_sep < 10) & (ne_sep > 0):
        lambda_ne = lam_n
        lambda_ne_unc = lam_n_unc

    else:
        print('ne fit values are unphysical')
        ne_sep = np.nan
        lambda_ne = np.nan
        lambda_ne_unc = np.nan

    # for now will skip the pressure calculation #
    '''
    # pe
    if fit_type == 'log_linear':

        fit_interval = 5e-3
        fit_range = [R_brun-fit_interval, R_brun+fit_interval]
        xx = np.linspace(fit_range[0], fit_range[1], 100)

        pe_mask_fit = (R_pe > fit_range[0]) & (R_pe < fit_range[1])

        xlog_tofit = R_pe[pe_mask_fit]
        ylog_tofit = np.log(pe[pe_mask_fit])
        yerrlog_tofit = pe_unc[pe_mask_fit]/pe[pe_mask_fit]

        try:

            lam_p, lam_p_unc, pe_popt_log = linear_fit_lstsq(xlog_tofit, ylog_tofit, yerrlog_tofit)    
            pe_sep = np.exp(pe_popt_log[1] + data_at_sep['x_sep']*pe_popt_log[0])

        except:
            print('Could not fit edge pe to an exponential for ind {}')
            pe_sep = np.nan
            lambda_pe = np.nan
            lambda_pe_unc = np.nan

    elif fit_type == 'tanh':

        # just do a simple interpolation of the fits
        xx = R_fit
        yfit = pe_fit
        dyfit = grad_pe_fit

        lam_p_unc_fit = -yfit/dyfit*np.sqrt((pe_fit_unc/pe_fit)**2 + (grad_pe_fit_unc/grad_pe_fit)**2)

        pe_sep = interp1d(xx, yfit)(data_at_sep['x_sep'])
        lam_p = interp1d(xx, -yfit/dyfit)(data_at_sep['x_sep'])
        lam_p_unc = interp1d(xx, lam_n_unc_fit)(data_at_sep['x_sep'])

    else:
        print('Fit type not recognized')

    # check for physicality of result - should probably add some stuff in about gradients here
    if (pe_sep < 10e3) & (pe_sep > 0):
        lambda_pe = lam_p
        lambda_pe_unc = lam_p_unc

    else:
        print('pe fit values are unphysical')
        pe_sep = np.nan
        lambda_pe = np.nan
        lambda_pe_unc = np.nan

    '''

    ### PLOTTING ###

    if plot:

        # Te
        fig,ax = plt.subplots(2, sharex=True)

        ax[0].errorbar(R_Te, Te, yerr=Te_unc, fmt='o', alpha=0.5, mfc='w')
        #ax[0].plot(R_fit, Te_fit, lw=2, linestyle='--')
        #ax[0].plot(R_brun, Te_brun, 'X', markersize=12, c='C3', markeredgecolor='k', markeredgewidth=1)
        ax[0].axvline(R_brun, linestyle='--', c='C3')
        ax[0].axvline(R_brun-fit_interval, linestyle='-.', c='C3')
        ax[0].axvline(R_brun+fit_interval, linestyle='-.', c='C3')

        ax[0].errorbar(R_Te[Te_mask_fit], Te[Te_mask_fit], yerr=Te_unc[Te_mask_fit], fmt='o', c='C3', alpha=0.75, mfc='w',zorder=1)
    
        ax[0].plot(xx, np.exp(popt_log[1] + xx*popt_log[0]), lw=5, c='C4', zorder=0)
        ax[0].plot(data_at_sep['x_sep'], data_at_sep['Te'], 'X', markersize=12, c='C4', markeredgecolor='k', markeredgewidth=1,zorder=3)
        ax[0].axvline(data_at_sep['x_sep'], linestyle='--', c='C4')

        ax[0].tick_params(axis='both', labelsize=14)
        ax[0].set_ylabel('$T_{e} (eV)$', fontsize=14)

        ax[0].set_yscale('log')

        # ne
        ax[1].errorbar(R_ne, ne, yerr=ne_unc, fmt='o', alpha=0.5, mfc='w')
        #ax[1].plot(R_fit, ne_fit, lw=2, linestyle='--')
        #ax[1].plot(R_brun, ne_brun, 'X', markersize=12, c='C3', markeredgecolor='k', markeredgewidth=1)
        ax[1].axvline(R_brun, linestyle='--', c='C3')#, label='$\\lambda_{q}^{EFIT}$ sep')
        ax[1].axvline(R_brun-fit_interval, linestyle='-.', c='C3')
        ax[1].axvline(R_brun+fit_interval, linestyle='-.', c='C3')
        #ax[1].axhline(res['nebar'][db]/1e20, linestyle=':', c='k', label='$\\overline{n}_{e}$')
        #ax[1].axhline(res['n_m3_avg'][db], linestyle=':', c='C2', label='$<n_{e}>$')

        ax[1].errorbar(R_ne[ne_mask_fit], ne[ne_mask_fit], yerr=ne_unc[ne_mask_fit], fmt='o', c='C3', alpha=0.75, mfc='w',zorder=1)

        ax[1].plot(xx, np.exp(ne_popt_log[1] + xx*ne_popt_log[0]), lw=5, c='C4', zorder=0)
        ax[1].plot(data_at_sep['x_sep'], ne_sep, 'X', markersize=12, c='C4', markeredgecolor='k', markeredgewidth=1,zorder=2)
        ax[1].axvline(data_at_sep['x_sep'], linestyle='--', c='C4')#, label='$\\lambda_{T}$ sep')

        ax[1].tick_params(axis='both', labelsize=14)
        ax[1].set_ylabel('$n_{e} (10^{20} m^{-3})$', fontsize=14)
        #ax[1].legend(loc='best', fontsize=14)

        ax[1].set_yscale('log')
        
        ax[1].set_xlabel('$R (m)$', fontsize=14)
        ax[1].set_xlim([0.865, 0.905])

        plt.show()

    return Te_sep, 2/7*lambda_Te
    #return R_sep, ne_sep, lambda_ne, lambda_ne_unc, Te_sep, lambda_Te, lambda_Te_unc, pe_sep, lambda_pe, lambda_pe_unc


def single_pysepest(res, xx, yfit, dyfit, bound=1e-3):

    model = pysepest.models.conduction_limited_local_lam_q

    Zeff = 1.4
    q_dist = 1
    P_to_e = 0.6
    P_to_LFS = 0.55
    L_par = np.pi*res['R_sep']*res['q95']
    P_e_SOL = res['P_net']*1e6*P_to_e*P_to_LFS

    scalars = dict(
        R = res['R_sep'],
        B_pol = res['Bp'],
        B_tor = res['Bt'],
        P_e_SOL = P_e_SOL,
        Zeff = Zeff,
        q_distribution = q_dist,
        L_par = L_par,
        )

    profiles = dict()

    profiles['Te'] = interp1d(xx, yfit)
    profiles['lam_Te'] = interp1d(xx, -yfit/dyfit)

    x0 = res['R_sep'] - bound
    x1 = res['R_sep'] + bound

    optimizer_options = dict(
                        optimizer = pysepest.optimizers.root_scalar,
                        x0 = x0, x1 = x1,
                        )

    # optimizer_options = dict(
    #         optimizer=pysepest.optimizers.minimize_scalar_L2,
    #         method='bounded', bounds=(x0, x1))

    data_at_sep, optim_res = pysepest.find_sep(scalars, profiles, model, optimizer_options)

    return data_at_sep


