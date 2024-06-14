'''
Methods to load and plot C-Mod Ly-alpha data.

sciortino, August 2020

Modified by Andres Miller
'''
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import xarray
from scipy.interpolate import interp1d, interp2d
#sys.path.append('/home/millerma/OMFIT-source/omfit')
from omfit_classes import omfit_eqdsk, omfit_mds
import shutil, os, scipy, copy
from IPython import embed
import MDSplus
from omfit_classes.omfit_mds import OMFITmdsValue

from scipy.constants import Boltzmann as kB, e as q_electron
from scipy.optimize import curve_fit
import aurora
import sys
sys.path.append('/home/sciortino/usr/python3modules/eqtools3')
sys.path.append('/home/sciortino/usr/python3modules/gptools3')
#sys.path.append('/home/millerma/python3modules/profiletools3')
#sys.path.append('/home/millerma/python3modules/eqtools3')
sys.path.append('/home/millerma/usr/profiletools3')
import profiletools
import eqtools

# from mitlya repo
import fit_profiles
import fit_2D
#import tomographic_inversion as tomo

# for eq
sys.path.append('/home/millerma/poloidalFieldGetter')
from getGfileDict import getGfileDict as gfd


# PFS this is the function that grabs the data and fits it - probably too big of a function and should break it down into chunks
def get_cmod_kin_profs(shot, tmin, tmax, geqdsk=None, pre_shift_TS=False, force_to_zero=False, frac_err=True, num_mc=100,
                       probes=['A'], osborne_fit=False, apply_final_sep_stretch=False, core_ts_mult=False, edge_ts_mult=False, core_ts_factor=1, edge_ts_factor=1):
    '''Function to load and fit modified-tanh functions to C-Mod ne and Te.

    This function is designed to be robust for operation within the construction of
    the C-Mod Ly-a database. It makes use of the profiletools package just to conveniently
    collect data and time-average it. An updated profiletools (python 3+) is available 
    from one of the github repos of sciortinof.

    Parameters
    ----------
    shot : int
        CMOD shot number
    tmin, tmax : floats
        Times in seconds for window interval of interest
    geqdsk : dict
        Dictionary containing processed EFIT geqdsk file
    pre_shift_TS : bool, opt
        If True, Thomson Scattering (TS) data are shifted based on the 2-point model after being fitted 
        by themselves. This is recommended only if TS has good coverage inside and outside of the LCFS
        and the LCFS is therefore reasonably covered by experimental data.
    force_to_zero : bool, opt
        If True, add fictitious points far into the SOL to enforce that both ne and Te must go towards
        zero at the wall.
    frac_err : bool, opt
        If True, error of fitted profiles will simply be 20% of the value of the profile.
        If False, a boostrapping technique with {num_mc} iterations will be run and the standard deviation
        of the distribution of the fits will be the error of the fit.
    num_mc: int, opt
        Number of Monte Carlo iterations of bootstrapping error estimation.
    probes : list of str
        List of strings indicating data from which probes should be loaded if possible. 
        Possible options are 'A' (ASP probe) and 'F' (FSP probe).
    osborne_fit : bool
        If True, use the Osborne mtanh fit, otherwise use the version modified by FS.
    apply_final_sep_stretch : bool
        If True, final kinetic profiles are stretched such that Te_sep from the 2-point model is matched.

    Returns
    -------
    f_ne : kinetic_profile object
        Fitted electron density object containing fitted profile and the distribution of fits from error estimation
    f_Te : kinetic_profile object
        Fitted electron temperature object containing fitted profile and the distribution of fits from error estimation
    f_pe : kinetic_profile object
        Fitted electron pressure object containing fitted profile and the distribution of fits from error estimation

    Some particularly useful attributes of these objects are:
        .x : 1D array
            Coordinates of raw data - currently uses rhop = sqrt(psinorm) units
        .y : 1D array
            Data values in the following units ne: :math: 10^{20} m^{-3}
                                               Te: :math: keV
                                               pe: :math: kPa
        .grad_y : 1D array
            Gradients in data values with respect to R in same units as .y attribute multiplied by :math: m^{-1}
        .fit_dist_prefilt: 2D array
            Distribution of fits computed by perturbing raw points randomly according to their experimental 
            uncertainties
        .grad_fit_dist_prefilt : 2D array
            Distribution of gradients of fits computed by perturbing raw points randomly according to their 
            experimental uncertainties
        .fit_dist : 2D array
            Distribution of fits computed by perturbing raw points randomly according to their experimental 
            uncertainties, with filtering performed for points with large chi squared values
        .grad_fit_dist : 2D array
            Distribution of gradients of fits computed by perturbing raw points randomly according to their experimental 
            uncertainties, with filtering performed for points with large chi squared values
        .fit_mean : 1D array
            Mean of .fit_dist_prefilt computed along 2nd axis
        .fit_std : 1D array
            Standard deviation of .fit_dist_prefilt computed along 2nd axis
        .grad_fit_mean : 1D array
            Mean of .grad_fit_dist_prefilt computed along 2nd axis
        .grad_fit_std : 1D array
            Standard deviation of .grad_fit_dist_prefilt computed along 2nd axis
        .popt : 2D array
            Array of coefficients of mtanh fit containing the coefficients of the original fit and the mean of the 
            distribution of fits

    p_ne : profiletools object
        Electron density object containing experimental data from all loaded diagnostics.
    p_Te : profiletools object
        Electron temperature object containing experimental data from all loaded diagnostics.
    p_pe : profiletools object
        Electron pressure object containing experimental data from all loaded diagnostics.
        
    Some particularly useful attributes of these objects are:
        .X[:,0] : 1D array
            Coordinates of raw data - currently uses rhop = sqrt(psinorm) units
        .y : 1D array
            Data values in the following units ne: :math: 10^{20} m^{-3}
                                               Te: :math: keV
                                               pe: :math: kPa
        .y_err : 1D array
            Error in data values in same units as .y attribute

    num_SP : int
        Number of points that belong to scanning probe diagnostic
        Can be used to index these since they are added at the end of the profiletools object, e.g. p_ne.y[-num_SP:]

    '''
   
    # again, not sure if we actually need the geqdsk here 
    if geqdsk is None:
        # the geqdsk is used only when using the Aurora radial coordinate transformations
        geqdsk = get_geqdsk_cmod(
            shot, (tmin+tmax)/2.*1e3, gfiles_loc = '/home/sciortino/EFIT/lya_gfiles/')

    # number of iterations for osborne fit to converge
    maxfev = 2000

    # identify mode to decide how to fit profile - empirical observation
    #mode = identify_mode(int(shot))
    mode = 'H'
    ped_width = 0.04 if mode == 'L' else 0.02 # will later be used to fit

    # PFS this should be all the equilibrium information that is needed
    try: # EFIT20 only exists for shots from certain years
        e = eqtools.CModEFITTree(int(shot), tree='EFIT20', length_unit='m')
    except:
        e = eqtools.CModEFITTree(int(shot), tree='analysis', length_unit='m')
                
    if (core_ts_mult & (core_ts_factor == 1)) or (edge_ts_mult & (edge_ts_factor == 1)):
        ts_factor = get_ts_tci_ratio(shot, tmin, tmax, plot=False)
        print('Computed factor of {:.2f} as ratio between TCI and TS'.format(ts_factor)) 

    try:
        # require edge Thomson to be available
        p_Te= profiletools.Te(int(shot), include=['ETS'], abscissa='sqrtpsinorm',t_min=tmin,t_max=tmax,efit_tree=e, remove_zeros=True)
        p_ne= profiletools.ne(int(shot), include=['ETS'], abscissa='sqrtpsinorm',t_min=tmin,t_max=tmax,efit_tree=e, remove_zeros=True)

        # multiply using nebar calibration if flag is trned on
        if edge_ts_mult:
            if edge_ts_factor != 1: ts_factor = edge_ts_factor
            p_ne.y *= ts_factor
            p_ne.err_y *= ts_factor
            print('Multiplying ETS by factor of {:.2f}'.format(ts_factor)) 

    except MDSplus.TreeNODATA:
        raise ValueError('No edge Thomson data!')
  
    try:
        equal_R = p_ne.X[:,1] == p_Te.X[:,1]
        assert np.sum(equal_R) == len(p_ne.X[:,1])
    except:
        raise ValueError('Edge Thomson rhobase differs between ne and Te')
    
    try:
        # try to add core Thomson, not strictly necessary 
        p_Te_CTS = profiletools.Te(int(shot), include=['CTS'], efit_tree=e,
                                   abscissa='sqrtpsinorm',t_min=tmin,t_max=tmax)
        p_ne_CTS = profiletools.ne(int(shot), include=['CTS'], efit_tree=e,
                                   abscissa='sqrtpsinorm',t_min=tmin,t_max=tmax)

        # multiply using nebar calibration if flag is trned on
        if core_ts_mult:
            if core_ts_factor != 1: ts_factor = core_ts_factor
            p_ne_CTS.y *= ts_factor
            p_ne_CTS.err_y *= ts_factor
            print('Multiplying CTS by factor of {:.2f}'.format(ts_factor)) 

        p_Te.add_profile(p_Te_CTS)
        p_ne.add_profile(p_ne_CTS)
    except Exception:
        pass
  
   

    # consider only flux surface on which points were measured, regardless of LFS or HFS
    p_Te.X=np.abs(p_Te.X)
    p_ne.X=np.abs(p_ne.X)


    # set some minimum uncertainties. Recall that units in objects are 1e20m^{-3} and keV
    p_ne.y[p_ne.y<=0.] = 0.01  # 10^18 m^-3
    p_Te.y[p_Te.y<=0.01] = 0.01 # 10 eV
    p_ne.err_y[p_ne.err_y<=0.01] = 0.01 # 10^18 m^-3
    p_Te.err_y[p_Te.err_y<=0.02] = 0.02 # 20 eV
    
    # PFS this removes the time-axis - might not want to do this if interested in temporal fitting
    p_ne.drop_axis(0)
    p_Te.drop_axis(0)

    ### save the raw points before points are filtered out

    ne_X_nofilters = p_ne.X[:,0]
    ne_Y_nofilters = p_ne.y
    ne_unc_Y_nofilters = p_ne.err_y
    
    Te_X_nofilters = p_Te.X[:,0]
    Te_Y_nofilters = p_Te.y
    Te_unc_Y_nofilters = p_Te.err_y
   
    # cleanup of low Te values
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]<1.03, p_Te.y<0.015))  # TS Te should be >15 eV inside near SOL
    
    def convert_rhop_to_R(rhop, geqdsk):

        # map kps and emiss to midplane
        from scipy.interpolate import UnivariateSpline
        mp_ind = np.where(geqdsk['AuxQuantities']['Z'] == 0)[0][0]
        R_mp = geqdsk['AuxQuantities']['R']
        rhop_mp = geqdsk['AuxQuantities']['RHOpRZ'][mp_ind]
        R0 = geqdsk['RMAXIS']
        omp = R_mp > R0
        rhop_to_R = UnivariateSpline(rhop_mp[omp], R_mp[omp])

        return rhop_to_R(rhop)

    #########################

    ### choose how to deal with separatrix


    efit_sep = True # this will not shift profiles at all
    brunner_scaling = True # this is to decide whether to use brunner scaling or profile lambda_T
 

    if brunner_scaling:
    
        Te_sep_eV, lam_q_mm = fit_2D.Teu_2pt_model(shot, tmin, tmax, geqdsk, pressure_opt = 3, lambdaq_opt=1)
        print('Brunner Te LCFS eV', Te_sep_eV)

    else:

        # will need to make a dictionary with parameters from shot that are needed in 2pt model
        sep_dict = {} # could just do this inside the find_separatrix function if preferred
    
        sep_dict['ne_R'] = convert_rhop_to_R(p_ne.X[:,0],geqdsk)
        sep_dict['Te_R'] = convert_rhop_to_R(p_Te.X[:,0],geqdsk)
        #sep_dict['pe_R'] = convert_rhop_to_R(p_pe.X[:,0],geqdsk) ignore pressure for now

        sep_dict['ne_raw'] = p_ne.y*1e14
        sep_dict['Te_raw'] = p_Te.y*1e3
        #sep_dict['pe_raw'] = p_pe.y
        
        sep_dict['ne_raw_unc'] = p_ne.err_y*1e14
        sep_dict['Te_raw_unc'] = p_Te.err_y*1e3
        #sep_dict['pe_raw_unc'] = p_pe.y
       
        # will not fill in any of the _prof values for now - can only do this after fitting with the tanh (or some other function)

        sep_dict['R_sep'] = aurora.rad_coord_transform(1.0, 'r/a', 'Rmid', geqdsk)

        t,ddata = get_CMOD_var(var='q95', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            q95 = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            q95 = np.nan
        sep_dict['q95'] = q95

        t,ddata = get_CMOD_var(var='Bp', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            Bp = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            Bp = np.nan
        sep_dict['Bp'] = Bp

        t,ddata = get_CMOD_var(var='Bt', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            Bt = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            Bt = np.nan
        sep_dict['Bt'] = Bt

        # calculate P_net

        t,ddata = get_CMOD_var(var='P_oh', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            P_oh = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            P_oh = np.nan

        t,ddata = get_CMOD_var(var='P_RF', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            P_RF = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            P_RF = np.nan
            
        ddata = get_CMOD_var(var='dWdt', shot=shot, tmin=tmin, tmax=tmax, return_time=False)
        dWdt = np.mean(ddata)
    
        t,ddata = get_CMOD_var(var='P_rad_main', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            P_rad_main = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            P_rad_main = np.nan

        t,ddata = get_CMOD_var(var='P_rad_diode', shot=shot, return_time=True)
        if ddata is not None and np.any(~np.isnan(ddata)):
            P_rad_diode = np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
        else:
            P_rad_diode = np.nan

        P_rad = P_rad_diode if np.isnan(P_rad_main) else P_rad_main

        sep_dict['P_net'] = P_oh + P_RF - dWdt - P_rad

        import find_separatrix as fp
        Te_sep_eV, lam_q_mm = fp.find_separatrix(sep_dict, fit_type='log_linear',plot=True)


    #########################

    ### now begin fitting ###

    #########################

    
    pre_shift_TS = True # this will shift the TS data according to Tesep (before any probe data if it exists) - dont think this is necessary - may want to consider again if we want to use probe data at some point
    rhop_kp = np.linspace(0.0, 1.1, 1000)
   
    # these are used to determine which points belong to TS vs. probes 
    ne_X_before = p_ne.X[:,0]
    Te_X_before = p_Te.X[:,0]

    reg = [8] # these will determine which coefficients in the fit are set to zero to prevent overfitting
    edge_chi = True # this will use chisqr from the edge instead of the whole profile to evaluate goodness of fit

    # begin the fitting
    if pre_shift_TS:

        # filter out some points with large uncetainties before the fit
        p_ne, p_Te = prefit_filter(p_ne, p_Te)
    
        if force_to_zero:
            # force fits to go down in the far SOL (r/a=1.2) to make the routine more robust
            p_Te.add_data( np.array([[1.05]]), np.array([5e-3]), err_X=np.array([0.001]), err_y=np.array([0.001]))
            p_ne.add_data( np.array([[1.1]]), np.array([0.1]), err_X=np.array([0.001]), err_y=np.array([0.001]))

        # need to fit TS first 
        
        idxs_ne = np.argsort(p_ne.X[:,0])
        idxs_Te = np.argsort(p_Te.X[:,0])

        # this was used in the past to only fit some portion of the profile - may not be needed anymore
        min_fit = 0
        max_fit = 0

        mask_ne = p_ne.X[idxs_ne,0] > min_fit
        mask_Te = p_Te.X[idxs_Te,0] > min_fit   

        # not sure why this is commented out, but in theory one should be able to specify
        # a max and a min to restrict the fit range

        #mask_ne = np.logical_or(p_ne.X[idxs_ne,0] > min_fit, p_ne.X[idxs_ne,0] < max_fit)
        #mask_Te = np.logical_or(p_Te.X[idxs_Te,0] > min_fit, p_Te.X[idxs_Te,0] < max_fit)

        # fit the profiles!        
        maxfev = 2000 # number of iterations for fit to converge
    
        ne, ne_popt, ne_perr, ne_chisqr = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te, Te_popt, Te_perr, Te_chisqr = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=False)

        # apply filter depending on initial fit - in short, excludes points far from the fit
        p_ne, p_Te = postfit_filter(p_ne, p_Te, ne, Te, rhop_kp)

        if force_to_zero:
            # add extra points in the far SOL again to force fits to go down
            p_Te.add_data( np.array([[1.05]]), np.array([5e-3]), err_X=np.array([0.001]), err_y=np.array([0.001]))
            p_ne.add_data( np.array([[1.1]]), np.array([0.1]), err_X=np.array([0.001]), err_y=np.array([0.001]))

        # fit them again after having filtered for outliers! - not sure if this actually does anything
        idxs_ne = np.argsort(p_ne.X[:,0])
        idxs_Te = np.argsort(p_Te.X[:,0])

        mask_ne = p_ne.X[idxs_ne,0] > min_fit
        mask_Te = p_Te.X[idxs_Te,0] > min_fit   
        #mask_ne = np.logical_or(p_ne.X[idxs_ne,0] > min_fit, p_ne.X[idxs_ne,0] < max_fit)
        #mask_Te = np.logical_or(p_Te.X[idxs_Te,0] > min_fit, p_Te.X[idxs_Te,0] < max_fit)
        
        ne, ne_popt, ne_perr, ne_chisqr = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te, Te_popt, Te_perr, Te_chisqr = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=False)

        ## shift profiles if efit_sep is set to False
        if efit_sep:
            xSep_TS = 1
        else:
            xSep_TS = fit_2D.shift_profs([1],rhop_kp,Te[None,:]*1e3,Te_LCFS=Te_sep_eV)
        print('Shifting raw TS points by rhop = {:.3f}'.format(float(1 - xSep_TS)))
        p_ne.X += 1 - xSep_TS
        p_Te.X += 1 - xSep_TS
   
        ne_X_nofilters += 1 - xSep_TS
        Te_X_nofilters += 1 - xSep_TS
 
        # before final fit, initalize electron pressure profile and fit that as well
        # initialize pressure profile tool object and propagate uncertainties from ne, Te
        p_pe = create_pe(p_ne, p_Te)  
        mask_pe = np.full(p_pe.X[:,0].shape, True)
        idxs_pe = np.argsort(p_pe.X[:,0])

        pe_Y_nofilters = p_pe.y
        pe_Y_unc_nofilters = p_pe.err_y

        # now fit!
        ne, ne_popt, ne_perr, ne_chisqr = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te, Te_popt, Te_perr, Te_chisqr = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
        pe, pe_popt, pe_perr, pe_chisqr = fit_profiles.best_osbourne(p_pe.X[idxs_pe,0][mask_pe], p_pe.y[idxs_pe][mask_pe], vals_unc=p_pe.err_y[idxs_pe][mask_pe], x_out=rhop_kp, maxfev=maxfev, reg=reg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
            
        # also fit with a different reg and see which has smaller chisqr
        reg2 = [7,8] # this should help cases that have shallower gradients
        ne2, ne_popt2, ne_perr2, ne_chisqr2 = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=reg2, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te2, Te_popt2, Te_perr2, Te_chisqr2 = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=reg2, edge_chi=edge_chi, ped_width=ped_width, ne=False)
        pe2, pe_popt2, pe_perr2, pe_chisqr2 = fit_profiles.best_osbourne(p_pe.X[idxs_pe,0][mask_pe], p_pe.y[idxs_pe][mask_pe], vals_unc=p_pe.err_y[idxs_pe][mask_pe], x_out=rhop_kp, maxfev=maxfev, reg=reg2, edge_chi=edge_chi, ped_width=ped_width, ne=False)
    
        if (ne_chisqr2 > ne_chisqr):
            nereg = reg
            print('Using ne reg option 1')
        else:
            ne, ne_popt, ne_perr, ne_chisqr = ne2, ne_popt2, ne_perr2, ne_chisqr2
            nereg = reg2
            print('Using ne reg option 2')

        if (Te_chisqr2 > Te_chisqr):
            Tereg = reg
            print('Using Te reg option 1')
        else:
            Te, Te_popt, Te_perr, Te_chisqr = Te2, Te_popt2, Te_perr2, Te_chisqr2
            Tereg = reg2
            print('Using Te reg option 2')

        if (pe_chisqr2 > pe_chisqr):
            pereg = reg
            print('Using pe reg option 1')
        else:
            pe, pe_popt, pe_perr, pe_chisqr = pe2, pe_popt2, pe_perr2, pe_chisqr2
            pereg = reg2
            print('Using pe reg option 2')


    # attempt to fetch ASP and FSP data if available

    use_pressure = False # if set to True will not shift probes and will try to match TS + SP pressure at separatrix
    
    p_ne_p, p_ne_T = None, None

    # this will try to grab probe data if it exists - probably should remove this
    try:
        p_ne_p, p_Te_p  = fetch_edge_probes(shot, (tmin+tmax)/2., Te_sep_eV, geqdsk = geqdsk,
                                        rhop_min=0.9, # 0.96,  # fetch all data before radially shifting
                                        rhop_max=1.1,  # don't trust data too far into the SOL...
                                        probes=probes, shift_probes = not use_pressure) # if use pressure don't shift probes
    except Exception as e:
        print('No good probe data for this shot!')
        print(e)

    # PFS this will be skipped if there is no probe data
    # combine profiles and refit + shift if there is probe data
    if p_ne_p is not None: 

        if use_pressure:
            _out = match_pressure(p_ne, p_Te, p_ne_p, p_Te_p, plot=True)
            p_ne_p, p_Te_p = _out

        filt_TS = False
        filt_SP = False

        # option to filter some data above certain range
        
        if filt_TS:
            _out = filter_TS(p_ne, p_Te, p_ne_p, p_Te_p, 0.02) # cutoff in keV
            p_ne, p_Te, p_ne_p, p_Te_p = _out

        if filt_SP:
            _out = filter_SP(p_ne, p_Te, p_ne_p, p_Te_p, 0.05) # cutoff in keV
            p_ne, p_Te, p_ne_p, p_Te_p = _out
    
        # add cleaned profiles
        p_ne.add_profile(p_ne_p)
        p_Te.add_profile(p_Te_p)
        
        # add the data to the unfiltered array of ETS points
        ne_X_nofilters = np.concatenate((ne_X_nofilters, p_ne_p.X[:,0]))
        ne_Y_nofilters = np.concatenate((ne_Y_nofilters, p_ne_p.y))
        ne_unc_Y_nofilters = np.concatenate((ne_unc_Y_nofilters, p_ne_p.err_y))
        
        Te_X_nofilters = np.concatenate((Te_X_nofilters, p_Te_p.X[:,0]))
        Te_Y_nofilters = np.concatenate((Te_Y_nofilters, p_Te_p.y))
        Te_unc_Y_nofilters = np.concatenate((Te_unc_Y_nofilters, p_Te_p.err_y))

        ne_probe_X = copy.deepcopy(p_ne_p.X)
        Te_probe_X = copy.deepcopy(p_Te_p.X)

        num_ne_SP = len(ne_probe_X)
        num_Te_SP = len(Te_probe_X)
        
        num_ne_SP_nofilter = len(ne_probe_X)
        num_Te_SP_nofilter = len(Te_probe_X)

        ne_X_before_filter = ne_probe_X
        Te_X_before_filter = Te_probe_X

        # perform a re-prefilter - probably only does a filter of SP at this point     
        p_ne, p_Te = prefit_filter(p_ne, p_Te, TS=False)
   
        ne_X_after_filter = p_ne.X[:,0]
        Te_X_after_filter = p_ne.X[:,0]

        ne_pts_removed = np.setdiff1d(ne_X_before_filter, ne_X_after_filter)
        Te_pts_removed = np.setdiff1d(Te_X_before_filter, Te_X_after_filter)

        ne_probe_pts_removed = len(np.intersect1d(ne_probe_X, ne_pts_removed))
        Te_probe_pts_removed = len(np.intersect1d(Te_probe_X, Te_pts_removed))

        num_ne_SP -= ne_probe_pts_removed
        num_Te_SP -= Te_probe_pts_removed
        
 
        #if force_to_zero:
            # force fits to go down in the far SOL (r/a=1.2) to make the routine more robust
        #    p_Te.add_data( np.array([[1.2]]), np.array([10e-3]), err_X=np.array([0.001]), err_y=np.array([0.001]))
        #    p_ne.add_data( np.array([[1.2]]), np.array([0.1]), err_X=np.array([0.001]), err_y=np.array([0.001]))

        #lim_dom = 0 

        # Now fit:    
        idxs_ne = np.argsort(p_ne.X[:,0])
        idxs_Te = np.argsort(p_Te.X[:,0])

        mask_ne = p_ne.X[idxs_ne,0] > min_fit
        mask_Te = p_Te.X[idxs_Te,0] > min_fit
        #mask_ne = np.logical_or(p_ne.X[idxs_ne,0] > min_fit, p_ne.X[idxs_ne,0] < max_fit)
        #mask_Te = np.logical_or(p_Te.X[idxs_Te,0] > min_fit, p_Te.X[idxs_Te,0] < max_fit)
        #mask_ne = np.logical_and(p_ne.X[:,0][idxs_ne] > lim_dom, p_ne.X[:,0][idxs_ne] < 1.05) 
        #mask_Te = np.logical_and(p_Te.X[:,0][idxs_Te] > lim_dom, p_Te.X[:,0][idxs_Te] < 1.05) 

        #mask_ne = np.full(p_ne.X[:,0].shape, True)
        #mask_Te = np.full(p_Te.X[:,0].shape, True)

        # note: reg is only used in osbourne fit, not in superfit


        # fit the profiles! (again)
        maxfev = 2000 # number of iterations for osbourne fit to converge
    
        ne, ne_popt, ne_perr, ne_chisqr = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=nereg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te, Te_popt, Te_perr, Te_chisqr = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=Tereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
    
        # check how many probe points are removed in postfit filter
        ne_probe_X = copy.deepcopy(p_ne_p.X)
        Te_probe_X = copy.deepcopy(p_Te_p.X)
        
        ne_X_before_filter = ne_probe_X
        Te_X_before_filter = Te_probe_X

        p_ne, p_Te = postfit_filter(p_ne, p_Te, ne, Te, rhop_kp)
        
        ne_X_after_filter = p_ne.X[:,0]
        Te_X_after_filter = p_ne.X[:,0]

        ne_pts_removed = np.setdiff1d(ne_X_before_filter, ne_X_after_filter)
        Te_pts_removed = np.setdiff1d(Te_X_before_filter, Te_X_after_filter)

        ne_probe_pts_removed = len(np.intersect1d(ne_probe_X, ne_pts_removed))
        Te_probe_pts_removed = len(np.intersect1d(Te_probe_X, Te_pts_removed))

        num_ne_SP -= ne_probe_pts_removed
        num_Te_SP -= Te_probe_pts_removed
        

        #if force_to_zero:
        #    # add xtra points in the far SOL again to force fits to go down
        #    p_Te.add_data( np.array([[1.2]]), np.array([10e-3]), err_X=np.array([0.001]), err_y=np.array([0.001]))
        #    p_ne.add_data( np.array([[1.2]]), np.array([0.1]), err_X=np.array([0.001]), err_y=np.array([0.001]))
    
        # Fit again:
        idxs_ne = np.argsort(p_ne.X[:,0])
        idxs_Te = np.argsort(p_Te.X[:,0])

        #lim_dom = 0

        mask_ne = p_ne.X[idxs_ne,0] > min_fit
        mask_Te = p_Te.X[idxs_Te,0] > min_fit
        #mask_ne = np.logical_or(p_ne.X[idxs_ne,0] > min_fit, p_ne.X[idxs_ne,0] < max_fit)
        #mask_Te = np.logical_or(p_Te.X[idxs_Te,0] > min_fit, p_Te.X[idxs_Te,0] < max_fit)
        #mask_ne = np.logical_and(p_ne.X[:,0][idxs_ne] > lim_dom, p_ne.X[:,0][idxs_ne] < 1.05) 
        #mask_Te = np.logical_and(p_Te.X[:,0][idxs_Te] > lim_dom, p_Te.X[:,0][idxs_Te] < 1.05) 
    
        #mask_ne = np.full(p_ne.X[:,0].shape, True)
        #mask_Te = np.full(p_Te.X[:,0].shape, True)

        # before final fit, initalize electron pressure profile and fit that as well
        # initialize pressure profile tool object and propagate uncertainties from ne, Te
        p_pe = create_pe(p_ne, p_Te)  
        mask_pe = np.full(p_pe.X[:,0].shape, True)
        idxs_pe = np.argsort(p_pe.X[:,0])
    
        # fit them again! (again)
        ne, ne_popt, ne_perr, ne_chisqr = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=nereg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te, Te_popt, Te_perr, Te_chisqr = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=Tereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
        pe, pe_popt, pe_perr, pe_chisqr = fit_profiles.best_osbourne(p_pe.X[idxs_pe,0][mask_pe], p_pe.y[idxs_pe][mask_pe], vals_unc=p_pe.err_y[idxs_pe][mask_pe], x_out=rhop_kp, maxfev=maxfev, reg=pereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)

        apply_final_sep_shift = False#True
        if apply_final_sep_shift:
            
            # shift combined profiles once more
    
            #xSep_all = fit_2D.shift_profs([1],rhop_kp,Te[None,:]*1e3,Te_LCFS=Te_sep_eV)
            if efit_sep:
                xSep_all = 1
            else:
                xSep_all = fit_2D.shift_profs([1],rhop_kp,Te[None,:]*1e3,Te_LCFS=Te_sep_eV)
            print('Combined profile shift: rhop = {:.3f}'.format(float(1 - xSep_all)))

            p_ne.X += 1 - xSep_all
            p_Te.X += 1 - xSep_all
       
            ne_X_nofilters += 1 - xSep_all
            Te_X_nofilters += 1 - xSep_all
 
            # before final fit, initalize electron pressure profile and fit that as well
            # initialize pressure profile tool object and propagate uncertainties from ne, Te
            p_pe = create_pe(p_ne, p_Te)  
            mask_pe = np.full(p_pe.X[:,0].shape, True)
            idxs_pe = np.argsort(p_pe.X[:,0])

            # now refit to get final profiles
            ne, ne_popt, ne_perr, ne_chisqr = fit_profiles.best_osbourne(p_ne.X[idxs_ne,0][mask_ne], p_ne.y[idxs_ne][mask_ne], vals_unc=p_ne.err_y[idxs_ne][mask_ne], x_out=rhop_kp, maxfev=maxfev, reg=nereg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
            Te, Te_popt, Te_perr, Te_chisqr = fit_profiles.best_osbourne(p_Te.X[idxs_Te,0][mask_Te], p_Te.y[idxs_Te][mask_Te], vals_unc=p_Te.err_y[idxs_Te][mask_Te], x_out=rhop_kp, maxfev=maxfev, reg=Tereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
            pe, pe_popt, pe_perr, pe_chisqr = fit_profiles.best_osbourne(p_pe.X[idxs_pe,0][mask_pe], p_pe.y[idxs_pe][mask_pe], vals_unc=p_pe.err_y[idxs_pe][mask_pe], x_out=rhop_kp, maxfev=maxfev, reg=pereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
            
    
    else:
        num_ne_SP = 0
        num_Te_SP = 0
        num_ne_SP_nofilter = 0
        num_Te_SP_nofilter = 0


    # PFS probably don't need anything else after this point if not interested in gradients

    # now that functions are fit, get gradients

    grad_ne = get_fit_gradient(ne, ne_popt, rhop_kp, 'osborne', geqdsk, grad_type='analytic', out='R', reg=nereg, plot=False)
    grad_Te = get_fit_gradient(Te, Te_popt, rhop_kp, 'osborne', geqdsk, grad_type='analytic', out='R', reg=Tereg, plot=False)
    grad_pe = get_fit_gradient(pe, pe_popt, rhop_kp, 'osborne', geqdsk, grad_type='analytic', out='R', reg=pereg, plot=False)


    '''
    grad_ne_R = get_fit_gradient(ne, ne_popt, roa_kp, 'osborne', geqdsk, grad_type='analytic', out='R', reg=reg, plot=False)
    fig,ax = plt.subplots()
    ax.plot(roa_kp, -grad_ne_R, '-r', lw=2, label='$\\frac{\\partial}{\\partial R}$')
    ax.plot(roa_kp, -grad_ne, '-b', lw=2, label='$\\frac{\\partial}{\\partial r_{vol}}$')
    ax.set_ylabel('$\\nabla n_{e} (10^{20} m^{-4})$')
    ax.set_xlabel('r/a')
    ax.legend(loc='best')
    '''

    # this initializes a profile object which will be useful for calculating errors using
    # the bootsrapping technique
    f_ne = kinetic_profile(rhop_kp, ne, grad_ne)
    f_Te = kinetic_profile(rhop_kp, Te, grad_Te)
    f_pe = kinetic_profile(rhop_kp, pe, grad_pe)

    if frac_err: # this just takes a fraction of the profile as the errorbar
        ne_std = ne*0.2 # 20% -- tanh fitting not set up to provide good uncertainties
        Te_std = Te*0.2 # 20%
        pe_std = np.sqrt(ne_std**2 + Te_std**2) # 20%
        
        grad_ne_std = grad_ne*0.2 # 20% -- tanh fitting not set up to provide good uncertainties
        grad_Te_std = grad_Te*0.2 # 20%
        grad_pe_std = np.sqrt(grad_ne_std**2 + grad_Te_std**2) # 20%

        # if did not do MC iteration, just set the mean coefficients as the fit coefficients
        ne_mean_popt = ne_popt
        Te_mean_popt = Te_popt
        pe_mean_popt = pe_popt
 
        f_ne.fit_std, f_ne.grad_fit_std = ne_std, grad_ne_std
        f_Te.fit_std, f_Te.grad_fit_std = Te_std, grad_Te_std
        f_pe.fit_std, f_pe.grad_fit_std = pe_std, grad_pe_std
       

    else: # this uses a MC perturbation technique to calculate the error
        f_ne.perturb_mc_err(p_ne, num_mc, maxfev, ped_width, nereg, geqdsk, ne=True, verbose=False, plot=False)
        f_Te.perturb_mc_err(p_Te, num_mc, maxfev, ped_width, Tereg, geqdsk, ne=False, verbose=False, plot=False)
        f_pe.perturb_mc_err(p_pe, num_mc, maxfev, ped_width, pereg, geqdsk, ne=False, verbose=False, plot=False)
            

        # now refit the mean and save coefficients of mean
    
        ne_mean_fit, ne_mean_popt, ne_mean_perr, ne_mean_chisqr = fit_profiles.best_osbourne(rhop_kp, f_ne.fit_mean, vals_unc=f_ne.fit_std, x_out=f_ne.x, maxfev=maxfev, reg=nereg, edge_chi=edge_chi, ped_width=ped_width, ne=True)
        Te_mean_fit, Te_mean_popt, Te_mean_perr, Te_mean_chisqr = fit_profiles.best_osbourne(rhop_kp, f_Te.fit_mean, vals_unc=f_Te.fit_std, x_out=f_Te.x, maxfev=maxfev, reg=Tereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
        pe_mean_fit, pe_mean_popt, pe_mean_perr, pe_mean_chisqr = fit_profiles.best_osbourne(rhop_kp, f_pe.fit_mean, vals_unc=f_pe.fit_std, x_out=f_pe.x, maxfev=maxfev, reg=pereg, edge_chi=edge_chi, ped_width=ped_width, ne=False)
    
        plot_fit = False
        if plot_fit:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].errorbar(p_ne.X[:,0], p_ne.y, yerr=p_ne.err_y, fmt='o', label='raw')
            ax[0].plot(f_ne.x, f_ne.y, lw=2.5, label='fit')
            ax[0].plot(f_ne.x, f_ne.fit_mean, lw=2.5, label='mean_fit')

            ax[1].errorbar(p_Te.X[:,0], p_Te.y, yerr=p_Te.err_y, fmt='o')
            ax[1].plot(f_Te.x, f_Te.y, lw=2.5)
            ax[1].plot(f_Te.x, f_Te.fit_mean, lw=2.5)

            #ax[0].legend(['raw','fit','mean_fit'])
            ax[0].legend(loc='best', fontsize=14)

            ax[1].set_xlabel('$\\rho_{p}$')
            ax[0].set_ylabel('$n_{e} (10^{20} m^{-3})$')
            ax[1].set_ylabel('$T_{e} (eV)$')

    # make a pressure profile object
    p_ne_pf = build_profile_prefilter(ne_X_nofilters, ne_Y_nofilters, ne_unc_Y_nofilters, 'ne')
    p_Te_pf = build_profile_prefilter(Te_X_nofilters, Te_Y_nofilters, Te_unc_Y_nofilters, 'Te')
    p_pe_pf = create_pe(p_ne_pf, p_Te_pf)

    # store fit coefficients
  
    f_ne.save_coefs(ne_popt, ne_mean_popt)
    f_Te.save_coefs(Te_popt, Te_mean_popt)
    f_pe.save_coefs(pe_popt, pe_mean_popt)
 
    # one last filter if there are probe points stored
    if p_ne_p is not None:
        ne_probe_X = copy.deepcopy(p_ne_p.X)
        Te_probe_X = copy.deepcopy(p_Te_p.X)
        
        ne_X_before_filter = ne_probe_X
        Te_X_before_filter = Te_probe_X

        p_ne.remove_points(p_ne.X[:,0]>1.1) # remove artificial point
        p_Te.remove_points(p_Te.X[:,0]>1.1) # remove artificial point
    
        ne_X_after_filter = p_ne.X[:,0]
        Te_X_after_filter = p_ne.X[:,0]

        ne_pts_removed = np.setdiff1d(ne_X_before_filter, ne_X_after_filter)
        Te_pts_removed = np.setdiff1d(Te_X_before_filter, Te_X_after_filter)

        ne_probe_pts_removed = len(np.intersect1d(ne_probe_X, ne_pts_removed))
        Te_probe_pts_removed = len(np.intersect1d(Te_probe_X, Te_pts_removed))

        num_ne_SP -= ne_probe_pts_removed
        num_Te_SP -= Te_probe_pts_removed

    else:
        # simple filter 
        p_ne.remove_points(p_ne.X[:,0]>1.1) # remove artificial point
        p_Te.remove_points(p_Te.X[:,0]>1.1) # remove artificial point
    
    #ne_X_after = p_ne.X[:,0] - (1 - xSep_all)
    #Te_X_after = p_Te.X[:,0] - (1 - xSep_all)
    
    #ne_pts_removed = np.setdiff1d(ne_X_before,ne_X_after)
    #Te_pts_removed = np.setdiff1d(Te_X_before,Te_X_after)
  
    #if p_ne_p is not None:
    #    for pt in ne_pts_removed:
    #        if pt in p_ne_p.X:
    #            num_ne_SP -= 1
    #    for pt in Te_pts_removed:
    #        if pt in p_Te_p.X:
    #            num_Te_SP -= 1

    nu_star = calc_nu_star(ne, Te)

    # PFS this is the lfag I use to plot the fits - probably should have this farther up / an argument to the function

    plot_fit = True
    if plot_fit:
        p_ne.plot_data()
        #plt.xlim([0.86,1.16])
        #plt.ylim([-0.01, 0.75])
        plt.gca().plot(rhop_kp, ne)
        #plt.legend(['super_fit', 'super_fit_osborne'])

        p_Te.plot_data()
        #plt.xlim([0.86,1.16])
        #plt.ylim([-0.01, 0.5])
        plt.gca().plot(rhop_kp, Te)
        #plt.legend(['super_fit', 'super_fit_osborne'])
        
        p_pe.plot_data()
        #plt.xlim([0.86,1.16])
        #plt.ylim([-0.01, 0.75])
        plt.gca().plot(rhop_kp, pe)

    num_SP = {}
    num_SP_nf = {}
    num_SP['ne'] = num_ne_SP; num_SP['Te'] = num_Te_SP
    num_SP_nf['ne'] = num_ne_SP_nofilter; num_SP_nf['Te'] = num_Te_SP_nofilter

    # check if there exist any data point in the SOL
    if np.all(p_ne.X[:,0]<0.99):
        #raise ValueError(f'No ne data points  for r/a<0.99 in shot {shot}!')
        print(f'No SOL ne data points for r/a>0.99 in shot {shot}!')
    if np.all(p_Te.X[:,0]<0.99):
     
        #raise ValueError(f'No SOL Te data points for r/a<0.99 in shot {shot}!')
        print(f'No SOL Te data points for r/a>0.99 in shot {shot}!')

    # there's been another filter, so before returning things, recreate the pe profile
    p_pe = create_pe(p_ne, p_Te)  
    
    # output fits + profiletool objects for ne and Te so that experimental data points are passed too
    return f_ne, f_Te, f_pe, p_ne_pf, p_Te_pf, p_pe_pf, num_SP_nf 

            

def prefit_filter(p_ne, p_Te, TS=True):
    '''Function to filter some TS points before fitting.
 
    Parameters
    ----------
    p_ne : profiletools object
        Electron density object before filtering
    p_ne : profiletools object
        Electron temperature object before filtering

    Returns
    -------
    p_ne : profiletools object
        Same electron density object but some points have been filtered
    p_ne : profiletools object
        Same electron temperature object but some points have been filtered

    '''


    # apply these minimum conditions to all TS data (probes sometimes have smaller error bars)
    if TS:
        p_ne.y[p_ne.y<=0.] = 0.01  # 10^18 m^-3
        p_Te.y[p_Te.y<=0.01] = 0.01 # 10 eV
        p_ne.err_y[p_ne.err_y<=0.1] = 0.1 # 10^19 m^-3
        p_Te.err_y[p_Te.err_y<=0.02] = 0.02 # 20 eV

    # remove points with excessively large error bars
    p_ne.remove_points(p_ne.err_y>1) # 10^20 m^-3
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>0.98, p_Te.err_y>0.2)) # max 200 eV of uncertainty in the pedestal
    
    # Remove points with too high values in the SOL:
    p_ne.remove_points(np.logical_and(p_ne.X[:,0]>1.0, p_ne.y>1.5)) # 5e20
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>1.0, p_Te.y>0.25)) # 250 eV

    # Remove ne points with too high uncertainty in the pedestal and SOL:
    p_ne.remove_points(np.logical_and(p_ne.X[:,0]>0.9, p_ne.err_y>0.3)) # 3e19 m^-3
    #p_ne.remove_points(np.logical_and(p_ne.X[:,0]>1.0, p_ne.err_y>0.2)) # 2e19 m^-3 , less in the SOL
    #if shot==1100308004:
    #    p_ne.remove_points(np.logical_and(p_ne.X[:,0]>1.0, p_ne.err_y>0.2)) # 2e19 m^-3 , less in the SOL

    # Remove Te points with too high uncertainty in the SOL (better not to filter in the pedestal, high variability)
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>1.0, p_Te.err_y>0.1))  # 100 eV

    # trivial clean up of Te in the pedestal
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>0.9, p_Te.err_y>0.5))  # 500 eV
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]<0.98, p_Te.y<0.05))  # Te inside r/a=0.98 must always be >50 eV
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]<0.95, p_Te.y<0.1))  # Te inside r/a=0.95 must always be >100 eV
   
    # clean up bad core points
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]<0.2, p_Te.y<1)) # Te inside r/a=0.2 > 1keV
    p_ne.remove_points(np.logical_and(p_ne.X[:,0]<0.2, p_ne.y<0.5)) # ne inside r/a=0.2 > 5e19

    # clean up points with points smaller than errorbars
    p_ne.remove_points(p_ne.y < p_ne.err_y)
    p_Te.remove_points(p_Te.y < p_Te.err_y)

    return p_ne, p_Te



def postfit_filter(p_ne, p_Te, ne, Te, rhop_kp):
    '''Filter some of the TS points based on an existing fit

    Parameters
    ----------
    p_ne : profiletools object
        Same electron density object that was returned by prefit_filter() and after a fit
    p_Te : profiletools object
        Same electron temperature object that was returned by prefit_filter() and after a fit
    ne : 1D array
        Electron density fit used for the filtering
    Te : 1D array
        Electron temperature fit used for the filtering
    rhop_kp : 1D array
        Coordinate used for fits of kinetic profiles that fit was based on

    Returns
    -------
    p_ne : profiletools object
        Same electron density object but has even more points removed based on chi square distance to fit
    p_Te : profiletools object
        Same electron temperature object but has even more points removed based on chi square distance to fit

    '''

    # impose positivity
    ne[ne<np.nanmin(p_ne.y)] = np.nanmin(p_ne.y)
    Te[Te<np.nanmin(p_Te.y)] = np.nanmin(p_Te.y)

    # eliminate points that are more than 3 sigma away and fit again
    p_ne.remove_points(p_ne.X[:,0]>1.1) # remove artificial point
    chi_ne = (p_ne.y - interp1d(rhop_kp, ne)(p_ne.X[:,0]))/p_ne.err_y
    p_ne.remove_points(chi_ne>3)
    p_Te.remove_points(p_Te.X[:,0]>1.1) # remove artificial point
    chi_Te = (p_Te.y - interp1d(rhop_kp, Te)(p_Te.X[:,0]))/p_Te.err_y
    p_Te.remove_points(chi_Te>3)

    return p_ne, p_Te
   
 
def create_pe(p_ne, p_Te):
    '''Creates a pressure profileobject to be fit in addition to ne and Te

    A rather complicated procedure that could probably be simplified, but requires some special attention since
    lengths of ne and Te may be different as a result of the filtering from above
 
    Parameters
    ----------
    p_ne, p_Te : profiletools objects
        Same objects as outlined above, after all filtering associated with one round of fitting is done

    Returns
    -------
    p_ne : profiletoolsobject
        Electron pressure in units of kPa of same length as most heavily filtered (shortest) object (ne or Te)

    '''

    # use whichever has less as template for pressure
    p_template_X = p_Te.X[:,0] if len(p_Te.X) <= len(p_ne.X) else p_ne.X[:,0]
    p_check_X = p_ne.X[:,0] if len(p_Te.X) <= len(p_ne.X) else p_Te.X[:,0]
       
    p_template_y = p_Te.y if len(p_Te.X) <= len(p_ne.X) else p_ne.y
    p_check_y = p_ne.y if len(p_Te.X) <= len(p_ne.X) else p_Te.y
        
    p_template_err_y = p_Te.err_y if len(p_Te.X) <= len(p_ne.X) else p_ne.err_y
    p_check_err_y = p_ne.err_y if len(p_Te.X) <= len(p_ne.X) else p_Te.err_y
        
    template_inds, check_inds = [], []
    template_avg, check_avg = [], []

    overlap = 0
    for xx in p_template_X:
        if xx in p_check_X:
            template_list = list(np.where(p_template_X == xx)[0])
            template_inds += template_list
            check_list = list(np.where(p_check_X == xx)[0])
            check_inds += check_list 
            overlap += 1

        if overlap == 0:
            template_list = []
            check_list = []
            print('No overlap between ne TS points and Te TS points')
        
        num_to_avg = np.abs(len(template_list) - len(check_list))
        if len(template_list) > len(check_list):
            if template_list[:num_to_avg+1] not in template_avg:
                template_avg.append(template_list[:num_to_avg+1])
                check_avg.append(check_list)
        if len(check_list) > len(template_list):
            if check_list[:num_to_avg+1] not in check_avg:
                check_avg.append(check_list[:num_to_avg+1])
                template_avg.append(template_list)

    # remove double counts
    template_inds = list(set(template_inds))
    check_inds = list(set(check_inds))

    unlist_templatelist = [ll for subtemp in check_avg for ll in subtemp]
    unlist_checklist = [ll for subcheck in check_avg for ll in subcheck]
    
    template_X = [p_template_X[tX] for tX in template_inds if tX not in unlist_templatelist]
    for subtemp in template_avg: # for some reason list comprehension doesn't work
        template_X += [np.mean(p_template_X[subtemp])]
    check_X = [p_check_X[tX] for tX in template_inds if tX not in unlist_checklist]
    for subcheck in check_avg:
        check_X += [np.mean(p_check_X[subcheck])]
    
    template_y = [p_template_y[ty] for ty in template_inds if ty not in unlist_templatelist]
    for subtemp in template_avg:
        template_y += [np.mean(p_template_y[subtemp])]
    check_y = [p_check_y[ty] for ty in template_inds if ty not in unlist_checklist]
    for subcheck in check_avg:
        check_y += [np.mean(p_check_y[subcheck])]
    
    template_err_y = [p_template_err_y[tyy] for tyy in template_inds if tyy not in unlist_templatelist]
    for subtemp in template_avg:
        template_err_y += [np.mean(p_template_err_y[subtemp])]
    check_err_y = [p_check_err_y[tyy] for tyy in template_inds if tyy not in unlist_checklist]
    for subcheck in check_avg:
        check_err_y += [np.mean(p_check_err_y[subcheck])]
    
    sort_template = np.argsort(template_X)
    sort_check = np.argsort(check_X)
 
    p_pe_X = np.array(template_X)[sort_template]
    p_pe_y = np.array(template_y)[sort_template]*np.array(check_y)[sort_check]
    p_pe_err_y = np.sqrt(np.array(template_err_y)[sort_template]**2 + np.array(check_err_y)[sort_check]**2)

    from profiletools import BivariatePlasmaProfile as BPP
    p_pe = BPP(X_dim=1,
                X_units=[''],
                y_units='kPa',
                X_labels=['$sqrtpsinorm$'],
                y_label=r'$p_e$')
    
    p_pe_y = p_pe_y*1e20*1.602e-19 # 10^{20}*keV --> kPa
    p_pe_err_y = p_pe_err_y*1e20*1.602e-19 # 10^{20}*keV --> kPa

    p_pe.add_data(p_pe_X, p_pe_y, err_y=p_pe_err_y)

    return p_pe


def build_profile_prefilter(X, y, y_err, kp):
    '''Sets up profiles for ne and Te to be used in create_pe() - could probably just be made into one function
    '''
    
    if kp == 'Te':
        y_units='keV'
        y_label=r'$T_e$'
    if kp == 'ne':
        y_units='10^{20}'
        y_label=r'$n_e$'

    from profiletools import BivariatePlasmaProfile as BPP
        
    p = BPP(X_dim=1,
            X_units=[''],
            y_units=y_units,
            X_labels=['$sqrtpsinorm$'],
            y_label=r'$p_e$')

    p.add_data(X,y,err_y=y_err)

    return p


def fetch_edge_probes(shot, time, Te_sep_eV, geqdsk=None, rhop_min=0.995, rhop_max=1.05, probes=['A'], max_ne_cm3=5e13, shift_probes=True):
    '''Load data for the ASP and FSP probes on Alcator C-Mod. 
    Time in seconds.

    rhop_min and rhop_max are used to subselect the radial range of the data.

    This function returns profiletools data structures.
    See https://profiletools.readthedocs.io/en/latest/#

    Parameters
    ----------
    shot : int
        Queried shot
    time : float
        Queried time slice
    Te_sep_eV :
        Value of electron temperature at separatrix, determined either using 2-point model or some other means
    geqdsk : dict
        Dictionary containing processed EFIT geqdsk file
    rhop_min : float
        Minimum value of rhop for probe radial range selection
    rhop_max : float
        Maximum value of rhop for probe radial range selection
    probes : list, subselection of ['A','F']
        Name of port corresponding to data from desired probe head
    max_ne_cm3 : float
        Maximum density to be permitted. All data points with density greater than this value will be eliminated.
        Default is to have this value extremely high (basically, not active).

    Returns
    -------
    p_ne, p_Te : profiletools object containing combined data from the fetched probes.

    '''
    if geqdsk is None:
        # the geqdsk is used only when using the Aurora radial coordinate transformations
        geqdsk = get_geqdsk_cmod(
            shot, time*1e3, gfiles_loc = '/home/millerma/lya/gfiles/')

    import afsp_probes
    
    has_A = False
    has_F = False
    if 'A' in probes:
      try:
          # if available, add A-Side Probe (ASP) data

          # check raw data first
          out_asp = afsp_probes.get_clean_data(shot, time, geqdsk, probe='A', plot=False)
          rhop_asp, t_range, ne_prof_asp, ne_unc_prof_asp, Te_prof_asp, Te_unc_prof_asp, ax = out_asp
          ne_rhop_asp, Te_rhop_asp = rhop_asp, rhop_asp # later will use the individual ones, so change now

          # these variables defined for conditional statements later
          ne_adb = []
          offset_adb = None
          
          # check edge db as well
          out_adb = afsp_probes.get_edgedb_data(shot, t_range, geqdsk, probe='A', plot=False)
          rhop_adb, ne_adb, Te_adb = out_adb

          # if edge db populated, take that data
          if any(ne_adb):
              
              ne_not_none = np.where(ne_adb != None)
              Te_not_none = np.where(Te_adb != None)

              offset_adb = np.abs(rhop_adb[ne_not_none][0] - rhop_asp[0]) # sometimes offset not populated even though rhop are different

              print(f'Using probe profiles from edge database')
              print(f'Shifting of ASP data from database, in rhop units: {offset_adb:.4f}')

              # edge db doesn't contain uncertainties so interpolate
              interp_shift = rhop_asp[0] - rhop_adb[ne_not_none][0]
              ne_unc_prof_adb = interp1d(rhop_asp, ne_unc_prof_asp, bounds_error=None, fill_value='extrapolate')(rhop_adb+interp_shift) 
              Te_unc_prof_adb = interp1d(rhop_asp, Te_unc_prof_asp, bounds_error=None, fill_value='extrapolate')(rhop_adb+interp_shift) 
             
              # find where None values are in arrays and set old values to new values
              ne_rhop_asp, Te_rhop_asp = rhop_adb[ne_not_none], rhop_adb[Te_not_none] # sometimes they're different...
              ne_prof_asp, Te_prof_asp = ne_adb[ne_not_none], Te_adb[Te_not_none]
              ne_unc_prof_asp, Te_unc_prof_asp = ne_unc_prof_adb[ne_not_none], Te_unc_prof_adb[Te_not_none]

              ne_unc_prof_asp = np.maximum(ne_unc_prof_asp, 0)
              Te_unc_prof_asp = np.maximum(Te_unc_prof_asp, 0)
              
              ne_unc_prof_asp = np.maximum(ne_unc_prof_asp, ne_prof_asp/10)
              Te_unc_prof_asp = np.maximum(Te_unc_prof_asp, Te_prof_asp/10)

          else:
              print('No A-probe data in edge database - using data in tree')

          # probe data is returned in SI units, change to units of 1e20m^-3 and keV
          ne_prof_asp /= 1e20
          ne_unc_prof_asp /= 1e20
          Te_prof_asp /= 1e3
          Te_unc_prof_asp /= 1e3

          ne_prof_asp = ne_prof_asp.astype('float64')
          ne_unc_prof_asp = ne_unc_prof_asp.astype('float64')
          
          Te_prof_asp = Te_prof_asp.astype('float64')
          Te_unc_prof_asp = Te_unc_prof_asp.astype('float64')

          # mask out data points outside of given range
          mask_ne_asp = (ne_rhop_asp>rhop_min)&(ne_rhop_asp<rhop_max)
          mask_Te_asp = (Te_rhop_asp>rhop_min)&(Te_rhop_asp<rhop_max)

          ne_ASP_X = np.ones((len(ne_rhop_asp),1))
          Te_ASP_X = np.ones((len(Te_rhop_asp),1))
          ne_ASP_X[:,0] = ne_rhop_asp
          Te_ASP_X[:,0] = Te_rhop_asp

          p_ne_ASP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='$10^{20}$ m$^{-3}$',
                                                         X_labels=r'$\\rho_{p}$', y_label=r'$n_e$, ASP')
          p_ne_ASP.abscissa = 'sqrtpsinorm'
          p_ne_ASP.shot = shot
          p_ne_ASP.t_min = t_range[0]; p_ne_ASP.t_max = t_range[1]
          channels = range(0, len(ne_prof_asp[mask_ne_asp]))
          p_ne_ASP.add_data(ne_ASP_X[mask_ne_asp], ne_prof_asp[mask_ne_asp], channels={0: channels}, err_y=ne_unc_prof_asp[mask_ne_asp])
          p_ne_ASP.remake_efit_tree() # needed to add this for some reason on 11/1/22

          p_Te_ASP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='keV',
                                                         X_labels=r'$\\rho_{p}$', y_label=r'$T_e$, ASP')
          p_Te_ASP.abscissa = 'sqrtpsinorm'
          p_Te_ASP.shot = shot
          p_Te_ASP.t_min = t_range[0]; p_Te_ASP.t_max = t_range[1]
          channels = range(0, len(Te_prof_asp[mask_Te_asp]))
          p_Te_ASP.add_data(Te_ASP_X[mask_Te_asp], Te_prof_asp[mask_Te_asp], channels={0: channels}, err_y=Te_unc_prof_asp[mask_Te_asp])
          p_Te_ASP.remake_efit_tree() # needed to add this for some reason on 11/1/22
          
          has_A = True
      except Exception as e:
          print('ASP fetch failed because:')
          print(e)
          pass

    ######
    if 'F' in probes:
      try:
          # if available, add F-Side Probe (FSP) data

          # check raw data first
          out = afsp_probes.get_clean_data(shot, time, geqdsk, probe='F', plot=False)
          rhop_fsp, t_range, ne_prof_fsp, ne_unc_prof_fsp, Te_prof_fsp, Te_unc_prof_fsp, ax = out
          ne_rhop_fsp, Te_rhop_fsp = rhop_fsp, rhop_fsp # later will use the individual ones, so change now

          # define variables for conditionals
          ne_fdb = []
          offset_fdb = None

          # check edge db as well
          out_fdb = afsp_probes.get_edgedb_data(shot, t_range, geqdsk, probe='F', plot=False)
          rhop_fdb, ne_fdb, Te_fdb = out_fdb
          

          # if edge db populated, take that data
          if any(ne_fdb):

              ne_not_none = np.where(ne_fdb != None)
              Te_not_none = np.where(Te_fdb != None)
              
              offset_fdb = np.abs(rhop_fdb[ne_not_none][0] - rhop_fsp[0]) # sometimes offset not populated even though rhop are different
              
              print(f'Using probe profiles from edge database')
              print(f'Shifting of FSP data from database, in rhop units: {offset_fdb:.4f}')

              # edge db doesn't contain uncertainties so interpolate
              interp_shift = rhop_fsp[0] - rhop_fdb[ne_not_none][0]
              ne_unc_prof_fdb = interp1d(rhop_fsp, ne_unc_prof_fsp, bounds_error=None, fill_value='extrapolate')(rhop_fdb+interp_shift) 
              Te_unc_prof_fdb = interp1d(rhop_fsp, Te_unc_prof_fsp, bounds_error=None, fill_value='extrapolate')(rhop_fdb+interp_shift) 
             
              # find where None values are in arrays and set old values to new values
              ne_rhop_fsp, Te_rhop_fsp = rhop_fdb[ne_not_none], rhop_fdb[Te_not_none] # sometimes they're different...
              ne_prof_fsp, Te_prof_fsp = ne_fdb[ne_not_none], Te_fdb[Te_not_none]
              ne_unc_prof_fsp, Te_unc_prof_fsp = ne_unc_prof_fdb[ne_not_none], Te_unc_prof_fdb[Te_not_none]

              ne_unc_prof_fsp = np.maximum(ne_unc_prof_fsp, 0)
              Te_unc_prof_fsp = np.maximum(Te_unc_prof_fsp, 0)
              
              ne_unc_prof_fsp = np.maximum(ne_unc_prof_fsp, ne_prof_fsp/10)
              Te_unc_prof_fsp = np.maximum(Te_unc_prof_fsp, Te_prof_fsp/10)


          else:
              print('No F-probe data in edge database - using data in tree')


          # probe data is returned in SI units, change to units of 1e20m^-3 and keV
          ne_prof_fsp /= 1e20
          ne_unc_prof_fsp /= 1e20
          Te_prof_fsp /= 1e3
          Te_unc_prof_fsp /= 1e3

          ne_prof_fsp = ne_prof_fsp.astype('float64')
          ne_unc_prof_fsp = ne_unc_prof_fsp.astype('float64')
          
          Te_prof_fsp = Te_prof_fsp.astype('float64')
          Te_unc_prof_fsp = Te_unc_prof_fsp.astype('float64')

          # mask out data points outside of given range
          mask_ne_fsp = (ne_rhop_fsp>rhop_min)&(ne_rhop_fsp<rhop_max)
          mask_Te_fsp = (Te_rhop_fsp>rhop_min)&(Te_rhop_fsp<rhop_max)

          ne_FSP_X = np.ones((len(ne_rhop_fsp),1))
          ne_FSP_X[:,0] = ne_rhop_fsp
          Te_FSP_X = np.ones((len(Te_rhop_fsp),1))
          Te_FSP_X[:,0] = Te_rhop_fsp

          p_ne_FSP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='$10^{20}$ m$^{-3}$',
                                                         X_labels=r'$\\rho_{p}$', y_label=r'$n_e$, FSP')
          p_ne_FSP.abscissa = 'sqrtpsinorm'
          p_ne_FSP.shot = shot
          p_ne_FSP.t_min = t_range[0]; p_ne_FSP.t_max = t_range[1]        
          channels = range(0, len(ne_prof_fsp[mask_ne_fsp]))
          p_ne_FSP.add_data(ne_FSP_X[mask_ne_fsp], ne_prof_fsp[mask_ne_fsp], channels={0: channels}, err_y=ne_unc_prof_fsp[mask_ne_fsp])
          p_ne_FSP.remake_efit_tree() # needed to add this for some reason on 11/1/22

          p_Te_FSP = profiletools.BivariatePlasmaProfile(X_dim=1, X_units='', y_units='keV',
                                                         X_labels=r'$\\rho_{p}$', y_label=r'$T_e$, FSP')
          p_Te_FSP.abscissa = 'sqrtpsinorm'
          p_Te_FSP.shot = shot
          p_Te_FSP.t_min = t_range[0]; p_Te_FSP.t_max = t_range[1]
          channels = range(0, len(Te_prof_fsp[mask_Te_fsp]))
          p_Te_FSP.add_data(Te_FSP_X[mask_Te_fsp], Te_prof_fsp[mask_Te_fsp], channels={0: channels}, err_y=Te_unc_prof_fsp[mask_Te_fsp])
          p_Te_FSP.remake_efit_tree() # needed to add this for some reason on 11/1/22
          
          has_F = True

      except Exception as e:
          print('FSP fetch failed because:')
          print(e)
          pass
        
    
    def probe_func(x,a,k,b):
        ''' Exponential function used to fit probe data'''
        return a*np.exp(-k*(x-1))+b

    # extended range for fitting -- in practice though, shifting should be <0.005 units typically
    x_out = np.linspace(0.96, 1.05, 300)

    # if offset_[a/f]db is True, probes have already been shifted - shift_probes flag might be turned off if we try to match pressure instead
    if has_A and shift_probes and offset_adb is None:
        
        # fit exponential over extended range near the LCFS for Te
        popt_Te, pcov_Te = curve_fit(probe_func,p_Te_ASP.X[:,0], p_Te_ASP.y,
                                     p0=[0.06,1e2,0.03]) # important to provide a decent guess
        Te_keV_ASP_fit = probe_func(x_out,*popt_Te)

        try:
            # now, based on this exponential, find location of the LCFS        
            if efit_sep:
                xSep_ASP = 1
            else:
                xSep_ASP = interp1d(Te_keV_ASP_fit, x_out, bounds_error=True)(Te_sep_eV*1e-3)
            print(f'Shifting of ASP data from fit, in rhop units: {1-xSep_ASP:.4f}')

            # shift both ne and Te data
            p_Te_ASP.X += 1 - xSep_ASP
            p_ne_ASP.X += 1 - xSep_ASP
            
        except ValueError:
            print('Shifting of ASP data failed, likely probe did not go very far into the plasma')

            has_A=False


    if has_F and shift_probes and offset_fdb is None:
        # fit exponential over extended range near the LCFS for Te
        popt_Te, pcov_Te = curve_fit(probe_func,p_Te_FSP.X[:,0], p_Te_FSP.y,
                                     p0=[0.06,1e2,0.03]) # important to provide a decent guess
        Te_keV_FSP_fit = probe_func(x_out,*popt_Te)

        try:
            # now, based on this exponential, find location of the LCFS        
            if efit_sep:
                xSep_FSP = 1
            else:
                #xSep_FSP = fit_2D.shift_profs([1],X_TS_fit,Te_TS_fit[None,:]*1e3,Te_LCFS=Te_sep_eV)
                xSep_FSP = interp1d(Te_keV_FSP_fit, x_out, bounds_error=True)(Te_sep_eV*1e-3)
            print(f'Shifting of FSP data from fit, in rhop units: {1-xSep_FSP:.4f}')
            
            # shift both ne and Te data
            p_Te_FSP.X += 1 - xSep_FSP
            p_ne_FSP.X += 1 - xSep_FSP

        except ValueError:
            print('Shifting of FSP data failed, likely probe did not go very far into the plasma')

            has_F=False

    import copy
    
    # combine data from ASP and FSP probes if both are available
    if has_A and has_F:
        p_ne = copy.deepcopy(p_ne_ASP)
        p_ne.add_profile(p_ne_FSP)
        p_Te = copy.deepcopy(p_Te_ASP)
        p_Te.add_profile(p_Te_FSP)
    elif has_A and not has_F:
        p_ne = copy.deepcopy(p_ne_ASP)
        p_Te = copy.deepcopy(p_Te_ASP)
    elif has_F and not has_A:
        p_ne = copy.deepcopy(p_ne_FSP)
        p_Te = copy.deepcopy(p_Te_FSP)
    else:
        p_ne = None
        p_Te = None

    # multiply probe ne by some factor to match TS
    #mult_factor = 2

    #p_ne.y*=mult_factor
    #p_ne.y*=mult_factor

    # remove probe points for r/a < 0.99 (typically unreliable)
    if p_ne is not None:
        p_ne.remove_points(p_ne.X[:,0] < 0.99)
        p_Te.remove_points(p_Te.X[:,0] < 0.99)

        # remove data corresponding to likely emissive probe behavior
        # have to filter Te this way because sometimes the length is different
        for ne_idx in range(len(p_ne.y)):
            if p_ne.y[ne_idx]*1e14 > max_ne_cm3:
                p_Te.remove_points(p_Te.X[:,0] == p_ne.X[ne_idx,0])
        p_ne.remove_points(p_ne.y*1e14 > max_ne_cm3)

        if p_ne.X is None:
            raise ValueError('All points from probe ne were removed! max_ne_cm3 is likely too small.')
  
    return p_ne, p_Te


def filter_TS(ne_TS, Te_TS, ne_SP, Te_SP, cutoff):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''
    
    if ne_SP is not None:

        # make sure X array is increasing
        order_SP = np.argsort(Te_SP.X[:,0])

        filt = np.where(Te_SP.y[order_SP] < cutoff)[0]
        x_filt = Te_SP.X[order_SP][filt[0],0]
        ne_TS.remove_points(ne_TS.X[:,0] > x_filt)
        Te_TS.remove_points(Te_TS.X[:,0] > x_filt)

    return ne_TS, Te_TS, ne_SP, Te_SP


def filter_SP(ne_TS, Te_TS, ne_SP, Te_SP, cutoff):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''

    # make sure X array is increasing
    order = np.argsort(Te_TS.X[:,0])

    filt = np.where(Te_TS.y[order] > cutoff)[0]
    x_filt = Te_TS.X[order][filt[-1],0]
    ne_SP.remove_points(ne_SP.X[:,0] < x_filt)
    Te_SP.remove_points(Te_SP.X[:,0] < x_filt)

    return ne_TS, Te_TS, ne_SP, Te_SP


def match_pressure(ne_TS, Te_TS, ne_SP, Te_SP, plot=False):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''

    ne_SP_sep = interp1d(ne_SP.X[:,0], ne_SP.y)(1)
    Te_SP_sep = interp1d(Te_SP.X[:,0], Te_SP.y)(1)
    # calculate electron pressure at rhop = 1 as measured by SP
    pe_SP_sep = ne_SP_sep*Te_SP_sep

    # find where the SP pressure matches the TS pressure
    # need to interpolate Te onto ne
    Te_TS_intp = interp1d(Te_TS.X[:,0], Te_TS.y, fill_value='extrapolate')(ne_TS.X[:,0])
    pe_TS = ne_TS.y*Te_TS_intp
    rhop_TS = ne_TS.X[:,0]

    rhop_TS_match = interp1d(pe_TS, rhop_TS, fill_value='extrapolate')(pe_SP_sep)

    # shift SP profiles
    SP_shift = rhop_TS_match - 1
    ne_SP.X[:,0] += SP_shift
    Te_SP.X[:,0] += SP_shift
            
    print(f'Shifting of ASP data, in rhop units: {SP_shift:.4f}')

    if plot:

        # calculate overall pressure SP profile to plot in same way as TS

        Te_SP_intp = interp1d(Te_SP.X[:,0], Te_SP.y, fill_value='extrapolate')(ne_SP.X[:,0])
        pe_SP = ne_SP.y*Te_SP_intp
        rhop_SP = ne_SP.X[:,0] - SP_shift

        fig, ax = plt.subplots()
        ax.plot(rhop_TS, pe_TS, 'o')
        ax.plot(rhop_SP, pe_SP, 'o')
        ax.plot(ne_SP.X[:,0], pe_SP, 'o')
        plt.xlim([0.86,1.16])
        plt.ylim([-0.01, 0.27])
        plt.legend(['TS', 'SP before shift', 'SP after shift'])

    return ne_SP, Te_SP


# def fav_vs_unfav(shot,time,geqdsk = None):
#     '''Determine whether grad-B field direction is favorable or unfavorable.
    
#     This function ignores the possibility of having a double null, use with care!
#     The active x-point is taken to be the point along the LCFS that is furthest from Z=0.

#     '''
#     if geqdsk is None:
#         geqdsk = get_geqdsk_cmod(shot,time*1e3)
#         geqdsk = get_geqdsk_cmod(shot,time*1e3)  # repeat to make sure it's loaded...
    
#     # find ion grad(B)-drift direction (determined by B field dir, since radial grad(B) is always inwards )
#     magTree = MDSplus.Tree('magnetics',shot)
#     nodeBt = magTree.getNode('\magnetics::Bt')
#     Bt = nodeBt.data()
#     time_Bt = nodeBt.dim_of().data()
#     tidx = np.argmin(np.abs(time_Bt - time)) 
#     gradB_drift_up = False if Bt[tidx]<0 else True
    
#     # find whether shot is USN or LSN -- assume not DN...
#     maxZ = np.max(geqdsk['ZBBBS'])
#     minZ = np.min(geqdsk['ZBBBS'])
    
#     #  pretty sure that the X-point is where the LCFS is furthest from the magnetic axis
#     USN = True if np.abs(maxZ)==np.max([np.abs(maxZ),np.abs(minZ)]) else False
    
#     # favorable or unfavorable grad-B drift direction?
#     favorable = (gradB_drift_up and USN) or (gradB_drift_up==False and USN==False)
    
#     return gradB_drift_up, USN, favorable

        
def get_vol_avg(shot,time,rhop,ne,Te,geqdsk=None, quantities=['p','n','T']):
    ''' Calculate volume-averaged pressure given some ne,Te radial profiles.

    ne must be in cm^-3 units and Te in eV.
    '''
    # find volume-averaged pressure    
    p_Pa = (ne*1e6) * (Te*q_electron)
    p_atm = p_Pa/101325.  # conversion factor between Pa and atm

    # load geqdsk dictionary
    if geqdsk is None:
        geqdsk = get_geqdsk_cmod(shot,time*1e3)
    
    # find volume average within LCFS
    indLCFS = np.argmin(np.abs(rhop-1.0))
    n_m3_vol_avg = aurora.vol_average(ne[:indLCFS]/1e14, rhop[:indLCFS], method='omfit',geqdsk = geqdsk)[-1]
    T_eV_vol_avg = aurora.vol_average(Te[:indLCFS], rhop[:indLCFS], method='omfit',geqdsk = geqdsk)[-1]
    p_Pa_vol_avg = aurora.vol_average(p_Pa[:indLCFS], rhop[:indLCFS], method='omfit',geqdsk = geqdsk)[-1]
    #p_atm_vol_avg = p_Pa_vol_avg/101325.

    return_list = []
    if 'p' in quantities:
        return_list.append(p_Pa_vol_avg)
    if 'n' in quantities:
        return_list.append(n_m3_vol_avg)
    if 'T' in quantities:
        return_list.append(T_eV_vol_avg)
    return return_list



def get_geqdsk_cmod(shot, time_ms, gfiles_loc = '/home/sciortino/EFIT/gfiles/', return_fname=False):
    ''' Get a geqdsk file in omfit_eqdsk format by loading it from disk, if available, 
    or from MDS+ otherwise.  

    This function tries to first load a EFIT20, if available.

    time must be in ms!

    Currently, the omfit_eqdsk class struggles to connect to MDS+ sometimes. To work 
    around this problem, this function uses a loop and try-except statements to try to 
    load multiple times until succeeding.
    '''
    time_ms = np.floor(time_ms)   # TODO: better to do this outside!!
    file_name=f'g{shot}.{str(int(time_ms)).zfill(5)}'

    def fetch_and_move():
        try:
            # try to fetch EFIT20 first
            geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                device='CMOD',shot=shot, time=time_ms, SNAPfile='EFIT20',
                fail_if_out_of_range=True,time_diff_warning_threshold=20
            )
        except:
            # if EFIT20 is not available, look for default ANALYSIS EFIT
            geqdsk = omfit_eqdsk.OMFITgeqdsk('').from_mdsplus(
                device='CMOD',shot=shot, time=time_ms, SNAPfile='ANALYSIS',
                fail_if_out_of_range=True,time_diff_warning_threshold=20
            )
    
        geqdsk.save(raw=True)
        shutil.move(file_name, gfiles_loc+file_name)

    def attempt_loading():
        if os.path.exists(gfiles_loc + file_name):
            # fetch local g-file if available
            try:
                geqdsk = omfit_eqdsk.OMFITgeqdsk(gfiles_loc + file_name)
                kk = geqdsk.keys()  # quick test
            except:
                geqdsk = fetch_and_move()
        else:
            geqdsk = fetch_and_move()

        return geqdsk
    
    # sometimes there are issues loading and this must be enforced as follows
    # this may become redundant in the future when this bug is solved...
    for ijk in np.arange(10): # 10 attempts
        try:
            geqdsk = attempt_loading()
            geqdsk.load()  # for safety
            geqdsk['fluxSurfaces'].load()  # for safety
            break
        except Exception as e:
            pass

    if return_fname:
        return geqdsk, gfiles_loc+file_name
    else:
        return geqdsk

    
def get_Greenwald_frac(shot, tmin,tmax, rhop, ne, Ip_MA, a_m=0.22, geqdsk=None):
    ''' Calculate Greenwald density fraction by normalizing volume-averaged density.

    INPUTS
    ------
    shot : int, shot number
    tmin and tmax: floats, time window (in [s]) to fetch equilibrium.
    ne: 1D array-like, expected as time-independent. Units of 1e20 m^-3.
    Ip_MA: float, plasma current in MA. 
    a_m : minor radius in units of [m]. Default of 0.69 is for C-Mod. 

    OUTPUTS
    -------
    n_by_nG : float
        Greenwald density fraction, defined with volume-averaged density.
    '''
    if geqdsk is None:
        tmin *= 1000.  # change to ms
        tmax *= 1000.  # change to ms
        time = (tmax+tmin)/2.
        geqdsk = get_geqdsk_cmod(shot,time)
    
    # find volume average within LCFS
    indLCFS = np.argmin(np.abs(rhop-1.0))
    n_volavg = aurora.vol_average(ne[:indLCFS], rhop[:indLCFS], geqdsk=geqdsk)[-1]

    # Greenwald density
    n_Gw = Ip_MA/(np.pi * a_m**2)   # units of 1e20 m^-3, same as densities above

    # Greenwald fraction:
    f_gw = n_volavg/n_Gw

    return f_gw



def get_CMOD_gas_fueling(shot, plot=False):
    '''Load injected gas amounts and give a grand total in Torr-l.
    Translated from gas_input2_ninja.dat scope. 
    '''
    
    cmod = MDSplus.Tree('cmod', shot)
    _c_side = smooth(cmod.getNode('\\plen_cside').data()[0,:],31)
    # _c_side = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                             # TDI='\\plen_cside').data()[0,:],31)
    _t = cmod.getNode('\\plen_cside').dim_of().data()
    # _t = omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                 # TDI='dim_of(\\plen_cside)').data()
    _b_sideu = smooth(cmod.getNode('\\plen_bsideu').data()[0,:],31)
    # _b_sideu = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                              # TDI='\\plen_bsideu').data()[0,:],31)
    _b_top = smooth(cmod.getNode('\\plen_btop').data()[0,:],31)
    # _b_top = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='cmod',
                                            # TDI='\\plen_btop').data()[0,:],31)

    ninja = True
    try:
        edge = MDSplus.Tree('edge', shot)
        plen_bot_time = edge.getNode('\\edge::gas_ninja.plen_bot').dim_of(0).data()
        # plen_bot_time = omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='edge',
                                            # TDI='\edge::gas_ninja.plen_bot').dim_of(0)
        plen_bot = smooth(edge.getNode('\\edge::gas_ninja.plen_bot').data()[0,:],31)
        # plen_bot = smooth(omfit_mds.OMFITmdsValue(server='CMOD',shot=shot,treename='edge',
                                              # TDI='\edge::gas_ninja.plen_bot').data()[0,:],31)
    except:
        ninja = False
        print('No gas fueling from ninja system')

    # only work with quantities within [0,2]s interval
    ind0 = np.argmin(np.abs(_t)); ind1 = np.argmin(np.abs(_t-2.0))

    time = _t[ind0:ind1]
    c_side = _c_side[ind0:ind1]
    b_sideu = _b_sideu[ind0:ind1]
    b_top = _b_top[ind0:ind1]

    # ninja system is on a different time base than the other measurements
    ninja2 = interp1d(plen_bot_time, plen_bot, bounds_error=False)(time) if ninja else 0
    
    gas_tot = c_side + b_sideu + b_top + ninja2

    if plot:
        fig,ax = plt.subplots()
        ax.plot(time, gas_tot, label='total')
        ax.plot(time, c_side, label='c-side')
        ax.plot(time, b_sideu, label='b-side u')
        ax.plot(time, b_top, label='b-top')
        ax.plot(time, ninja2, label='ninja2')
        ax.legend(loc='best').set_draggable(True)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Total injected gas [Torr-l]')
        
    return time, gas_tot


def calc_nu_star(ne, Te):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''

    ne = ne*1e20
    Te = Te*1e3

    m_i = 1.67e-27
    m_e = 9.11e-31
    mu = m_i/m_e
    q_e = 1.602e-19
    e0 = 8.85e-12
    a = 0.22
    R0 = 0.68
    safety_factor = 2 ### how to get safety factor??
    
    v_th = np.sqrt(Te*q_e/m_e)
    nu = 1/(4*np.pi)*q_e**4*ne*15/(e0**2*m_e**(1/2)*(Te*q_e)**(3/2))
    epsilon = a/R0
    nu_star = nu/epsilon**(3/2)/(v_th/safety_factor*R0)

    return nu_star


def identify_mode(shot):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''
    
    label = 'mam'
    lrf07_array = np.genfromtxt('db_shotlists/fy07/lyman_lmode_fy07_{}.txt'.format(label),skip_header=2)
    loh07_array = np.genfromtxt('db_shotlists/fy07/lyman_ohmic_fy07_{}.txt'.format(label),skip_header=2)
    h07_array = np.genfromtxt('db_shotlists/fy07/lyman_hmode_fy07_{}.txt'.format(label),skip_header=2)
  
    label = 'fs'  
    lrf08_array = np.genfromtxt('db_shotlists/fy08/lyman_lmodes_fy08_{}.txt'.format(label),skip_header=2)
    loh08_array = np.genfromtxt('db_shotlists/fy08/lyman_ohmic_fy08_{}.txt'.format(label),skip_header=2)
    h08_array = np.genfromtxt('db_shotlists/fy08/lyman_hmodes_fy08_{}.txt'.format(label),skip_header=2)
#    i08_array = np.genfromtxt('lyman_imode_fy08_test.txt',skip_header=2)

    l_modes = [int(s) for s in  lrf07_array[:,1]]
    l_modes += [int(s) for s in  loh07_array[:,1]]
    l_modes += [int(s) for s in  lrf08_array[:,1]]
    l_modes += [int(s) for s in  loh08_array[:,1]]
    h_modes = [int(s) for s in  h07_array[:,1]]
    h_modes += [int(s) for s in  h08_array[:,1]]
#    i_modes = [int(s) for s in  i08_array[:,1]]

    if shot in l_modes:
        mode = 'L'
    elif shot in h_modes:
        mode = 'H'
#    elif shot in i_modes:
#        mode = 'I'
    else:
        print('Shot not in database')
        mode = input('Specify mode if known (H/L/I), else L-mode will be assumed:')
        if mode == 'N': mode = 'L' 
        return mode

    return mode


def get_fit_gradient(y, c, rhop, fit_type, geqdsk, grad_type='analytic', out='rvol', reg=None, plot=True):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''

    x = rhop
    
    # compute derivatives of different coordinates for ease - let's use R as the base
    
    #Rsep = aurora.rad_coord_transform(1.0, 'r/a', 'Rmid', geqdsk)
    #rminor = Rsep - geqdsk['RMAXIS'] 
    #R = roa * rminor + geqdsk['RMAXIS']
    #rhop = aurora.get_rhop_RZ(R, np.zeros_like(R), geqdsk)
    
    import single_shot as ss
    _rvol, _rhop = ss.get_rvol(geqdsk, dr0=0.03, dr1=0.03)
    rvol = interp1d(_rhop, _rvol, fill_value='extrapolate')(rhop)
    psin = rhop**2
    
    from scipy.interpolate import UnivariateSpline
    mp_ind = np.where(geqdsk['AuxQuantities']['Z'] == 0)[0][0]
    R_mp = geqdsk['AuxQuantities']['R']
    rhop_mp = geqdsk['AuxQuantities']['RHOpRZ'][mp_ind]
    R0 = geqdsk['RMAXIS']
    omp = R_mp > R0
    R = UnivariateSpline(rhop_mp[omp], R_mp[omp])(rhop)

    if grad_type == 'analytic':

        if fit_type == 'osborne': grad = fit_profiles.Osbourne_Tanh_gradient(x, c, reg=reg)
        if fit_type == 'superfit': grad = fit_profiles.mtanh_profile_gradient(x, edge=c[0], ped=c[1], core=c[2], expin=c[3], expout=c[4], widthp=c[5], xphalf=c[6])

    else:
        grad = np.gradient(y)

    # jacobians
    if out == 'rhop': jacobian = np.ones(len(rhop))
    if out == 'R': jacobian = np.gradient(rhop, R)
    if out == 'psin': jacobian = np.gradient(rhop, psin)
    if out == 'rvol': jacobian = np.gradient(rhop, rvol)

    if plot:

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(x, y, 'k-')    
        ax[1].plot(x, grad, 'k-')

    return jacobian*grad


class kinetic_profile:

    # small class to store results from fits so that we can do monte carlo error estimation more easily
    
    def __init__(self, x, y, grad_y):

        self.x = x
        self.y = y
        self.grad_y = grad_y

    def perturb_mc_err(self, raw, num_mc, maxfev, ped_width, reg, geqdsk, ne=True, verbose=True, plot=False):
        
        err = raw.err_y

        if plot:
    
            fig, ax = plt.subplots()
            ax.errorbar(raw.X[:-1,0], raw.y[:-1], raw.err_y[:-1], fmt='.')
            ax.set_ylabel('$n_{e}$ (10$^{20}$m$^{-3})$')
            ax.set_xlabel('r/a')


        max_iter = np.max(num_mc) # get biggest one
    
        if type(num_mc) == int: num_mc = np.array([num_mc]) # if pass in int, convert to array
            
        # these arrays hold standard deviations and mean for different number of iterations
        npts_fit = len(self.y) # number of points in fit

        # these arrays hold the overall distributions of ne, Te, and their chisqr
        fit_dist = np.zeros([max_iter, npts_fit])
        grad_fit_dist = np.zeros([max_iter, npts_fit])
        xsqr_dist = np.zeros([max_iter])

        # iterate through MC perturbation of largest number

        for ii in range(max_iter):
            new_pts = raw.y + np.random.normal(scale=raw.err_y)
    
            try:
                _out = fit_profiles.best_osbourne(raw.X[:,0], new_pts, vals_unc=raw.err_y, x_out=self.x, maxfev=maxfev, reg=reg, ped_width=ped_width, ne=ne)

                fit_dist[ii], popt, perr, xsqr_dist[ii] = _out
                grad_fit_dist[ii] = get_fit_gradient(fit_dist[ii], popt, self.x, 'osborne', geqdsk, grad_type='analytic', out='R', reg=reg, plot=False)

                if verbose: 
                    print('MC iteration {}'.format(ii), 'XSQR = {:.3f}'.format(xsqr_dist[ii]))

                    if plot: ax.plot(self.x, fit_dist[ii], 'k-')
           
                if not np.isfinite(fit_dist[ii]).all():
                    inf_vals_fd = ~np.isfinite(fit_dist[ii])
                    inf_vals_gfd = ~np.isfinite(grad_fit_dist[ii])
                    # just fill them with nans for consistency / handling later
                    fit_dist[inf_vals_fd] = np.nan
                    grad_fit_dist[inf_vals_gfd] = np.nan
    
 
            except:
                placeholder = np.empty(self.x.shape)
                placeholder[:] = np.nan
                
                fit_dist[ii] = placeholder
                grad_fit_dist[ii] = placeholder
                xsqr_dist[ii] = np.nan

                print('MC fit number {} didnt work - ignoring'.format(ii))


        # now sort chisqr distributions and get rid of top 10%
        num_mc_10p = np.int(np.ceil(num_mc/10))

        xsqr_inds = np.argsort(xsqr_dist)[-num_mc_10p:]
        keep_mask = np.ones(len(xsqr_dist), dtype=bool)
        keep_mask[xsqr_inds] = False

        
        fit_mean = np.zeros([len(num_mc),npts_fit])
        fit_std = np.zeros([len(num_mc),npts_fit])
        grad_fit_mean = np.zeros([len(num_mc),npts_fit])
        grad_fit_std = np.zeros([len(num_mc),npts_fit])

        # now iterate through number of iterations and take standard deviations of all of them
        for ss in range(len(num_mc)):
        
            fit_std[ss] = np.std(fit_dist[:num_mc[ss]][keep_mask], axis=0)
            fit_mean[ss] = np.mean(fit_dist[:num_mc[ss]][keep_mask], axis=0)
            grad_fit_std[ss] = np.std(grad_fit_dist[:num_mc[ss]][keep_mask], axis=0)
            grad_fit_mean[ss] = np.mean(grad_fit_dist[:num_mc[ss]][keep_mask], axis=0)

            # z_score = 3
            # if remove_outliers:
        
            #     # throw out points in fits that are more than 3std away
            #     ne_norm = np.abs(ne_fit_dist[:num_mc[ss]] - ne_fit_mean[ss])
            #     grad_ne_norm = np.abs(grad_ne_fit_dist[:num_mc[ss]] - grad_ne_fit_mean[ss])
            #     Te_norm = np.abs(Te_fit_dist[:num_mc[ss]] - Te_fit_mean[ss])
            #     grad_Te_norm = np.abs(grad_Te_fit_dist[:num_mc[ss]] - grad_Te_fit_mean[ss])

            #     mask_3std_ne = np.logical_and(ne_norm <= z_score*ne_fit_std[ss], grad_ne_norm <= z_score*grad_ne_fit_std[ss])
            #     ne_fit_std[ss] = np.array([np.std(ne_fit_dist[:,rr][mask_3std_ne[:,rr]]) for rr in range(len(roa_kp))])
            #     grad_ne_fit_std[ss] = np.array([np.std(grad_ne_fit_dist[:,rr][mask_3std_ne[:,rr]]) for rr in range(len(roa_kp))])
        
            #     mask_3std_Te = np.logical_and(Te_norm <= z_score*Te_fit_std[ss], grad_Te_norm <= z_score*grad_Te_fit_std[ss])
            #     Te_fit_std[ss] = np.array([np.std(Te_fit_dist[:,rr][mask_3std_Te[:,rr]]) for rr in range(len(roa_kp))])
            #     grad_Te_fit_std[ss] = np.array([np.std(grad_Te_fit_dist[:,rr][mask_3std_Te[:,rr]]) for rr in range(len(roa_kp))])

        z_score = 2

        conf_up = fit_mean + z_score*fit_std 
        conf_down = fit_mean - z_score*fit_std 
    
        num_out = 0
        in_mask = np.ones(fit_dist.shape[0], dtype=bool)
        outlier_inds = []

        ind_cut = 0 # index from where to begin the confidence interval (0 is full profile, 150 is outer half)

        for sss in range(len(num_mc)):
                for mc in range(max_iter):
   
                    prof_out = np.logical_or(fit_dist[mc][ind_cut:] > conf_up[0,ind_cut:], fit_dist[mc][ind_cut:] < conf_down[0,ind_cut:])

                    if np.sum(prof_out) > 15:
                        num_out += 1
                        outlier_inds.append(mc)
                
        keep_mask[outlier_inds] = False
        removed_inds = np.union1d(outlier_inds, xsqr_inds)

        print('removed inds: {} ({}, {})'.format(len(removed_inds), len(outlier_inds), len(xsqr_inds)))

        if plot:
            conf_up = fit_mean + fit_std 
            conf_down = fit_mean - fit_std 
       
            maxval = np.max(raw.y[raw.X[:,0] > 0.85])
            headroom = 1.05      

            ax.plot(self.x, conf_up[0], 'r-')
            ax.plot(self.x, conf_down[0], 'r-')
        
            ax.set_xlim([0.85,1.1])
            ax.set_ylim([0, maxval*headroom])
        
            # plot chi_square distribution
            fig, ax = plt.subplots()
            ax[0].hist(xsqr_dist, bins=50)
    
    
            # plot and recompute statistics w/o outliers

            fig, ax = plt.subplots()
            ax.errorbar(raw.X[:-1,0], raw.y[:-1], raw.err_y[:-1], fmt='.')
            ax.set_ylabel('$n_{e}$ (10$^{20}$m$^{-3})$')
            ax.set_xlabel('r/a')
        
            for ii in range(max_iter):
                if ii not in ne_outlier_inds:
                    ax.plot(rhop_kp, fit_dist[ii], 'k-')
                    ax.plot(rhop_kp, conf_up[0], 'r-')
                    ax.plot(rhop_kp, conf_down[0], 'r-')
                
            ax.set_xlim([0.85,1.1])
            ax.set_ylim([0, maxval*headroom])
        

        # final fits
        fit_dist_filtered = fit_dist[keep_mask]
        grad_fit_dist_filtered = grad_fit_dist[keep_mask]

        # recompute statistics
        for ss in range(len(num_mc)):
        
            fit_mean[ss] = np.mean(fit_dist[:num_mc[ss]][keep_mask], axis=0)
            fit_std[ss] = np.std(fit_dist[:num_mc[ss]][keep_mask], axis=0)
            grad_fit_mean[ss] = np.mean(grad_fit_dist[:num_mc[ss]][keep_mask], axis=0)
            grad_fit_std[ss] = np.std(grad_fit_dist[:num_mc[ss]][keep_mask], axis=0)

        self.fit_dist_prefilt = fit_dist
        self.grad_fit_dist_prefilt = grad_fit_dist
        
        self.fit_dist = fit_dist_filtered
        self.grad_fit_dist = grad_fit_dist_filtered
       
        new_filt = True # filter just by fitting Gaussian instead
    
        if not new_filt:
            self.fit_mean = fit_mean[0] # need to index like this because of earlier test done to find what best MC number is
            self.fit_std = fit_std[0]
            self.grad_fit_mean = grad_fit_mean[0]
            self.grad_fit_std = grad_fit_std[0]
        
        else:
            from scipy.stats import norm

            for jj in range(npts_fit):
        
                if np.isnan(self.fit_dist_prefilt[:,jj]).all():
                    self.grad_fit_mean, self.grad_fit_std = np.nan, np.nan
                
                else:
                    nonan_fit_dist_prefilt = ~np.isnan(self.fit_dist_prefilt[:,jj])
                    nonan_grad_fit_dist_prefilt = ~np.isnan(self.grad_fit_dist_prefilt[:,jj])

            fit_stats = np.array([norm.fit(self.fit_dist_prefilt[nonan_fit_dist_prefilt,jj]) for jj in range(npts_fit)])
            grad_fit_stats = np.array([norm.fit(self.grad_fit_dist_prefilt[nonan_grad_fit_dist_prefilt,jj]) for jj in range(npts_fit)])
            self.fit_mean, self.fit_std = fit_stats.transpose()        
            self.grad_fit_mean, self.grad_fit_std = grad_fit_stats.transpose()        

        return None
   
 
    def save_coefs(self, fit_popt, mean_popt):
    
        self.popt = np.vstack((fit_popt, mean_popt))
   
    def interpolate_new_grid(self, new_grid):
    
        self.x_interp = new_grid
        self.y_interp = interp1d(self.x, self.y, bounds_error=False, fill_value=None)(self.x_interp)

        self.y_dist_prefilt_interp = interp1d(self.x, self.fit_dist_prefilt, axis=1, bounds_error=False, fill_value=None)(self.x_interp)
        self.y_dist_interp = interp1d(self.x, self.fit_dist, axis=1, bounds_error=False, fill_value=None)(self.x_interp)


def get_CMOD_1D_geom(shot,time):

    # right gap
    analysis = MDSplus.Tree('analysis', shot)
    tmp = analysis.getNode('\\analysis::top.efit.results.a_eqdsk.oright')
    # tmp = omfit_mds.OMFITmdsValue(server='CMOD', treename='ANALYSIS', shot=shot,
                        # TDI='\\ANALYSIS::TOP.EFIT.RESULTS.A_EQDSK.ORIGHT')
    time_vec = tmp.dim_of(0)
    _gap_R = tmp.data()
    gap_R = _gap_R[time_vec.searchsorted(time)-1]

    # R location of LFS LCFS
    tmp = analysis.getNode('\\analysis::top.efit.results.g_eqdsk.rbbbs')
    # tmp = omfit_mds.OMFITmdsValue(server='CMOD', treename='ANALYSIS', shot=shot,
                        # TDI='\\ANALYSIS::TOP.EFIT.RESULTS.G_EQDSK.RBBBS')
    time_vec = tmp.dim_of(0)
    _rbbbs = tmp.data()*1e2 # m --> cm
    rbbbs = _rbbbs[:,time_vec.searchsorted(time)-1]

    Rsep = np.max(rbbbs)

    return Rsep,gap_R


### look into trying to get gradient scale length of temperature and comparing to lambda_q

def calculate_lambdaT(p_Te, geqdsk, plot_lambdas=False):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''


    Rsep = aurora.rad_coord_transform(1.0, 'r/a', 'Rmid', geqdsk)
    rminor = Rsep - geqdsk['RMAXIS'] 
    
    def decay_length(x, a, b):
        return np.exp(-(x-b)/a)

    def decay_length_log(x, a, b):
        return a*x + b


    from scipy.optimize import curve_fit

    min_scan = [0.97,0.975,0.98,0.985,0.99]
    Rend = 1.02
    
    #min_width=4
    #max_width=8
    #width_scan = np.linspace(min_width,max_width,50)/1e3/rminor

    Rrange = [0.95, 1.05]
     
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>0.98, p_Te.err_y>0.2))
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>1.0, p_Te.err_y>0.1))
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>1.0, p_Te.y>0.25)) # 250 eV
    p_Te.remove_points(np.logical_and(p_Te.X[:,0]>0.9, p_Te.err_y>0.5))  # 500 eV
    
    mask_fit = (p_Te.X[:,0] > Rrange[0]) & (p_Te.X[:,0] < Rrange[1])
    mask_fit_plot = (p_Te.X[:,0] > (Rrange[0]-0.01)) & (p_Te.X[:,0] < (Rrange[1]+0.01))
    
    popt, pcov = curve_fit(decay_length, p_Te.X[:,0][mask_fit], p_Te.y[mask_fit], sigma=p_Te.err_y[mask_fit])
#   popt_log, pcov_log = curve_fit(decay_length_log, p_Te.X[:,0][mask_fit], np.log(p_Te.y[mask_fit]), sigma=p_Te.err_y[mask_fit])

    # let's do a linear least squares fit
    xlog_fit = p_Te.X[:,0][mask_fit]
    ylog_fit = np.log(p_Te.y[mask_fit])
    #yerrlog_fit = np.log(p_Te.err_y[mask_fit])
    yerrlog_fit = p_Te.err_y[mask_fit]
    
    Alog_mat = np.vstack([xlog_fit, np.ones(len(xlog_fit))]).T
    Wlog_mat = np.sqrt(np.diag(np.abs(1/yerrlog_fit)))
    Aw = np.dot(Wlog_mat, Alog_mat)
    yw = np.dot(ylog_fit, Wlog_mat)

    popt_log = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    
    rss = np.sum((np.exp(ylog_fit) - np.exp(popt_log[1] + xlog_fit*popt_log[0]))**2)
    cov_mat = np.linalg.inv(np.dot(Alog_mat.T, Alog_mat))*rss

    lam_T_lin = -1/popt_log[0]*rminor*1e3
    lam_T_lin_unc = np.sqrt(cov_mat[0,0])

    xx = np.linspace(Rrange[0], Rrange[1], 100)
    lam_T_nl = popt[0]*rminor*1e3
    lam_T_nl_unc = np.sqrt(np.abs(pcov[0,0])*rminor*1e3)

    if plot_lambdas:
        fig, ax = plt.subplots()
        ax.plot(p_Te.X[:,0], np.log(p_Te.y), 'o', markersize=14)
        ax.plot(xx, popt_log[1] + xx*popt_log[0], lw=2, c='C4')
        ax.set_xlim([0.965, 1.045])
        ax.set_xlabel('r/a', fontsize=14)
        ax.set_ylabel('$T_{e}$ (eV)', fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        fig,ax = plt.subplots()
        ax.errorbar(p_Te.X[:,0], p_Te.y, yerr=p_Te.err_y, fmt='.', markersize=14)
        ax.axvline(1, linestyle='--', color='gray', lw=2)  
        ax.plot(xx, decay_length(xx, *popt), lw=2, label='$\\lambda_{q}$' + ' (mm) = {:.2f} $\pm$ {:.2f}'.format(2/7*lam_T_nl, 2/7*lam_T_nl_unc))
        ax.plot(xx, np.exp(decay_length_log(xx, *popt_log)), lw=2, label='$\\lambda_{q}$' + ' (mm) = {:.2f} $\pm$ {:.2f}'.format(2/7*lam_T_lin, 2/7*lam_T_lin_unc), c='C4')
        ax.set_xlim([0.965, 1.045])
        ax.set_ylim([np.min(p_Te.y[mask_fit_plot]-p_Te.err_y[mask_fit_plot])*.85, np.max(p_Te.y[mask_fit_plot]+p_Te.err_y[mask_fit_plot])*1.05])
        ax.set_xlabel('r/a', fontsize=14)
        ax.set_ylabel('$T_{e}$ (eV)', fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(loc='best', fontsize=14)
    

    lambda_Ts = []
    for width in width_scan:
        Rrange = [0.99, 1.02]
        maskSOL = (p_Te.X[:,0] > Rrange[0]) & (p_Te.X[:,0] < Rrange[1])
        popt, pcov = curve_fit(decay_length, p_Te.X[:,0][maskSOL], p_Te.y[maskSOL], sigma=p_Te.err_y[maskSOL])
        xx = np.linspace(Rrange[0], Rrange[1], 100)

        lambda_Ts.append(popt[0]*rminor*1e3)   

    print(lambda_Ts)
    plot_lambdaT_fit = False
    if plot_lambdaT_fit:    
        width_scan_toplot = np.array([4,6,8])/1e3/rminor
        
        fig,ax = plt.subplots()
        ax.errorbar(p_Te.X[:,0], p_Te.y, yerr=p_Te.err_y, fmt='.', markersize=14)
        ax.axvline(1, linestyle='--', color='gray', lw=2)  
        
        for width in width_scan_toplot:
            Rrange = [1-width, 1+width]
            maskSOL = (p_Te.X[:,0] > Rrange[0]) & (p_Te.X[:,0] < Rrange[1])
            popt, pcov = curve_fit(decay_length, p_Te.X[:,0][maskSOL], p_Te.y[maskSOL], sigma=p_Te.err_y[maskSOL])
            xx = np.linspace(Rrange[0], Rrange[1], 100)

            ax.plot(xx, decay_length(xx, *popt), lw=2, label='$\\lambda_{T}$' + ' (mm) = {:.2f}'.format(popt[0]*rminor*1e3))
        
        ax.set_xlim([1-width_scan[-1]-0.01, 1+width_scan[-1]+0.01])
        ax.set_xlabel('r/a', fontsize=14)
        ax.set_ylabel('$T_{e}$ (eV)', fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(loc='best', fontsize=14)
        plt.show()
    
    mean_lambdaT = np.mean(lambda_Ts) 
    std_lambdaT = np.mean(lambda_Ts) 
    zscore_lim = 3

    in_lambdaT = []
    for lT in lambda_Ts:
        zscore = np.abs(lT-mean_lambdaT)/std_lambdaT
        if zscore < zscore_lim:
            in_lambdaT.append(lT)
        

    print('lambda_{T} (mm)'+' = {:.2f}'.format(np.mean(in_lambdaT)))
    print('lambda_{T} unc (mm)'+' = {:.2f}'.format(np.std(in_lambdaT)))
    
    return lam_T_nl    


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_P_ohmic(shot):
    ''' Get Ohmic power

    Translated/adapted from scopes:
    _vsurf =  deriv(smooth1d(\ANALYSIS::EFIT_SSIBRY,2))*$2pi ;
    _ip=abs(\ANALYSIS::EFIT_AEQDSK:CPASMA);
    _li = \analysis::efit_aeqdsk:ali;
    _L = _li*6.28*67.*1.e-9;
    _vi = _L*deriv(smooth1d(_ip,2));
    _poh=_ip*(_vsurf-_vi)/1.e6
    '''

    # psi at the edge:
    analysis = MDSplus.Tree('analysis', shot)
    ssibry_node = analysis.getNode('\\analysis::efit_ssibry')
    # ssibry_node = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::efit_ssibry')
    time = ssibry_node.dim_of(0).data()
    ssibry = ssibry_node.data()
    
    # total voltage associated with magnetic flux inside LCFS
    vsurf = np.gradient(smooth(ssibry,5),time) * 2 * np.pi

    # calculated plasma current
    ip_node = analysis.getNode('\\analysis::efit_aeqdsk:cpasma')
    # ip_node= OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:CPASMA')
    ip = np.abs(ip_node.data())

    # internal inductance
    li = analysis.getNode('\\analysis::efit_aeqdsk:ali').data()
    # li = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:ali').data()

    R_cm = 67.0 # value chosen/fixed in scopes
    L = li*2.*np.pi*R_cm*1e-9  # total inductance (nH)
    
    #vi = L * np.gradient(smooth(ip,2),time)   # induced voltage
    vi = L * np.gradient(smooth(ip,2),time) + 0.5*ip*np.gradient(smooth(L,2),time)  # induced voltage - 2nd term added from /home/jwhughes/idl/get_confinement.pro    

    P_oh = ip * (vsurf - vi)/1e6 # P=IV   #MW
    return time, P_oh

    
def get_CMOD_var(var,shot, tmin=None, tmax=None, plot=False, return_time=False):
    ''' Get tree variable for a CMOD shot. If a time window is given, the value averaged over that window is returned,
    or else the time series is given.  See list below for acceptable input variables.
    '''

    if var=='Bt':
        magnetics = MDSplus.Tree('magnetics', shot)
        node = magnetics.getNode('\\magnetics::Bt')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='magnetics',TDI='\\magnetics::Bt')
    elif var=='Bp':
        # use Bpolav, average poloidal B field --> see definition in Silvagni NF 2020
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\EFIT_AEQDSK:bpolav')
        # node = OMFITmdsValue(server='CMOD',shot=shot,treename='analysis', TDI='\EFIT_AEQDSK:bpolav')
    elif var=='Ip':
        magnetics = MDSplus.Tree('magnetics', shot)
        node = magnetics.getNode('\\magnetics::Ip')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='magnetics',TDI='\\magnetics::Ip')
    elif var=='nebar':
        electrons = MDSplus.Tree('electrons', shot)
        node = electrons.getNode('\\electrons::top.tci.results:nl_04')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons',TDI='\\electrons::top.tci.results:nl_04')
    elif var=='P_RF':
        RF = MDSplus.Tree('RF', shot)
        node = RF.getNode('\\RF::RF_power_net')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='RF',TDI='\\RF::RF_power_net')
    elif var=='P_rad_main':
        try: 
            spectroscopy = MDSplus.Tree('spectroscopy', shot)
            node = spectroscopy.getNode('\\spectroscopy::top.bolometer:results:foil:main_power')
            data = node.data()
            t = node.dim_of(0).data()
            # node = OMFITmdsValue(server='CMOD', shot=shot, treename='spectroscopy',TDI='\\spectroscopy::top.bolometer:results:foil:main_power') # W
            # data = node.data() # just as a check if data exists
            # t = node.dim_of(0)
        except:
            data,t = None, None
    elif var=='P_rad_diode':
        spectroscopy = MDSplus.Tree('spectroscopy', shot)
        node = spectroscopy.getNode('\\spectroscopy::top.bolometer:twopi_diode')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='spectroscopy',TDI='\\spectroscopy::top.bolometer:twopi_diode') # kW
    elif var=='p_D2':
        edge = MDSplus.Tree('edge', shot)
        node = edge.getNode('\\edge::top.gas.ratiomatic.f_side')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::TOP.GAS.RATIOMATIC.F_SIDE')  # mTorr
    elif var=='p_E_BOT_MKS':
        edge = MDSplus.Tree('edge', shot)
        node = edge.getNode('\\edge::e_bot_mks')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::E_BOT_MKS')  # mTorr   #lower divertor
    elif var=='p_B_BOT_MKS':
        edge = MDSplus.Tree('edge', shot)
        node = edge.getNode('\\edge::b_bot_mks')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::B_BOT_MKS')     # mTorr  # lower divertor
    elif var=='p_F_CRYO_MKS':
        edge = MDSplus.Tree('edge', shot)
        node = edge.getNode('\\edge::f_cryo_mks')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::F_CRYO_MKS')     # mTorr, only post 2006
    elif var=='p_G_SIDE_RAT':
        edge = MDSplus.Tree('edge', shot)
        node = edge.getNode('\\edge::g_side_rat')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE',TDI='\\EDGE::G_SIDE_RAT')   
    elif var=='q95':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:qpsib')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:qpsib')
    elif var=='Wmhd':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:wplasm')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:wplasm')
    elif var=='dWdt':
        t,data = get_dWdt(shot, tmin=tmin, tmax=tmax)   # tries to fit Wmhd and get gradient
    elif var=='areao':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:areao')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:areao')
    elif var=='betat':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:betat')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:betat')
    elif var=='betap':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:betap')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:betap')
    elif var=='P_oh':
        t,data = get_P_ohmic(shot)   # accurate routine to estimate Ohmic power
    elif var=='li':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:ali')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='ANALYSIS',TDI='\\analysis::EFIT_AEQDSK:ali')
    elif var=='h_alpha':
        spectroscopy = MDSplus.Tree('spectroscopy', shot)
        node = spectroscopy.getNode('\\ha_2_bright')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='spectroscopy', TDI='\ha_2_bright')
    elif var=='cryo_on':
        edge = MDSplus.Tree('edge', shot)
        ndoe = edge.getNode('\\edge::top.cryopump:message')
        # node = OMFITmdsValue(server='CMOD',shot=shot, treename='EDGE', TDI='\EDGE::TOP.CRYOPUMP:MESSAGE')
    elif var=='ssep':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:ssep')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:ssep')
    elif var=='Lgap':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:oleft')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:oleft')
    elif var=='Rgap':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:oright')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:oright')
    elif var=='kappa':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:eout')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:eout')
    elif var=='Udelta':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:doutu')
        node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:doutu')
    elif var=='Ldelta':
        analysis = MDSplus.Tree('analysis', shot)
        node = analysis.getNode('\\efit_aeqdsk:doutl')
        # node = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:doutl')
    else:
        raise ValueError('Variable '+var+' was not recognized!')


    if var not in ['P_oh','dWdt', 'P_rad_main']: 
        data = node.data()
        t = node.dim_of(0)

        if var=='p_E_BOT_MKS' or var=='p_B_BOT_MKS' or var=='p_F_CRYO_MKS':  # anomalies in data storage
            if data is not None:
                data = data[0,:]
    
    if var=='P_rad_diode':
        #if radvar == 'main', no need to scale
        if data is not None:
            # From B.Granetz's matlab scripts: factor of 4.5 from cross-calibration with 2pi_foil during flattop
            # NB: Bob's scripts mention that this is likely not accurate when p_rad (uncalibrated) <= 0.5 MW
            #data *= 4.5
            data *= 3 # suggestion by JWH
            # data from the twopi_diode is output in kW. Change to MW for consistency
            data /= 1e3
    if var=='P_rad_main':
        # if radvar == 'main', just need to convert to MW
        try:
            data /= 1e6
        except:
            print('No P_rad_main')

    if var=='nebar':
        # nl needs to be divided by the chord length
        node_l = analysis.getNode('\\efit_aeqdsk:rco2v')
        # node_l = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:RCO2V')

        try:
            l04_m = node_l.data()[:,3]/1e2 # 4th channel, convert cm to m
            t_l04 = node_l.dim_of(1).data()

            # need to interpolate onto nl04 timebase
            from scipy.interpolate import interp1d
            l04_m_tb = interp1d(t_l04, l04_m, bounds_error=False, fill_value='extrapolate')(t)
       
            zero_inds = np.where(l04_m_tb == 0)[0]
            l04_m_tb[zero_inds] = 1e-10 # dummy to avoid error

            if len(data.shape) > 1: # error catcher
                data = data[:,0]
            data /= l04_m_tb # nl04/l04
        except:
            print('ne_l or chord length not available')
            t, data = np.nan, np.nan
    
    if var=='ssep':
        mask_ssep = np.logical_and(data<3, data>-3)
        data = data[mask_ssep]

    if plot:
        plt.figure()
        plt.plot(t,data)
        plt.xlabel('time [s]')
        plt.ylabel(var)

    if tmin is not None and tmax is not None and t is not None:
        tidx0 = np.argmin(np.abs(t - tmin))
        tidx1 = np.argmin(np.abs(t - tmax))
        return np.mean(data[tidx0:tidx1])
    else:
        if return_time:
            return t, data
        else:
            return data


def load_fmp_neTe(shot, tmin, tmax, loc='outer', get_max=False, plot=False):
    '''Load slow ne and Te from Flush Mounted Probes (FMP) on the divertor from the nodes
    \EDGE::top.probes.fmp.osd_0{ii}.p0.*e_Slow

    If get_max=True, returns the maximum of all the loaded probe signals over time. Otherwise, return
    individual signals.
    '''
    ne_fmp = []
    Te_fmp = []
    Js_fmp = []
    rho_fmp = []
    t_fmp = []

    probe_list = check_probes_avail(shot, loc)

    for probe_str in probe_list:
        edge = MDSplus.Tree('edge')
        node_ne = edge.getNode(probe_str+'ne_slow')
        # node_ne = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI=probe_str+'ne_slow')
        node_Te = edge.getNode(probe_str+'Te_slow')
        # node_Te = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI=probe_str+'Te_slow')
        node_Js = edge.getNode(probe_str+'Js_slow')
        # node_Js = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI=probe_str+'Js_slow')

        # need to interpolate this onto ne/Te timebase
        if node_ne.data() is None:
            continue
        
        node_time = node_ne.dim_of(0).data() # should be same for all vars

        _node_rho = edge.getNode(probe_str+'rho')
        # _node_rho = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI=probe_str+'rho')
        rho_data = interp1d(_node_rho.dim_of(0).data(), _node_rho.data(), bounds_error=False)(node_time)

        in_window = (node_time > tmin) & (node_time < tmax)
        num_inds = np.sum(in_window)

        for iw in range(num_inds):
            t_fmp.append(node_time[in_window][iw])
            rho_fmp.append(rho_data[in_window][iw])
            ne_fmp.append(node_ne.data()[in_window][iw])
            Te_fmp.append(node_Te.data()[in_window][iw])
            Js_fmp.append(node_Js.data()[in_window][iw])
    
    # turn them into arrays
    t_fmp = np.array(t_fmp)
    rho_fmp = np.array(rho_fmp)
    ne_fmp = np.array(ne_fmp)
    Te_fmp = np.array(Te_fmp)
    Js_fmp = np.array(Js_fmp)

    if loc == 'upper':

        # if we're upper, we want to find the minimum rho
        rho_in_fmp, rho_out_fmp = [], []
        ne_in_fmp, ne_out_fmp = [], []
        Te_in_fmp, Te_out_fmp = [], []
        Js_in_fmp, Js_out_fmp = [], []

        t_even = np.round(t_fmp, 3) # this puts them all in the same tbase
        t_even_set = set(t_even)

        for tt in t_even_set:
        
            mask_t = tt == t_even

            rho_masked = rho_fmp[mask_t]
            ne_masked = ne_fmp[mask_t]
            Te_masked = Te_fmp[mask_t]
            Js_masked = Js_fmp[mask_t]

            # find where rho is min at that t
            min_rho = np.min(rho_masked)
            min_rho_ind = np.where(rho_masked == min_rho)[0][0]

            # need to add all w/ rho_ind < min_rho_ind to the in one 
            rho_in_fmp.append(rho_masked[:min_rho_ind])
            ne_in_fmp.append(ne_masked[:min_rho_ind])    
            Te_in_fmp.append(Te_masked[:min_rho_ind])    
            Js_in_fmp.append(Js_masked[:min_rho_ind])    

            # need to add all w/ rho_ind > min_rho_ind to the out one 
            rho_out_fmp.append(rho_masked[min_rho_ind:])
            ne_out_fmp.append(ne_masked[min_rho_ind:])    
            Te_out_fmp.append(Te_masked[min_rho_ind:])    
            Js_out_fmp.append(Js_masked[min_rho_ind:])    

        # flatten them and turn them into arrays
        rho_in_fmp = np.array([item for sublist in rho_in_fmp for item in sublist])
        ne_in_fmp = np.array([item for sublist in ne_in_fmp for item in sublist])
        Te_in_fmp = np.array([item for sublist in Te_in_fmp for item in sublist])
        Js_in_fmp = np.array([item for sublist in Js_in_fmp for item in sublist])
    
        rho_out_fmp = np.array([item for sublist in rho_out_fmp for item in sublist])
        ne_out_fmp = np.array([item for sublist in ne_out_fmp for item in sublist])
        Te_out_fmp = np.array([item for sublist in Te_out_fmp for item in sublist])
        Js_out_fmp = np.array([item for sublist in Js_out_fmp for item in sublist])

    '''

        ne_fmp.append(node_ne.data())
        Te_fmp.append(node_Te.data())
        Js_fmp.append(node_Js.data())
        t_fmp.append(node_ne.dim_of(0))
        rho_fmp.append(rho_data)

    
    ne_fmp_interp = np.zeros((len(ne_fmp),200)) # 200 time points is enough, usually ~100 in signals
    Te_fmp_interp = np.zeros((len(Te_fmp),200)) # 200 time points is enough, usually ~100 in signals
    Js_fmp_interp = np.zeros((len(Js_fmp),200)) # 200 time points is enough, usually ~100 in signals
    rho_fmp_interp = np.zeros((len(rho_fmp),200)) # 200 time points is enough, usually ~100 in signals
               
    # each probe has a different time base. Interpolate and then sum
    tmin = np.min([np.min(tlist) for tlist in t_fmp])
    tmax = np.max([np.max(tlist) for tlist in t_fmp])
    time = np.linspace(tmin, tmax, ne_fmp_interp.shape[1])
        
    for ii in np.arange(len(ne_fmp)):
        ne_fmp_interp[ii,:] = interp1d(t_fmp[ii], ne_fmp[ii], bounds_error=False)(time)
        Te_fmp_interp[ii,:] = interp1d(t_fmp[ii], Te_fmp[ii], bounds_error=False)(time)
        Js_fmp_interp[ii,:] = interp1d(t_fmp[ii], Js_fmp[ii], bounds_error=False)(time)
        rho_fmp_interp[ii,:] = interp1d(t_fmp[ii], rho_fmp[ii], bounds_error=False)(time)
        '''

   
    if get_max:

        ne_fmp_max = np.nanmax(ne_fmp, axis=0)
        Te_fmp_max = np.nanmax(Te_fmp, axis=0)
        Js_fmp_max = np.nanmax(Js_fmp, axis=0)

        rho_ne_max = rho_fmp[(ne_fmp_max == ne_fmp)]
        rho_Te_max = rho_fmp[(Te_fmp_max == Te_fmp)]
        rho_Js_max = rho_fmp[(Js_fmp_max == Js_fmp)]

        '''
        if plot:
            fig,ax = plt.subplots()
            ax.plot(time, ne_fmp_max)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(r'$n_e$ FMP max [m$^{-3}$]')
            
            fig,ax = plt.subplots()
            ax.plot(time, Te_fmp_max)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(r'$T_e$ FMP max [eV]')
                
            fig,ax = plt.subplots()
            ax.plot(time, Js_fmp_max)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(r'$J_s$ FMP max [Am^{-2}]')
        '''

        return ne_fmp_max, rho_ne_max, Te_fmp_max, rho_Te_max, Js_fmp_max, rho_Js_max
    
    if not get_max and plot:
        fig1,ax1 = plt.subplots()
        fig2,ax2 = plt.subplots()
        fig3,ax3 = plt.subplots()
        
        for elem in np.arange(len(ne_fmp)):
            ax1.plot(t_fmp[elem], ne_fmp[elem], label=f'{elem}')
            ax2.plot(t_fmp[elem], Te_fmp[elem], label=f'{elem}')
            ax3.plot(t_fmp[elem], Js_fmp[elem], label=f'{elem}')
            
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel(r'$n_e$ [m$^{-3}$]')
        plt.tight_layout(); plt.legend()
        ax2.set_xlabel('time [s]')
        ax2.set_ylabel(r'$T_e$ [m$^{-3}$]')
        plt.tight_layout(); plt.legend()
        ax3.set_xlabel('time [s]')
        ax3.set_ylabel(r'$J_s$ [Am$^{-2}$]')
        plt.tight_layout(); plt.legend()

    if loc == 'upper':
        return ne_out_fmp, rho_out_fmp, Te_out_fmp, rho_out_fmp, Js_out_fmp, rho_out_fmp, ne_in_fmp, rho_in_fmp, Te_in_fmp, rho_in_fmp, Js_in_fmp, rho_in_fmp

    else:
        return ne_fmp, rho_fmp, Te_fmp, rho_fmp, Js_fmp, rho_fmp


def check_probes_avail(shot, loc):
    
    # probe info as found in load_[o/i/u]div.m in /home/kuang/Matlab
    if loc == 'outer': 
        probe_num = ['01','02','03','04','05','06','07','08','09','10']
        probe_head = ['P0','P2']
        probe_type = 'fmp'
        probe_loc = 'osd'
    elif loc == 'inner':
        probe_num = ['01','04','07','10','13','16']
        probe_head = ['P0','P1','P2']
        probe_type = 'fmp'
        probe_loc = 'id'
    elif loc == 'upper':
        probe_num = ['01','04','07','10','13']
        probe_head = ['P0','P1','P2']
        probe_type = 'udiv'
        probe_loc = 'ud'
    else:
        print('Requested probe location not valid')

    probes_avail = []

    edge = MDSplus.Tree('edge', shot)

    for pn in probe_num:
        for ph in probe_head:
            node_str = '\EDGE::top.probes.{}.{}_{}.{}:'.format(probe_type, probe_loc, pn, ph)
    
            node_ne = edge.getNode(node_str+'ne_slow')
            # node_ne = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE', TDI=node_str+'ne_slow')

            try: 
                if len(node_ne.data()) < 100: # this is from AK load_{}div.m
                    print('Warning: {} data was bad'.format(node_str))
                else:
                    probes_avail.append(node_str)
            except:
                print('Warning: {} data was bad'.format(node_str))

    return probes_avail

def get_dWdt(shot, tmin=None, tmax=None, plot=False):
    '''Function to do X

    This function does X by doing Y   
 
    Parameters
    ----------
    parameter : type
        This is a parameter

    Returns
    -------
    returned : type
        This is what is returned

    '''
        
    analysis = MDSplus.Tree('analysis', shot)
    node = analysis.getNode('\\efit_aeqdsk:wplasm')
    # node = OMFITmdsValue(server='CMOD',shot=shot, treename='analysis', TDI='\EFIT_AEQDSK:wplasm')
    
    Wmhd = node.data()
    t = node.dim_of(0).data()

    def powerlaw(x,a,b):
        return a*x**b


    from scipy.interpolate import interp1d, UnivariateSpline
    from scipy.optimize import curve_fit
    
    if tmin is not None and tmax is not None:
        tidx0 = np.argmin(np.abs(t - (tmin-0.05))) # give some slack for better fit
        tidx1 = np.argmin(np.abs(t - (tmax+0.05))) # give some slack for better fit

        # cut the window        
        t_win = t[tidx0:tidx1]
        Wmhd_win = Wmhd[tidx0:tidx1]

        tt = np.linspace(t_win[0], t_win[-1], int(len(t_win)*10)) # increase resolution by a lot to help derivative
        popt, pcov = curve_fit(powerlaw, t_win, Wmhd_win)
        
        Wmhd_fit = powerlaw(tt, *popt)        
        grad_Wmhd_fit = np.gradient(Wmhd_fit, tt)
        
        return tt, grad_Wmhd_fit/1e6


    else:
        tt = np.linspace(t[0], t[-1], int(len(t)*10)) # increase resolution by a lot to help derivative
        popt, pcov = curve_fit(powerlaw, t, Wmhd) # probably a bad function to use for the whole time trace - would need to check this
        
        Wmhd_fit = powerlaw(tt, *popt)        
        grad_Wmhd_fit = np.gradient(Wmhd_fit, tt)
        
        # interpolate back to t grid
        grad_Wmhd = interp1d(tt, grad_Wmhd_fit)(t)
    
        return t,grad_Wmhd/1e6


def wrap_qcm(shot, tmin, tmax, samples_exp=7, plot=False, plot_chan='10'):

    n_chans = np.arange(1,32)

    # lists to hold outputs
    channels = []
    peak_centers = []
    peak_heights = []
    peak_fwhm = []
    peak_fwfm = []
    f_str = []
    h_str = []

    for nc in range(len(n_chans)):

        if plot and int(plot_chan) == n_chans[nc]:
            _out = find_pci_qcm(shot, tmin=tmin, tmax=tmax, samples_exp=7, channel='{}'.format(str(n_chans[nc])), plot=True)

        else:
            _out = find_pci_qcm(shot, tmin=tmin, tmax=tmax, samples_exp=7, channel='{}'.format(str(n_chans[nc])), plot=False)
                    


        pc, ph, pfh, pff, fs, hs = _out

        num_peaks = len(pc)

        for oo in range(num_peaks):
            channels.append(nc)
            peak_centers.append(pc[oo])
            peak_heights.append(ph[oo])
            peak_fwhm.append(pfh[oo])
            peak_fwfm.append(pff[oo])
            f_str.append(fs[oo])
            h_str.append(hs[oo])

    # keep peaks that are within 1std of the median
    median_peak = np.nanmedian(peak_centers)
    std_peak = np.nanstd(peak_centers)

    peak_down = median_peak - std_peak
    peak_up = median_peak + std_peak

    mask_peaks = (peak_centers > peak_down) & (peak_centers < peak_up)
    
    #turn them into arrays
    peak_centers, peak_heights = np.array(peak_centers), np.array(peak_heights) 
    peak_fwhm, peak_fwfm = np.array(peak_fwhm), np.array(peak_fwfm)
    f_str, h_str = np.array(f_str), np.array(h_str)

    qcm = {}

    if len(peak_centers) > 0:

        qcm['center'], qcm['center_unc'] = np.mean(peak_centers[mask_peaks]), np.std(peak_centers[mask_peaks])
        qcm['height'], qcm['height_unc'] = np.mean(peak_heights[mask_peaks]), np.std(peak_heights[mask_peaks])
        qcm['fwidth'], qcm['fwidth_unc'] = np.mean(peak_fwhm[mask_peaks]), np.std(peak_fwhm[mask_peaks])
        qcm['hwidth'], qcm['hwidth_unc'] = np.mean(peak_fwfm[mask_peaks]), np.std(peak_fwfm[mask_peaks])
        qcm['fstr'], qcm['fstr_unc'] = np.mean(f_str[mask_peaks]), np.std(f_str[mask_peaks])
        qcm['hstr'], qcm['hstr_unc'] = np.mean(h_str[mask_peaks]), np.std(h_str[mask_peaks])

    else:
        qcm['center'], qcm['center_unc'] = np.nan, np.nan
        qcm['height'], qcm['height_unc'] = np.nan, np.nan
        qcm['fwidth'], qcm['fwidth_unc'] = np.nan, np.nan
        qcm['hwidth'], qcm['hwidth_unc'] = np.nan, np.nan
        qcm['fstr'], qcm['fstr_unc'] = np.nan, np.nan
        qcm['hstr'], qcm['hstr_unc'] = np.nan, np.nan

    return qcm


def find_pci_qcm(shot, tmin, tmax, samples_exp=7, channel='10', plot=False):

    if len(channel) == 1: channel = '0' + str(channel)

    '''
    try:
        pcilocal = MDSplus.Tree('pcilocal', shot)
        time = pcilocal.getNode('\\pcilocal::top.results.timebase').data()
        data = pcilocal.getNode('\\pcilocal::top.results.pci_{}'.format(channel)).data()
        time = OMFITmdsValue(server='CMOD', shot=shot, treename='pcilocal', TDI='\\pcilocal::top.results.timebase').data()
        data = OMFITmdsValue(server='CMOD', shot=shot, treename='pcilocal', TDI='\\pcilocal::top.results.pci_{}'.format(channel)).data()

        tmin_ind = np.where(time >= tmin)[0][0]
        tmax_ind = np.where(time > tmax)[0][0]
        num_samples = int(2**samples_exp)

    except:
        print('Channel {} empty'.format(str(channel)))
        return np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan

    '''

    sys.path.append('/home/millerma')
    from MDSpull_python import MDSpull

#    channel = '75 GHz'

    try:
        data75 = MDSpull(shot, 'Reflect', channel)
        time = data75['Time'][:-3]
        data = data75['Mag']
        
        tmin_ind = np.where(time >= tmin)[0][0]        
        tmax_ind = np.where(time > tmax)[0][0]        
        num_samples = int(2**samples_exp)   
 
    except Exception as e:
        print(e)
        print('Chanel {} empty'.format(str(channel)))


    try:
        freqs, pows = get_fourier_spectra(time[tmin_ind:tmax_ind], data[tmin_ind:tmax_ind], num_samples, plot=plot)

        # calculate a moving average
        window_width = 20#int(np.floor(len(freqs)/75)) # Hz
        cumsum_vec = np.cumsum(np.insert(pows, 0, 0))
        y_ma = (cumsum_vec[window_width:] - cumsum_vec[:-window_width])/window_width
        x_ma = np.linspace(freqs[int(window_width/2)], freqs[-int(window_width/2)-1], len(y_ma)) 

        # try to fit an lmode powerlaw spectrum to subtract it out
        def lmode(x, a, b):
            return a*x**b

        def loglaw(x,a,b):
            return a*np.exp(-b*x)

        from scipy.optimize import curve_fit
        #mask_lowk = freqs < 1000
        mask_lowk = freqs < 1000
        #popt, pcov = curve_fit(lmode, freqs[mask_lowk], pows[mask_lowk]/1e13)
        popt, pcov = curve_fit(lmode, freqs[mask_lowk], pows[mask_lowk]*1e3)
        #popt, pcov = curve_fit(loglaw, freqs[mask_lowk], pows[mask_lowk]*1e3)
    
        p_pow = np.polyfit(freqs[mask_lowk], np.log(pows[mask_lowk]*1e3), deg=1)
        #from IPython import embed; embed()
        poly_pow = np.exp(np.polyval(p_pow, freqs))/1e3
        poly_ma = np.exp(np.polyval(p_pow, x_ma))/1e3

        #pows_sub = pows - lmode(freqs, *popt)*1e13
        #y_ma_sub = y_ma - lmode(x_ma, *popt)*1e13
        
        #pows_sub = pows - lmode(freqs, *popt)/1e3
        #y_ma_sub = y_ma - lmode(x_ma, *popt)/1e3

        pows_sub = pows - poly_pow
        y_ma_sub = y_ma - poly_ma

        mask_spline = (x_ma < 200)

        from scipy.interpolate import UnivariateSpline, interp1d
        xspl = np.linspace(np.min(x_ma), 200, 1000)
#        yspl = UnivariateSpline(x_ma[mask_spline], y_ma_sub[mask_spline]/1e17, s=1/5/window_width, k=2)(xspl)*1e17
        yspl = UnivariateSpline(x_ma[mask_spline], y_ma_sub[mask_spline]*1e3, s=5, k=2)(xspl)/1e3
#        from IPython import embed; embed()

    except Exception as e:
        print(e)
        print('Fourier transform failed for channel {}'.format(str(channel)))
        return np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan

    # identify qcm

    # let's try to find them using the scipy function
    from scipy.signal import find_peaks, peak_widths

    try: 
        # find peaks that are at least 40% of the max and have a prominence of 10% 
        peak_inds, prop_dict = find_peaks(yspl, height=np.max(yspl)*2/5, prominence=np.max(yspl/10))
        
        peak_freqs = xspl[peak_inds]
        #peak_heights = interp1d(freqs, pows)(peak_freqs)
        peak_heights = interp1d(xspl, yspl)(peak_freqs)

        num_peaks = len(peak_inds)

        # get the widths
        half_widths = peak_widths(yspl, peak_inds, rel_height=0.5)
        full_widths = peak_widths(yspl, peak_inds, rel_height=1.0)
   
        peak_fwhm_inds, prom_half, lhm_ind, rhm_ind = half_widths
        peak_fwfm_inds, prom_full, lfm_ind, rfm_ind = full_widths

        lhm = xspl[np.ceil(lhm_ind).astype(int)]
        rhm = xspl[np.ceil(rhm_ind).astype(int)]
    
        lfm = xspl[np.ceil(lfm_ind).astype(int)]
        rfm = xspl[np.ceil(rfm_ind).astype(int)]
    
        peak_fwhm = rhm - lhm
        peak_fwfm = rfm - lfm

        # find the mode strength by integrating the fluctuation
        from scipy.integrate import cumtrapz
    
        # try integrating the moving average and spline

        h_strength_spl, f_strength_spl = np.zeros(num_peaks), np.zeros(num_peaks)
        h_strength_ma, f_strength_ma = np.zeros(num_peaks), np.zeros(num_peaks)
        h_strength_sig, f_strength_sig = np.zeros(num_peaks), np.zeros(num_peaks)
           
        for pp in range(num_peaks):
 
            mask_hpeak_ma = (x_ma > lhm[pp]) & (x_ma < rhm[pp])
            mask_fpeak_ma = (x_ma > lfm[pp]) & (x_ma < rfm[pp])

            h_strength_ma[pp] = cumtrapz(y_ma_sub[mask_hpeak_ma], x_ma[mask_hpeak_ma])[-1]
            f_strength_ma[pp] = cumtrapz(y_ma_sub[mask_fpeak_ma], x_ma[mask_fpeak_ma])[-1]
   
            mask_hpeak_spl = (xspl > lhm[pp]) & (xspl < rhm[pp])
            mask_fpeak_spl = (xspl > lfm[pp]) & (xspl < rfm[pp])
    
            h_strength_spl[pp] = cumtrapz(yspl[mask_hpeak_spl], xspl[mask_hpeak_spl])[-1]
            f_strength_spl[pp] = cumtrapz(yspl[mask_fpeak_spl], xspl[mask_fpeak_spl])[-1]
  
            mask_hpeak_sig = (freqs > lhm[pp]) & (freqs < rhm[pp])
            mask_fpeak_sig = (freqs > lfm[pp]) & (freqs < rfm[pp])
    
            h_strength_sig[pp] = cumtrapz(pows_sub[mask_hpeak_sig], freqs[mask_hpeak_sig])[-1]
            f_strength_sig[pp] = cumtrapz(pows_sub[mask_fpeak_sig], freqs[mask_fpeak_sig])[-1]

    except:
        from IPython import embed; embed()
        print('No peaks found for channel {}'.format(str(channel)))
        return np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan


    if plot:

        fig,ax = plt.subplots(1,2)

        ax[0].plot(freqs, pows, zorder=1, label='FFT signal')
        ax[0].plot(x_ma, y_ma, zorder=2, lw=2, label='moving average')        
        #ax[0].plot(freqs, lmode(freqs, *popt)*1e13, lw=3, zorder=3, label='powerlaw fit')
        #ax[0].plot(freqs, lmode(freqs, *popt)/1e3, lw=3, zorder=3, label='powerlaw fit')
        ax[0].plot(freqs, poly_pow, lw=3, zorder=3, label='exp fit')

        fig.suptitle('shot {}, t: [{:.2f}, {:.2f}], channel = {} \n N = {}'.format(shot, tmin, tmax, channel, str(num_samples)), fontsize=16)
        #ax[0].set_title('shot {}, t: [{:.2f}, {:.2f}], channel = {} \n N = {}'.format(shot, tmin, tmax, channel, str(num_samples)), fontsize=16)
        #ax[0].set_xlim([-10, 200]) # QCM normally found below this
        
        ax[0].set_xlabel('f (kHz)', fontsize=14)
        ax[0].legend(loc='best', fontsize=14)
        ax[0].tick_params(axis='x', labelsize=14)
        ax[0].tick_params(axis='y', labelsize=14)    

        ax[1].plot(freqs, pows_sub, zorder=1)
        ax[1].plot(x_ma, y_ma_sub, lw=2, zorder=2)
        ax[1].plot(xspl, yspl, lw=3, zorder=3)
        ax[1].set_xlabel('f (kHz)', fontsize=14)
        ax[1].tick_params(axis='x', labelsize=14)
        ax[1].tick_params(axis='y', labelsize=14)    

        if num_peaks > 0:
            ax[1].plot(peak_freqs, peak_heights, 'X', c='k', markersize=14, zorder=4)
            ax[1].hlines(prom_half, lhm, rhm, linestyle='--', lw=2, color='k', zorder=4)
            ax[1].hlines(prom_full, lfm, rfm, linestyle='--', lw=2, color='b', zorder=4)
        
        ax[1].set_xlim([-10, 200]) # QCM normally found below this

        plt.show()

    return np.array(peak_freqs), np.array(peak_heights), np.array(peak_fwhm), np.array(peak_fwfm), np.array(f_strength_sig), np.array(h_strength_sig)


def get_fourier_spectra(time, data, n_samples, plot=False):

    from scipy.fft import fft, fftfreq
    
    t_range = time[-1] - time[0]
    len_t = len(time)
    
    N = int(np.floor(len_t / n_samples)) # same as num_spectra
    T = t_range / len_t

    yf = fft(data, n=N)
    xf = fftfreq(N, T)/1e3 # in kHz

    return xf[1:N//2], 2.0/N*np.abs(yf[1:N//2])


def compare_ts_tci(shot, tmin, tmax, nl_num=4, plot=False):

    nl_ts1 = 1e32
    nl_ts2 = 1e32
    nl_tci1 = 1e32
    nl_tci2 = 1e32
    ts_time1 = 1e32
    ts_time2 = 1e32


    shot_str = str(shot)
    ch_str = '0' + str(nl_num) if nl_num < 10 else str(nl_num)
    tci_node = '.tci.results:nl_{}'.format(ch_str)

    electrons = MDSplus.Tree('electrons', shot)
    try:
        ts_time = electrons.getNode('\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0).data()
        # ts_time = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0)
        _ = len(ts_time) # this is just to test if it's empty 
    except:
        ts_time = electrons.getNode('\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0).data()
        # ts_time = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0)
    
    tci = electrons.getNode('\\electrons::top{}'.format(tci_node)).data()
    tci_t = electrons.getNode('\\electrons::top{}'.format(tci_node)).dim_of(0).data()
    # tci = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top{}'.format(tci_node)).data()
    # tci_t = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top{}'.format(tci_node)).dim_of(0)

    # for some older shots, tci is stored in 2D array - fix
    if tci.ndim == 2: tci = tci[:,0]

    nl_ts, nl_ts_t = integrate_ts2tci(shot, tmin, tmax, nl_num=nl_num, plot=plot)

    t0 = np.maximum(np.min(nl_ts_t), np.min(tci_t))
    t1 = np.minimum(np.max(nl_ts_t), np.max(tci_t))

    n_yag1, n_yag2, indices1, indices2 = parse_yags(shot)

    
    # laser 1
    if n_yag1 > 0:

        ts_time1 = ts_time[indices1]
        ind = (ts_time1 >=  t0) & (ts_time1 <= t1)
        cnt = np.sum(ind)

        if cnt > 0:

            nl_tci1 = interp1d(tci_t, tci)(ts_time1[ind])
            nl_ts1 = interp1d(nl_ts_t, nl_ts)(ts_time1[ind])
            time1 = ts_time1[ind]

    else:

        time1 = -1

    # laser 2
    if n_yag2 > 0:

        ts_time2 = ts_time[indices2]
        ind = (ts_time2 >=  t0) & (ts_time2 <= t1)
        cnt = np.sum(ind)
        if cnt > 0:
            nl_tci2 = interp1d(tci_t, tci)(ts_time2[ind])
            nl_ts2 = interp1d(nl_ts_t, nl_ts)(ts_time2[ind])
            time2 = ts_time2[ind]

    else:

        time2 = -1

    if plot:
        fig,ax = plt.subplots(2, sharex=True)

        ax[0].plot(time1, nl_ts1, lw=2)
        ax[0].plot(time1, nl_tci1, lw=2)

        ax[1].plot(time2, nl_ts2, lw=2)
        ax[1].plot(time2, nl_tci2, lw=2)

        ax[0].axvline(tmin, color='gray', linestyle='--')
        ax[0].axvline(tmax, color='gray', linestyle='--')

        ax[1].axvline(tmin, color='gray', linestyle='--')
        ax[1].axvline(tmax, color='gray', linestyle='--')


        ax[1].set_xlabel('time (s)', fontsize=14)
        ax[0].set_ylabel('$n_{e,l} (m^{-3})$', fontsize=14)
        ax[1].set_ylabel('$n_{e,l} (m^{-3})$', fontsize=14)

        ax[0].set_title('laser 1', fontsize=14)
        ax[1].set_title('laser 2', fontsize=14)
    
        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].tick_params(axis='both', labelsize=14)

        plt.show()


    return nl_ts1, nl_ts2, nl_tci1, nl_tci2, time1, time2



def integrate_ts2tci(shot, tmin, tmax, nl_num=4, plot=False):

    shot_str = str(shot)

    t, z, n_e, n_e_sig = map_ts2tci(shot, tmin, tmax, nl_num=nl_num)

    n_ts = len(t)
    nl_ts_t = t
    nl_ts = np.zeros(n_ts)

    ### perform integral

    from scipy.integrate import cumtrapz

    # define this silly idl function
    def idl_uniq(x):

        y_inds = []
        for x_ind in range(len(x)):
            y_inds.append(True)
            if x[x_ind] == x[x_ind-1]:
                y_inds[-2] = False

        return np.array(y_inds)


    for i in range(n_ts):

        n_e_sig[i,:] = np.select([n_e_sig[i,:] == 0], [np.nan], n_e_sig[i,:]) # replaces 0s with nans so no error is thrown

        ind = (np.abs(z[i,:]) < 0.5) & (n_e[i,:] > 0) & (n_e[i,:] < 1e21) & (n_e[i,:]/n_e_sig[i,:] > 2)
        cnt = np.sum(ind)

        if cnt < 3:
            nl_ts[i] = 0

        else:
            x = z[i,ind]
            y = n_e[i,ind]
            ind_uniq = idl_uniq(x) # this function returns the indices that have unique adjacent indices, but will only choose the 2nd adjacent index to return to you for some reason
    
            x = x[ind_uniq]
            y = y[ind_uniq]

            nl_ts[i] = cumtrapz(y,x)[-1]

    if plot:

        ch_str = '0' + str(nl_num) if nl_num < 10 else str(nl_num)
        tci_node = '.tci.results:nl_{}'.format(ch_str)
        tci = electrons.getNode('\\electrons::top{}'.format(tci_node)).data()
        tci_t = electrons.getNode('\\electrons::top{}'.format(tci_node)).dim_of(0).data()
        # tci = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top{}'.format(tci_node)).data()
        # tci_t = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top{}'.format(tci_node)).dim_of(0)

        fig,ax = plt.subplots()
        ax.plot(nl_ts_t, nl_ts, label='TS')
        ax.plot(tci_t, tci, label='TCI')

        ax.set_xlabel('time (s)', fontsize=14)
        ax.set_ylabel('$n_{e,l} (m^{-3})$', fontsize=14)
        ax.legend(loc='best', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(tci_node, fontsize=16)

    return nl_ts, nl_ts_t


def map_ts2tci(shot, tmin, tmax, nl_num=4):

    shot_str = str(shot)

    analysis = MDSplus.Tree('analysis', shot)
    psi_a_t = analysis.getNode('\\efit_aeqdsk:sibdry').dim_of(0).data()
    # psi_a_t = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis', TDI='\\EFIT_AEQDSK:sibdry').dim_of(0)

    t1, t2 = np.min(psi_a_t), np.max(psi_a_t) # in JWH's script, it uses efit_times from efit_check, 
                                            # but I'm not sure how to get that so hopefully this is good enough

    ### get thomson data

    # core

    electrons = MDSplus.Tree('electrons', shot)
    try: # needs to be shot > ~1020900000 for this one to work

        t_ts = electrons.getNode('\\electrons:top.yag_new.results.profiles:ne_rz').dim_of(0).data()
        ne_ts_core = electrons.getNode('\\electrons:top.yag_new.results.profiles:ne_rz').data()
        ne_ts_core_err = electrons.getNode('\\electrons:top.yag_new.results.profiles:ne_err').data()
        z_ts_core = electrons.getNode('\\electrons:top.yag_new.results.profiles:z_sorted').data()
        # t_ts = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').dim_of(0)
        # ne_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_rz').data()
        # ne_ts_core_err = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:ne_err').data()
        # z_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_new.results.profiles:z_sorted').data()
        m_ts_core = len(z_ts_core)

    except: # if not, old core TS system

        t_ts = electrons.getNode('\\electrons:top.yag.results.global.profile:ne_rz_t').dim_of(0).data()
        ne_ts_core = electrons.getNode('\\electrons:top.yag.results.global.profile:ne_rz_t').data()
        ne_ts_core_err = electrons.getNode('\\electrons:top.yag.results.global.profile:ne_err_zt').data()
        z_ts_core = electrons.getNode('\\electrons:top.yag.results.global.profile:z_sorted').data()
        # t_ts = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_rz_t').dim_of(0)
        # ne_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_rz_t').data()
        # ne_ts_core_err = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:ne_err_zt').data()
        # z_ts_core = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.global.profile:z_sorted').data()
        m_ts_core = len(z_ts_core)

    # edge

    ne_ts_edge = electrons.getNode('\\electrons::top.yag_edgets.results:ne').data()
    ne_ts_edge_err = electrons.getNode('\\electrons::top.yag_edgets.results:ne:error').data()
    # ne_ts_edge = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_edgets.results:ne').data()
    # ne_ts_edge_err = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_edgets.results:ne:error').data()

    z_ts_edge = electrons.getNode('\\electrons::top.yag_edgets.data:fiber_z').data()
    # z_ts_edge = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag_edgets.data:fiber_z').data()
    m_ts_edge = len(z_ts_edge)


    m_ts = m_ts_core + m_ts_edge

    r_ts = electrons.getNode('\\electrons::top.yag.results.param:r').data()
    r_tci = electrons.getNode('\\electrons::top.tci.results:rad').data()
    # r_ts = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.yag.results.param:r').data()
    # r_tci = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\electrons::top.tci.results:rad').data()


    ### compute mapping

    n_ts = len(t_ts)
    z_ts = np.zeros(m_ts)

    z_ts[:m_ts_core] = z_ts_core
    z_ts[m_ts_core:] = z_ts_edge

    ne_ts = np.zeros((n_ts, m_ts))
    ne_ts_err = np.zeros((n_ts, m_ts))

    ne_ts[:,:m_ts_core] = np.transpose(ne_ts_core)
    ne_ts_err[:,:m_ts_core] = np.transpose(ne_ts_core_err)
    ne_ts[:,m_ts_core:] = np.transpose(ne_ts_edge)
    ne_ts_err[:,m_ts_core:] = np.transpose(ne_ts_edge_err)

    ind = (t_ts > t1) & (t_ts < t2)
    n_ts = np.sum(ind)
    t_ts = t_ts[ind]
    ne_ts = ne_ts[ind,:]
    ne_ts_err = ne_ts_err[ind,:]

    # IMPORTANT: the best way to do this is to use some mapping tool that can take in many time slices and then query the efit for each time point
    # like the IDL routine efit_rz2psi does - for the purposes of M.A.Miller's workflow, let's just assume the equilibrium does not change much and 
    # just use one equilbrium point to perform the mapping - therefore, want to pass in t_min and t_max # this will likely introduce a significant 
    # error in parts of the shot that do not have the same equilbria, so only trust the resulting integrals within the t_min and t_max bounds

    # maybe what's done in profiletools is a solution to this - look into it

    time_efit = (tmin + tmax)/2 # just an average

    geqdsk = get_geqdsk_cmod(
        shot, time_efit*1e3, gfiles_loc = '/home/millerma/lya/gfiles/')

    rhop_ts = np.transpose(aurora.get_rhop_RZ(r_ts, z_ts, geqdsk))
    psin_ts = rhop_ts**2
    psin_ts = np.tile(psin_ts, (n_ts,1)) # only doing this because I just have one geqdsk!

    m_tci = 101
    z_tci = -0.4 + 0.8*np.linspace(1,m_tci-1,m_tci)/np.float(m_tci - 1) ## What is findgen?
    r_tci = r_tci[nl_num-1] + np.zeros(m_tci)

    rhop_tci = np.transpose(aurora.get_rhop_RZ(r_tci, z_tci, geqdsk))
    psin_tci = rhop_tci**2
    psin_tci = np.tile(psin_tci, (n_ts,1)) # only doing this because I just have one geqdsk!


    # perform mapping

    z_mapped = np.zeros((n_ts, int(2*m_ts)))
    ne_mapped = np.zeros_like(z_mapped)
    ne_mapped_err = np.zeros_like(z_mapped)


    for i in range(n_ts):

        psin_min = np.min(psin_tci[i,:])
        i_min = np.argmin(psin_tci[i,:])

        for j in range(m_ts):

            if psin_ts[i,j] >= psin_min:

                a1 = interp1d(psin_tci[i,:i_min+1], z_tci[:i_min+1], bounds_error=False, fill_value='extrapolate')(psin_ts[i,j])
                a2 = interp1d(psin_tci[i,i_min+1:], z_tci[i_min+1:], bounds_error=False, fill_value='extrapolate')(psin_ts[i,j])

                z_mapped[i,[j,j+m_ts]] = [a1,a2]
                ne_mapped[i,[j,j+m_ts]] = ne_ts[i,j]
                ne_mapped_err[i,[j,j+m_ts]] = ne_ts_err[i,j]

        # sort before returning
        ind = np.argsort(z_mapped[i,:])
        z_mapped[i,:] = z_mapped[i,ind]
        ne_mapped[i,:] = ne_mapped[i,ind]
        ne_mapped_err[i,:] = ne_mapped_err[i,ind]

    z = z_mapped
    n_e = ne_mapped
    n_e_sig = ne_mapped_err
    t = t_ts

    return t, z, n_e, n_e_sig


def parse_yags(shot):

    ### PARSE THOSE YAGS!

    electrons = MDSplus.Tree('electrons', shot)
    try:    
        n_yag1 = electrons.getNode('\\knobs:pulses_q').data()[0]
        # n_yag1 = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\knobs:pulses_q').data()[0]
    except:
        n_yag1 = 0
        print('Laser 1 may not be working')
    try:
        n_yag2 = electrons.getNode('\\knobs:pulses_q_2').data()[0]
        # n_yag2 = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\knobs:pulses_q_2').data()[0]
    except:
        n_yag2 = 0
        print('Laser 2 may not be working')

    dark = electrons.getNode('\\n_dark_prior').data()[0]
    n_total = electrons.getNode('\\n_total').data()[0]
    # dark = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\n_dark_prior').data()[0]
    # n_total = OMFITmdsValue(server='CMOD', shot=shot, treename='electrons', TDI='\\n_total').data()[0]
    n_t = n_total - dark

    if n_yag1 == 0:
        if n_yag2 == 0:
            indices1 = -1
            indices2 = -1
        else:
            indices1 = -1
            indices2 = np.linspace(0,n_yag2-1,n_yag2)
    else:
        if n_yag2 == 0:
            indices1 = np.linspace(0,n_yag1-1,n_yag1)
            indices2 = -1
        else:
            if n_yag1 == n_yag2:
                indices1 = 2*np.linspace(0,n_yag1-1,n_yag1)
                indices2 = indices1 + 1
            else:
                if n_yag1 < n_yag2:
                    indices1 = 2*np.linspace(0,n_yag1-1,n_yag1)
                    indices2 = np.hstack((2*np.linspace(0,n_yag1-1,n_yag1)+1,2*n_yag1 + np.linspace(0,n_yag2-n_yag1-1,n_yag2-n_yag1)))
                else:
                    indices2 = 2*np.linspace(0,n_yag2-1,n_yag2)
                    indices1 = np.hstack((2*np.linspace(0,n_yag2-1,n_yag2),2*n_yag2 + np.linspace(0,n_yag1-n_yag2-1,n_yag1-n_yag2)))
 
    ind = indices1 < n_t
    cnt = np.sum(ind)
    indices1 = indices1[ind] if (n_yag1 > 0) & (cnt > 0) else -1
    
    ind = indices2 < n_t
    cnt = np.sum(ind)
    indices2 = indices2[ind] if (n_yag2 > 0) & (cnt > 0) else -1

    indices1, indices2 = np.array(indices1).astype(int), np.array(indices2).astype(int)

    return n_yag1, n_yag2, indices1, indices2 


def get_ts_tci_ratio(shot, tmin, tmax, plot=False):

    nl_ts1, nl_ts2, nl_tci1, nl_tci2, time1, time2 = compare_ts_tci(shot, tmin, tmax, nl_num=4, plot=plot)
    nl_ts1 = np.select([nl_ts1 == 0], [np.nan], nl_ts1)
    nl_ts2 = np.select([nl_ts2 == 0], [np.nan], nl_ts2)

    ratio_laser1 = nl_tci1/nl_ts1
    ratio_laser2 = nl_tci2/nl_ts2

    in_window1 = (time1 > tmin) & (time1 < tmax)
    in_window2 = (time2 > tmin) & (time2 < tmax)

    if np.sum(in_window1) > 0:
        mult_factor1 = np.nanmean(ratio_laser1[in_window1])
    else:
        mult_factor1 = 0
        print('No ratio computed for laser1')
    if np.sum(in_window2) > 0:
        mult_factor2 = np.nanmean(ratio_laser2[in_window2])
    else:
        mult_factor2 = 0
        print('No ratio computed for laser2')

    if (mult_factor1 > 0) & (mult_factor2 > 0):
        mult_factor = (mult_factor1 + mult_factor2)/2
    else:
        if mult_factor1 > 0:
            mult_factor = mult_factor1
        elif mult_factor2 > 0:
            mult_factor = mult_factor2
        else:
            mult_factor = np.nan

    return mult_factor


def get_Bpol_RZ(shot, tmin, tmax, R, Z, gfile_name=None):

    # adapted from Grant Rutherford's plotMagneticField2D.py script in /home/millerma/poloidalFieldGetter
    
    if gfile_name is None:

        time = (tmax + tmin)/2
        _, gfile_name = get_geqdsk_cmod(shot, time*1e3, gfiles_loc='/home/millerma/lya/gfiles/', return_fname=True)
    

    gfileDict = gfd(eqdskName=gfile_name)
    
    rgrid = gfileDict['rgrid']
    zgrid = gfileDict['zgrid']
    Br_grid = gfileDict['brrz']
    Bz_grid = gfileDict['bzrz']
           
    from scipy.interpolate import RectBivariateSpline 
    Bpol_strength = np.sqrt(np.square(Bz_grid) + np.square(Br_grid)) 
    Bpol = RectBivariateSpline(zgrid,rgrid,Bpol_strength)(Z,R)[0][0]

    return Bpol



