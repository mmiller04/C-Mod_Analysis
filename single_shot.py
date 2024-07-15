'''
Obtain C-Mod neutral density profiles for a single shot and time interval. 
sciortino, Aug 2020

Modifications made by Andres Miller
'''
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import os, copy
#import fit_2D
import pickle as pkl
from scipy.interpolate import interp1d, RectBivariateSpline, UnivariateSpline, splev, splrep
from scipy.optimize import curve_fit
from cmod_tools import get_cmod_kin_profs
import aurora
from IPython import embed
from scipy import stats

# from this repo
import data_access as da


# PFS this is a function to facilitate output for database work
def assemble_into_dict(shot, tmin, tmax,
                f_ne, f_Te, f_pe,
                p_ne, p_Te, p_pe,
                SOL_exp_decay=True, decay_from_LCFS=False): #W/cm^3

    ''' Process Lyman-alpha data for a single shot/time interval. 
        Performs calculation by fitting kinetic profiles and mapping them onto Ly-a data.



    Parameters
    ----------
    shot : int
        CMOD shot number
    tmin : float
        Lower bound of time window
    tmax : float
        Upper bound of time window
    roa_kp : 1D array
        r/a grid
    ne : 1D array
        Electron density in units of :math:`10^{20} m^{-3}`
    ne_std : 1D array
        Uncertainties on electron density in units of :math:`10^{20} m^{-3}`
    Te : 1D array
        Electron temperature in units of :math:`keV`
    Te_std : 1D array
        Uncertainties on electron temperature in units of :math:`keV`
    p_ne : profiletools object
        Electron density object containing experimental data from all loaded diagnostics.
    p_Te : profiletools object
        Electron temperature object containing experimental data from all loaded diagnostics.
    SOL_exp_decay : bool
        If True, apply an exponential decay (fixed to 1cm length scale) in the region outside of the last radius covered by experimental data.
    decay_from_LCFS : bool
        If True, force exponential decay everywhere outside of the LCFS, ignoring any potential
        data in that region.

    Returns
    -------
    res: dict 
        Contains following keys: 

        R : 1D array
            Major radius grid
        roa : 1D array
            r/a grid
        rhop : 1D array
            rhop grid
        ne : 1D array
            Interpolated electron density in units of :math:`cm^{-3}`
        ne_unc : 1D array
            Interpolated electron density uncertaities in units of :math:`cm^{-3}`
        Te : 1D array
            Interpolated electron temperature in units of :math:`eV`
        Te_unc : 1D array
            Interpolated electron temperature uncertainties in units of :math:`eV`

    '''
    time = (tmax+tmin)/2.

    # this is just me being lazy and not wanting to go through every variable and substitute the object
    # will try to fix later
    ne, ne_std, grad_ne, grad_ne_std = f_ne.y, f_ne.fit_std, f_ne.grad_y, f_ne.grad_fit_std 
    Te, Te_std, grad_Te, grad_Te_std = f_Te.y, f_Te.fit_std, f_Te.grad_y, f_Te.grad_fit_std 
    pe, pe_std, grad_pe, grad_pe_std = f_pe.y, f_pe.fit_std, f_pe.grad_y, f_pe.grad_fit_std
   
    # transform coordinates:
    try: # EFIT20 only exists for shots from certain years
        e = da.CModEFITTree(int(shot), tree='EFIT20', length_unit='m')
    except:
        e = da.CModEFITTree(int(shot), tree='analysis', length_unit='m')
    time = (tmin + tmax)/2

    rhop_kp = f_ne.x

    R_kp = e.rho2rho('sqrtpsinorm', 'Rmid', rhop_kp, time)
    Rsep = e.rho2rho('sqrtpsinorm', 'Rmid', 1, time)
    
    # exponential decays of kinetic profs from last point of experimental data:
    # ne and Te profiles can miss last point depending on filtering?
    max_rhop_expt = np.maximum(np.max(p_ne.X[:,0]), np.max(p_Te.X[:,0]))  
    print('Experimental TS data extending to rhop={:.4}'.format(max_rhop_expt))

    # no uncertainties larger than 30 eV outside of LCFS
    Te_std[np.logical_and(R_kp>Rsep, Te_std>30e-3)] = 30e-3

    # set appropriate minima
    Te_min=3.0/1e3    # intended to be a better approximation overall than 3eV
    Te[Te<Te_min] = Te_min  
    ne[ne<1e12] = 1e12/1e14
    pe[pe<1.602] = 1.602/1e3

    # output results in a dictionary, to allow us to add/subtract keys in the future
    res = {'fit':{}, 'raw':{}}
    res['shot'] = shot
    res['eq'] = e 
    
    out_fit = res['fit']
    out_raw = res['raw']

    # interpolate kinetic profiles on emissivity radial grid
    ped_start, ped_end = 0.7, 1.05
    min_R = e.rho2rho('sqrtpsinorm', 'Rmid', ped_start, time)
    max_R = e.rho2rho('sqrtpsinorm', 'Rmid', ped_end, time)

    # different radius coordinates
    out_fit['R'] = np.linspace(min_R, max_R, 1000)
    out_fit['rhop'] = e.rho2rho('Rmid', 'sqrtpsinorm', out_fit['R'], time)

    # save profiles
    out_fit['ne'] = interp1d(rhop_kp,ne, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['Te'] = interp1d(rhop_kp,Te, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['pe'] = interp1d(rhop_kp,pe, bounds_error=False, fill_value=None)(out_fit['rhop'])
 
    out_fit['grad_ne'] = interp1d(rhop_kp,grad_ne, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['grad_Te'] = interp1d(rhop_kp,grad_Te, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['grad_pe'] = interp1d(rhop_kp,grad_pe, bounds_error=False, fill_value=None)(out_fit['rhop'])
  
    # and uncertainties
    out_fit['ne_unc'] = interp1d(rhop_kp,ne_std, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['Te_unc'] = interp1d(rhop_kp,Te_std, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['pe_unc'] = interp1d(rhop_kp,pe_std, bounds_error=False, fill_value=None)(out_fit['rhop'])
    
    out_fit['grad_ne_unc'] = interp1d(rhop_kp,grad_ne_std, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['grad_Te_unc'] = interp1d(rhop_kp,grad_Te_std, bounds_error=False, fill_value=None)(out_fit['rhop'])
    out_fit['grad_pe_unc'] = interp1d(rhop_kp,grad_pe_std, bounds_error=False, fill_value=None)(out_fit['rhop'])


    ## calculate also from raw data points
    ne = p_ne.y
    ne_unc = p_ne.err_y
    ne_rhop = p_ne.X[:,0]

    Te = p_Te.y
    Te_unc = p_Te.err_y
    Te_rhop = p_Te.X[:,0]
    
    pe = p_pe.y
    pe_unc = p_pe.err_y
    pe_rhop = p_pe.X[:,0]
 
    # map onto whichever has fewer points - want to sort points, interpolate, and then unsort them
    map_Te_on_ne = True if len(ne_rhop) < len(Te_rhop) else False

    # 
    out_raw['ne_rhop'] = ne_rhop
    out_raw['ne'] = ne
    out_raw['ne_unc'] = ne_unc
        
    out_raw['Te_rhop'] = Te_rhop
    out_raw['Te'] = Te
    out_raw['Te_unc'] = Te_unc
    
    out_raw['pe_rhop'] = pe_rhop
    out_raw['pe'] = pe
    out_raw['pe_unc'] = pe_unc

    out_raw['rhop'] = out_raw['ne_rhop'] if map_Te_on_ne else out_raw['Te_rhop']

    # make sure ne/Te is not negative
    ne_min = 1e12/1e14
    Te_min = 10/1e3
    pe_min = 1.602/1e3

    out_raw['ne'] = np.maximum(out_raw['ne'], ne_min)
    out_raw['Te'] = np.maximum(out_raw['Te'], Te_min)
    out_raw['pe'] = np.maximum(out_raw['pe'], Te_min)

    out_raw['R'] = e.rho2rho('sqrtpsinorm', 'Rmid', out_raw['rhop'], time)
    out_raw['ne_R'] = e.rho2rho('sqrtpsinorm', 'Rmid', out_raw['ne_rhop'], time)
    out_raw['Te_R'] = e.rho2rho('sqrtpsinorm', 'Rmid', out_raw['Te_rhop'], time)
    out_raw['pe_R'] = e.rho2rho('sqrtpsinorm', 'Rmid', out_raw['pe_rhop'], time)

    return res


def gaussian_shading(ax, x, y, y_unc, c='k', min_val=0.0):
    ''' Plot profile with uncertainties displayed as a shading whose color intensity represents a 
    gaussian PDF.
    '''
    norm_val = stats.norm.pdf(0)
    
    num=50  # discrete number of shades    
    for ij in np.arange(num):

        # below mean
        ax.fill_between(x,
                        np.maximum(y - 5*y_unc*(ij-1)/num, min_val),
                        np.maximum(y - 5*y_unc*ij/num, min_val),
                        alpha=stats.norm.pdf(5*ij/num)/norm_val,
                        linewidth=0.0,
                        color=c)

    # start looping from 2 to avoid overshading the same region
    for ij in np.arange(2,num):
        # above mean
        ax.fill_between(x, 
                        y + 5*y_unc*(ij-1.)/num,
                        y + 5*y_unc*ij/num,
                        alpha=stats.norm.pdf(5*ij/num)/norm_val,
                        linewidth=0.0,
                        color=c)

    ax.plot(x,y,linewidth=1.0,color=c)


def gaussian_3quantiles(ax, a, y, y_unc, c='k'):
    '''Alternative style of plotting where we show the 1-99, 10-90 and 25-75 quantiles of the
    probability distribution, as in Francesco's impurity transport inferences. 

    This may be over-complicated and somewhat redundant if the uncertainties are actually gaussian
    (as assumed in this function). It makes more sense when distributions are arbitrary...
    but this function could be useful to keep consistency of plotting styles.
    '''
    alp=0.6
    ax[0].fill_between(x,
                       y+stats.norm.ppf(0.25)*ff*y_unc/y,
                       y+stats.norm.ppf(0.75)*ff*y_unc/y,
                       alpha=alp, color=c)
    ax[0].fill_between(x,
                       y+stats.norm.ppf(0.10)*ff*y_unc/y,
                       y+stats.norm.ppf(0.90)*ff*y_unc/y,
                       alpha=alp/2, color=c)    
    ax[0].fill_between(x,
                       y+stats.norm.ppf(0.01)*ff*y_unc/y,
                       y+stats.norm.ppf(0.99)*ff*y_unc/y,
                       alpha=alp/3, color=c)


def plot_TS_overview(res, overplot_raw=False, Te_min=10., num_SP=None):
    '''Plot overview of Lyman-alpha results, showing emissivity, electron density and temperature, 
    neutral density and ionization profile

    Parameters:
    res : dict
        Dictionary containing the processed result of :py:fun:`get_lya_nn_prof`.
    overplot_raw : bool
        If True, overplot data points mapped from TS and probe data all the way to the ionization
        rate. This isn't "raw data", but rather a mapping from the raw data locations.
    '''

    fit = res['fit']
    if overplot_raw: raw = res['raw']
    geqdsk = res['geqdsk']
    Rsep = aurora.rad_coord_transform(1.0, 'rhop', 'Rmid', geqdsk)
    rminor = Rsep - geqdsk['RMAXIS'] # minor radius at the midplane

    ff = 1./np.log(10.)
   
    # complete analysis layout:
    fit['ne'][fit['ne']<1e10] = 1e10
    fit['nn'][fit['nn']<1.] = 1.0

    
    # -------------------------------

    fig,ax = plt.subplots(3,1, figsize=(8,12), sharex=True)

    set_xlim=True
    
    # Plot ne
    gaussian_shading(ax[0], fit['rhop'], fit['ne'], fit['ne_unc'], c='b', min_val = 1e11)
    if overplot_raw:
        ne_ts_mask = raw['ne_rhop'] > np.min(fit['rhop'])
        ne_sp_mask = raw['ne_rhop'] > np.min(fit['rhop']) 
        if num_SP['ne'] == 0: 
            ne_ts_mask[:] = True
            ne_sp_mask[:] = False
        else:
            ne_ts_mask[-num_SP['ne']:] = False
            ne_sp_mask[:-num_SP['ne']] = False
        ax[0].errorbar(raw['ne_rhop'][ne_sp_mask], raw['ne'][ne_sp_mask], raw['ne_unc'][ne_sp_mask], color='blue', fmt='x')
        ax[0].errorbar(raw['ne_rhop'][ne_ts_mask], raw['ne'][ne_ts_mask], raw['ne_unc'][ne_ts_mask], color='blue', fmt='.')

    # find appropriate max of y scale
    _ne_range = raw['ne'][raw['ne_rhop']>np.min(fit['rhop'])]
    ax[0].set_ylim([0, np.max(_ne_range)+3e13])  # 3e13 cm^-3 above max
    
    # plot Te
    gaussian_shading(ax[1], fit['rhop'], fit['Te'], fit['Te_unc'], c='r', min_val = Te_min)
    if overplot_raw:
        Te_ts_mask = raw['Te_rhop'] > np.min(fit['rhop'])
        Te_sp_mask = raw['Te_rhop'] > np.min(fit['rhop']) 
        if num_SP['Te'] == 0: 
            Te_ts_mask[:] = True
            Te_sp_mask[:] = False
        else:
            Te_sp_mask[:-num_SP['Te']] = False
            Te_sp_mask[:-num_SP['Te']] = False
        ax[1].errorbar(raw['Te_rhop'][Te_sp_mask], raw['Te'][Te_sp_mask], raw['Te_unc'][Te_sp_mask], color='red', fmt='x')
        ax[1].errorbar(raw['Te_rhop'][Te_ts_mask], raw['Te'][Te_ts_mask], raw['Te_unc'][Te_ts_mask], color='red', fmt='.')

    # find appropriate max of y scale
    _Te_range = raw['Te'][raw['Te_rhop']>np.min(fit['rhop'])]
    ax[1].set_ylim([0, np.max(_Te_range)+100])  # 100 eV above max

    # plot pe
    gaussian_shading(ax[1], fit['rhop'], fit['pe'], fit['pe_unc'], c='r', min_val = Te_min)
    if overplot_raw:
        pe_ts_mask = raw['pe_rhop'] > np.min(fit['rhop'])
        pe_sp_mask = raw['pe_rhop'] > np.min(fit['rhop']) 
        if num_SP['pe'] == 0: 
            pe_ts_mask[:] = True
            pe_sp_mask[:] = False
        else:
            pe_sp_mask[:-num_SP['pe']] = False
            pe_sp_mask[:-num_SP['pe']] = False
        ax[2].errorbar(raw['pe_rhop'][Te_sp_mask], raw['pe'][Te_sp_mask], raw['pe_unc'][Te_sp_mask], color='orange', fmt='x')
        ax[2].errorbar(raw['pe_rhop'][Te_ts_mask], raw['pe'][Te_ts_mask], raw['pe_unc'][Te_ts_mask], color='orange', fmt='.')

    # find appropriate max of y scale
    _pe_range = raw['pe'][raw['pe_rhop']>np.min(fit['rhop'])]
    ax[2].set_ylim([0, np.max(_pe_range)+100])  # WILL PROBABLY NEED TO UPDATE THIS

    # set axis labels
    ax[0].set_ylabel(r'$n_e$ [$cm^{-3}$]', color='b')
    ax[1].set_ylabel(r'$T_e$ [$eV$]', color='r')
    ax[2].set_ylabel(r'$p_e$ [$Pa$])', color='orange')

    # limit radial range to where we have some kinetic profile data and some lya data
    te_mask = np.logical_and(raw['Te_rhop']>np.min(fit['rhop']), raw['Te_rhop']<np.max(fit['rhop']))
    ne_mask = np.logical_and(raw['ne_rhop']>np.min(fit['rhop']), raw['ne_rhop']<np.max(fit['rhop']))

    if set_xlim:
        xmin_val = np.min([np.min(raw['ne_rhop'][ne_mask]),np.min(raw['Te_rhop'][te_mask]), np.min(fit['rhop'])]) #-0.01
        xmax_val = np.max([np.max(raw['ne_rhop'][ne_mask]),np.max(raw['Te_rhop'][te_mask])])  # lya data always goes further out
        ax[0].set_xlim([xmin_val, xmax_val])

    ax[-1].set_xlabel(r'$\rho_p$')
   
    fig.suptitle(f'C-Mod shot {res["shot"]}')
    fig.tight_layout()


def plot_check_fits(res,  Te_min=10.):
    
    try: # EFIT20 only exists for shots from certain years
        e = da.CModEFITTree(int(res['shot']), tree='EFIT20', length_unit='m')
    except:
        e = da.CModEFITTree(int(res['shot']), tree='analysis', length_unit='m')
    time = (tmin + tmax)/2

    Rsep = e.rho2rho('sqrtpsinorm', 'Rmid', 1, time)
    Rsep = res['eq'].rho2rho('sqrtpsinorm', 'Rmid', 1, time)
    
    fig, ax = plt.subplots(1,2, sharex='col')
        
    raw = res['raw']
    fit = res['fit']    

    ne_mask = raw['ne_rhop'] > np.min(fit['rhop'])
    ax[0].errorbar(res['raw']['ne_rhop'][ne_mask], res['raw']['ne'][ne_mask], res['raw']['ne_unc'][ne_mask], fmt='.', c='b', mec='b', mfc='w',zorder=1)
    ax[0].plot(res['fit']['rhop'], res['fit']['ne'], lw=2, c='k',zorder=2)
    ax[0].set_ylabel(r'$n_e$')

    Te_mask = raw['Te_rhop'] > np.min(fit['rhop'])
    ax[1].errorbar(res['raw']['Te_rhop'][Te_mask], res['raw']['Te'][Te_mask], res['raw']['Te_unc'][Te_mask], fmt='.', c='r', mec='r', mfc='w',zorder=1)
    ax[1].plot(res['fit']['rhop'], res['fit']['Te'], lw=2, c='k',zorder=2)
    ax[1].set_ylabel(r'$T_e$')

    kinmin_val = np.min(res['fit']['rhop'])
    kinmax_val = np.max(res['fit']['rhop'])

    nemin_val = np.minimum(np.nanmin(res['raw']['ne']), np.nanmin(res['fit']['ne'])) - 3e13
    nemax_val = np.nanmax(res['fit']['ne']) + 1
    Temin_val = np.minimum(np.nanmin(res['raw']['Te']), np.nanmin(res['fit']['Te'])) - 50   
    Temax_val = np.nanmax(res['fit']['Te']) + 200/1e3

    ax[0].set_xlim([kinmin_val - 0.1, kinmax_val])
    ax[0].set_ylim([nemin_val, nemax_val])
    ax[0].set_xlabel(r'$\rho_p$')
    
    ax[1].set_xlim([kinmin_val - 0.1, kinmax_val])
    ax[1].set_ylim([Temin_val, Temax_val])
    ax[1].set_xlabel(r'$\rho_p$')


def plot_kp_gaussian(kp_dict, toplot=[], colors=[]): 
 
    rows = len(toplot)
    if colors == []: colors = ['b']*rows

    fig, ax = plt.subplots(rows,sharex=True)
    for tp in range(len(toplot)):
        gaussian_shading(ax[tp], kp_dict['rhop'], kp_dict[toplot[tp]]['var'], kp_dict[toplot[tp]]['std'], c=colors[tp], min_val=0.0)
            
        if toplot[tp][0] != 'g':
            ylabel = '${}$'.format(toplot[tp][0]) + '$_{e}$'
        else:
            ylabel = '$\\nabla {}$'.format(toplot[tp][-2]) + '$_{e}$'
        ax[tp].set_ylabel(ylabel)
    ax[-1].set_xlabel('$\\rho_{p}$')        
    
    return None


if __name__=='__main__':

    ##### SHOT INFO #####

    # record pressure shot
    shot = 1160930033
    tmin = 1.1
    tmax = 1.2

    # high H98 L-mode
    shot = 1070803016
    tmin = 0.7
    tmax = 1.2

    # high H98 L-mode
    #shot = 1030530016
    #tmin = 0.83
    #tmax = 0.9

    # mid H98, high ne L-mode
    shot = 1070816013
    tmin = 1.0
    tmax = 1.1

    # test L-mode
    shot = 1070829009
    tmin = 1.0
    tmax = 1.05
        

    ############
    ne_min = 1e12 # cm^{-3}
    Te_min = 10. #eV

    force_to_zero = True # helps constrain fits if there's not good SOL coverage
    num_mc = 5 # to estimate fitting error - can probably speed this up jamie's method of repassing
               # in fit parameters into new iteration and vectorizing
 
    import time
    start_time = time.time()

    # this is the call to grab TS data and fit it
    kp_out = get_cmod_kin_profs(shot, tmin, tmax,
                                           apply_final_sep_stretch=True, force_to_zero=force_to_zero,
                                           frac_err=False, num_mc=num_mc, core_ts_mult=False, edge_ts_mult=False) 
    f_ne, f_Te, f_pe, p_ne, p_Te, p_pe = kp_out

    kp_time = time.time()
    print('Time for fits: ', kp_time - start_time)


    ##### ASSEMBLE RESULTS INTO DICT FOR OUTPUT #####

    res = assemble_into_dict(shot, tmin, tmax, 
                                f_ne, f_Te, f_pe, 
                                p_ne, p_Te, p_pe)


    ##### PLOT RESULTS #####
    
    plot_check_fits(res, Te_min=Te_min)

    # make some plots to check uncertainties - use dictionaries to plot
    kp_dict = {'ne':{}, 'grad_ne':{}, 'Te':{}, 'grad_Te':{}, 'pe':{}, 'grad_pe':{}}
    kp_dict['ne']['var'], kp_dict['ne']['std'], kp_dict['grad_ne']['var'], kp_dict['grad_ne']['std'] = f_ne.y, f_ne.fit_std, -f_ne.grad_y, f_ne.grad_fit_std
    kp_dict['Te']['var'], kp_dict['Te']['std'], kp_dict['grad_Te']['var'], kp_dict['grad_Te']['std'] = f_Te.y, f_Te.fit_std, -f_Te.grad_y, f_Te.grad_fit_std
    kp_dict['pe']['var'], kp_dict['pe']['std'], kp_dict['grad_pe']['var'], kp_dict['grad_pe']['std'] = f_pe.y, f_pe.fit_std, -f_pe.grad_y, f_pe.grad_fit_std
    kp_dict['rhop'] = f_ne.x # can grab .x for any parameter (f_ne, f_Te, f_pe) - should all be the same


    gaussian_plots = ['grad_ne','grad_Te'] #'Te', 'Te_std', 'grad_Te', 'grad_Te_std'
    plot_kp_gaussian(kp_dict, toplot=gaussian_plots)


    ##### SAVE SINGLE SHOT RESULTS #####

    output = False # should be put in as an argument into run

    if output:

        # for outputting
        tmin_ms = int(tmin*1e3)
        tmax_ms = int(tmax*1e3)

        # store profiles in a pickle file (warning: changing this data structure will break things)
        fit = res['fit']
        out2 = [fit['rhop'],fit['r/a'],fit['R'],
                fit['ne'],fit['ne_unc'],fit['Te'],fit['Te_unc'],
                fit['grad_ne'], fit['grad_ne_unc'], fit['grad_Te'], fit['grad_Te_unc']]
            
        with open(f'Dicts/ts_{shot}_{tmin_ms}_{tmax_ms}.pkl','wb') as f:
            pkl.dump(out2,f)

        # store raw data as well
        raw = res['raw']
        out3 = [raw['rhop'],raw['r/a'],raw['R'],raw['nn'],
                raw['ne_rhop'], raw['ne'],raw['ne_unc'],
                raw['Te_rhop'], raw['Te'],raw['Te_unc']]

        with open(f'Dicts/ts_raw_{shot}_{tmin_ms}_{tmax_ms}.pkl','wb') as f:
            pkl.dump(out3,f)


