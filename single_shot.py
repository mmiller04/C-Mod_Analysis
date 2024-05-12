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
import cmod_tools
import aurora
from IPython import embed
from scipy import stats


# PFS this is a function to facilitate output for database work
def assemble_fit_into_dict(shot,tmin,tmax,f_ne,f_Te,f_pe,
                p_ne, p_Te, p_pe, geqdsk=None,
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
    geqdsk : dict
        Dictionary containing processed geqdsk file
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
   
    ne_decay_len_SOL = 0.01 # m
    Te_decay_len_SOL = 0.01 # m
    pe_decay_len_SOL = 0.01 # m

    # transform coordinates:
    if geqdsk is None:
        geqdsk = cmod_tools.get_geqdsk_cmod(
            shot, time*1e3, gfiles_loc = '/home/millerma/lya/gfiles/')

    rhop_kp = f_ne.x

    from scipy.interpolate import UnivariateSpline
    mp_ind = np.where(geqdsk['AuxQuantities']['Z'] == 0)[0][0]
    R_mp = geqdsk['AuxQuantities']['R']
    rhop_mp = geqdsk['AuxQuantities']['RHOpRZ'][mp_ind]
    R0 = geqdsk['RMAXIS']
    omp = R_mp > R0
    R_kp = UnivariateSpline(rhop_mp[omp], R_mp[omp])(rhop_kp) 
    #R_kp = aurora.rad_coord_transform(rhop_kp, 'rhop', 'Rmid', geqdsk)
    
    Rsep = aurora.rad_coord_transform(1.0, 'rhop', 'Rmid', geqdsk)
    rminor = Rsep - geqdsk['RMAXIS'] # minor radius at the midplane
    
    # exponential decays of kinetic profs from last point of experimental data:
    # ne and Te profiles can miss last point depending on filtering?
    max_rhop_expt = np.maximum(np.max(p_ne.X[:,0]), np.max(p_Te.X[:,0]))  
    #max_rhop_expt = aurora.get_rhop_RZ(geqdsk['RMAXIS']+max_roa_expt*rminor, 0., geqdsk)
    print('Experimental TS data extending to rhop={:.4}'.format(max_rhop_expt))

    indLCFS = np.argmin(np.abs(R_kp - Rsep))
    ind_max = np.argmin(np.abs(rhop_kp - max_rhop_expt))
    if SOL_exp_decay and decay_from_LCFS: # exponential decay from the LCFS
        ind_max = indLCFS
        
    ne_std_av = copy.deepcopy(ne_std)
    Te_std_av = copy.deepcopy(Te_std)
    pe_std_av = copy.deepcopy(pe_std)
    
    if SOL_exp_decay:
        # apply exponential decay outside of region covered by data
        ne_sol = ne[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/ne_decay_len_SOL)
        grad_ne_sol = ne[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/ne_decay_len_SOL)/ne_decay_len_SOL
        ne_av = np.concatenate((ne[:ind_max], ne_sol))
        grad_ne_av = np.concatenate((grad_ne[:ind_max], grad_ne_sol))

        Te_sol = Te[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/Te_decay_len_SOL)
        grad_Te_sol = Te[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/Te_decay_len_SOL)/Te_decay_len_SOL
        Te_av = np.concatenate((Te[:ind_max], Te_sol))
        grad_Te_av = np.concatenate((grad_Te[:ind_max], grad_Te_sol))

        pe_sol = pe[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/pe_decay_len_SOL)
        grad_pe_sol = pe[ind_max-1]*np.exp(-(R_kp[ind_max:] - R_kp[ind_max-1])/pe_decay_len_SOL)/pe_decay_len_SOL
        pe_av = np.concatenate((pe[:ind_max], pe_sol))
        grad_pe_av = np.concatenate((grad_pe[:ind_max], grad_pe_sol))

        # set all ne/Te std outside of experimental range to mean of outer-most values
        ne_edge_unc = np.mean(ne_std[ind_max-3:ind_max])
        Te_edge_unc = np.mean(Te_std[ind_max-3:ind_max])
        pe_edge_unc = np.mean(pe_std[ind_max-3:ind_max])
        ne_std_av[ind_max:] = ne_edge_unc if ne_edge_unc<5e13 else 5e13
        Te_std_av[ind_max:] = Te_edge_unc if Te_edge_unc<30e-3 else 30e-3
        pe_std_av[ind_max:] = pe_edge_unc if pe_edge_unc<1.5e12 else 1.5e12
        
        grad_ne_edge_unc = np.mean(grad_ne_std[ind_max-3:ind_max])
        grad_Te_edge_unc = np.mean(grad_Te_std[ind_max-3:ind_max])
        grad_pe_edge_unc = np.mean(grad_pe_std[ind_max-3:ind_max])
        grad_ne_std_av[ind_max:] = grad_ne_edge_unc if grad_ne_edge_unc<5e13 else 5e13
        grad_Te_std_av[ind_max:] = grad_Te_edge_unc if grad_Te_edge_unc<30e-3 else 30e-3
        grad_pe_std_av[ind_max:] = grad_pe_edge_unc if grad_pe_edge_unc<1.5e12 else 1.5e12
    else:
        ne_av = copy.deepcopy(ne)
        Te_av = copy.deepcopy(Te)
        pe_av = copy.deepcopy(pe)
        ne_std_av = copy.deepcopy(ne_std)
        Te_std_av = copy.deepcopy(Te_std)
        pe_std_av = copy.deepcopy(pe_std)
        
        grad_ne_av = copy.deepcopy(grad_ne)
        grad_Te_av = copy.deepcopy(grad_Te)
        grad_pe_av = copy.deepcopy(grad_pe)
        grad_ne_std_av = copy.deepcopy(grad_ne_std)
        grad_Te_std_av = copy.deepcopy(grad_Te_std)
        grad_pe_std_av = copy.deepcopy(grad_pe_std)

    # no uncertainties larger than 30 eV outside of LCFS
    Te_std_av[np.logical_and(R_kp>Rsep, Te_std_av>30e-3)] = 30e-3
    #Te_std_av_lcfs = Te_std_av[indLCFS:]
    #Te_std_av_lcfs[Te_std_av_lcfs>30e-3] = 30e-3

    # set ne to cm^-3 and Te in eV and pe in Pa
    ne_av *= 1e14
    ne_std_av *= 1e14
    Te_av *= 1e3
    Te_std_av *= 1e3
    pe_av *= 1e13
    pe_std_av *= 1e13
 
    grad_ne_av *= 1e14
    grad_ne_std_av *= 1e14
    grad_Te_av *= 1e3
    grad_Te_std_av *= 1e3
    grad_pe_av *= 1e3
    grad_pe_std_av *= 1e3

    # set appropriate minima
    Te_min=3.0    # intended to be a better approximation overall than 3eV
    Te_av[Te_av<Te_min] = Te_min  
    ne_av[ne_av<1e12] = 1e12
    pe_av[pe_av<1.602] = 1.602

    # output results in a dictionary, to allow us to add/subtract keys in the future
    res = {'fit':{}}
    out = res['fit']
    res['geqdsk'] = geqdsk  # also save geqdsk and shot number in results dictionary
    res['shot'] = shot
    
    # interpolate kinetic profiles on emissivity radial grid
    ped_start, ped_end = 0.7, 1.05
    min_R = aurora.rad_coord_transform(ped_start, 'rhop', 'Rmid', geqdsk)
    max_R = aurora.rad_coord_transform(ped_end, 'rhop', 'Rmid', geqdsk)

    # three different radius coordinates
    out['R'] = np.linspace(min_R, max_R, 1000)
    out['r/a'] = (out['R'] - geqdsk['RMAXIS'])/rminor
    out['rhop'] = aurora.get_rhop_RZ(out['R'], np.zeros_like(out['R']), geqdsk)

    # get rvol coordinate as well
    _rvol, _rhop = get_rvol(geqdsk, dr0=0.03, dr1=0.03)
    out['rvol'] = interp1d(_rhop, _rvol, fill_value='extrapolate')(out['rhop'])

    # save profiles
    out['ne'] = np.exp(interp1d(rhop_kp,np.log(ne_av), bounds_error=False, fill_value=None)(out['rhop'])) # not sure why this is log? maybe it helps interpolation
    out['Te'] = interp1d(rhop_kp,Te_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['pe'] = interp1d(rhop_kp,pe_av, bounds_error=False, fill_value=None)(out['rhop'])
 
    out['grad_ne'] = interp1d(rhop_kp,grad_ne_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['grad_Te'] = interp1d(rhop_kp,grad_Te_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['grad_pe'] = interp1d(rhop_kp,grad_pe_av, bounds_error=False, fill_value=None)(out['rhop'])
  
    # and uncertainties
    out['ne_unc'] = interp1d(rhop_kp,ne_std_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['Te_unc'] = interp1d(rhop_kp,Te_std_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['pe_unc'] = interp1d(rhop_kp,pe_std_av, bounds_error=False, fill_value=None)(out['rhop'])
    
    out['grad_ne_unc'] = interp1d(rhop_kp,grad_ne_std_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['grad_Te_unc'] = interp1d(rhop_kp,grad_Te_std_av, bounds_error=False, fill_value=None)(out['rhop'])
    out['grad_pe_unc'] = interp1d(rhop_kp,grad_pe_std_av, bounds_error=False, fill_value=None)(out['rhop'])

    return res
    

# PFS probably don't need this one either
def assemble_raw_into_dict(shot,tmin,tmax, p_ne, p_Te, p_pe, geqdsk=None,
                        SOL_exp_decay=True, decay_from_LCFS=False):

    ''' Process Lyman-alpha data for a single shot/time interval. 
        Performs calculation by mapping Ly-a data onto "raw" TS/SP points.

    Parameters
    ----------
    shot : int
        CMOD shot number
    tmin : float
        Lower bound of time window
    tmax : float
        Upper bound of time window
    p_ne : profiletools object
        Electron density object containing experimental data from all loaded diagnostics.
    p_Te : profiletools object
        Electron temperature object containing experimental data from all loaded diagnostics.
    geqdsk : dict
        Dictionary containing processed geqdsk file
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
            "Raw" electron density in units of :math:`cm^{-3}`
        ne_unc : 1D array
            "Raw" electron density uncertaities in units of :math:`cm^{-3}`
        Te : 1D array
            "Raw" electron temperature in units of :math:`eV`
        Te_unc : 1D array
            "Raw" electron temperature uncertainties in units of :math:`eV`

    '''

    # transform coordinates:
    if geqdsk is None:
        geqdsk = cmod_tools.get_geqdsk_cmod(
            shot, time*1e3, gfiles_loc = '/home/millerma/lya/gfiles/')

    Rsep = aurora.rad_coord_transform(1.0, 'rhop', 'Rmid', geqdsk)
    rminor = Rsep - geqdsk['RMAXIS'] # minor radius at the midplane

    # output results in a dictionary, to allow us to add/subtract keys in the future
    res = {'raw':{}}
    out = res['raw']

    ## calculate also from raw data points
    # set ne to cm^-3 and Te in eV and pe in Pa

    ne = p_ne.y*1e14
    ne_unc = p_ne.err_y*1e14
    ne_rhop = p_ne.X[:,0]

    Te = p_Te.y*1e3
    Te_unc = p_Te.err_y*1e3
    Te_rhop = p_Te.X[:,0]
    
    pe = p_pe.y*1e3
    pe_unc = p_pe.err_y*1e3
    pe_rhop = p_pe.X[:,0]
 
    # map onto whichever has fewer points - want to sort points, interpolate, and then unsort them
    map_Te_on_ne = True if len(ne_rhop) < len(Te_rhop) else False

    # 
    out['ne_rhop'] = ne_rhop
    out['ne'] = ne
    out['ne_unc'] = ne_unc
        
    out['Te_rhop'] = Te_rhop
    out['Te'] = Te
    out['Te_unc'] = Te_unc
    
    out['pe_rhop'] = pe_rhop
    out['pe'] = pe
    out['pe_unc'] = pe_unc

    out['rhop'] = out['ne_rhop'] if map_Te_on_ne else out['Te_rhop']

    # make sure ne/Te is not negative
    ne_min = 1e12
    Te_min = 10
    pe_min = 1.602

    out['ne'] = np.maximum(out['ne'], ne_min)
    out['Te'] = np.maximum(out['Te'], Te_min)
    out['pe'] = np.maximum(out['pe'], Te_min)

    # map kps and emiss to midplane
    from scipy.interpolate import UnivariateSpline
    mp_ind = np.where(geqdsk['AuxQuantities']['Z'] == 0)[0][0]
    R_mp = geqdsk['AuxQuantities']['R']
    rhop_mp = geqdsk['AuxQuantities']['RHOpRZ'][mp_ind]
    R0 = geqdsk['RMAXIS']
    omp = R_mp > R0
    rhop_to_R = UnivariateSpline(rhop_mp[omp], R_mp[omp])

    out['R'] = rhop_to_R(out['rhop'])
    out['ne_R'] = rhop_to_R(out['ne_rhop'])
    out['Te_R'] = rhop_to_R(out['Te_rhop'])
    out['pe_R'] = rhop_to_R(out['pe_rhop'])
    
    out['r/a'] = (out['R'] - geqdsk['RMAXIS'])/rminor
    out['ne_r/a'] = (out['ne_R'] - geqdsk['RMAXIS'])/rminor
    out['Te_r/a'] = (out['Te_R'] - geqdsk['RMAXIS'])/rminor
    out['pe_r/a'] = (out['pe_R'] - geqdsk['RMAXIS'])/rminor
    
    _rvol, _rhop = get_rvol(geqdsk, dr0=0.03, dr1=0.03)
    out['rvol'] = interp1d(_rhop, _rvol, fill_value='extrapolate')(out['rhop'])
    out['ne_rvol'] = interp1d(_rhop, _rvol, fill_value='extrapolate')(out['ne_rhop'])
    out['Te_rvol'] = interp1d(_rhop, _rvol, fill_value='extrapolate')(out['Te_rhop'])
    out['pe_rvol'] = interp1d(_rhop, _rvol, fill_value='extrapolate')(out['pe_rhop'])

    return res


def fit_spline(ydata, xdata, yerr, factor=1, degree=3, plot=False):

    # dealing with NaNs
    w = np.isnan(ydata)
    
    xdata = xdata[~w]
    ydata = ydata[~w]
    yerr = yerr[~w]

    # need to sort data
    sorted_inds = np.argsort(xdata)
    xdata = xdata[sorted_inds]
    ydata = ydata[sorted_inds]
    yerr = yerr[sorted_inds]
    
    # need to get rid of duplicates
    xclean = []
    [xclean.append(x) for x in xdata if x not in xclean]
    
    yclean = []
    yclean_err = []
    for x in xclean:
        vals = np.where(xdata == x)[0]

        # clean y
        yvals = []
        [yvals.append(ydata[v]) for v in vals]
        yclean.append(np.mean(yvals))

        # clean yerr
        yvals_err = []
        [yvals_err.append(yerr[e]) for e in vals]
        yclean_err.append(np.mean(yvals_err))

    try:
    
        yerr_norm = yclean_err/np.max(yclean_err)

        #spl = UnivariateSpline(xclean, np.log(yclean), s=factor, k=degree) # fit log of function
        spl = UnivariateSpline(xclean, np.log(yclean), w=1-yerr_norm, s=factor, k=degree) # fit log of function
    
        xspl = np.linspace(xclean[0], xclean[-1], 100)
        yspl = np.exp(spl(xspl)) # need to undo log
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(xclean,yclean,'o')
            ax.plot(xspl,yspl)
            plt.show()
    
    except Exception as e:
        print('Spline fitting of nn/S_ion did not work because:')
        print(e)
        # return null values
        if len(xdata) > 0:
            xspl = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 100)
            yspl = np.full(len(xspl),np.nan)
        else:
            xspl, yspl = np.nan, np.nan

    return yspl, xspl


def fit_func(ydata, xdata, k):

    # dealing with NaNs
    w = np.isnan(ydata)
    xdata = xdata[~w]
    ydata = ydata[~w]

    # need to sort data
    sorted_inds = np.argsort(xdata)
    xdata = xdata[sorted_inds]
    ydata = ydata[sorted_inds]
    
    # need to get rid of duplicates
    xclean = []
    [xclean.append(x) for x in xdata if x not in xclean]
    
    yclean = []
    for x in xclean:
        vals = np.where(xdata == x)[0]
        yvals = []
        [yvals.append(ydata[v]) for v in vals]
        yclean.append(np.mean(yvals))
    
    xfit = np.linspace(xclean[0], xclean[-1], 1000)
    poly = np.polyfit(xclean, np.log(yclean), k)
    yfit = np.exp(np.polyval(poly,xfit))
    
    return yfit, xfit


def get_rvol(geqdsk, k=10, dr0=0.1, dr1=0.1, bound_sep=8):

    
    # function based off Aaron's createradialgrid function in finegrid.py

    nml = {}
    rhop, _rvol = aurora.grids_utils.get_rhopol_rvol_mapping(geqdsk)
    
    rvol_lcfs = interp1d(rhop, _rvol)(1.0)
    
    rvol_lcfs = nml['rvol_lcfs'] = np.round(rvol_lcfs, 3)

    nml['K'] = k
    nml['dr_0'] = dr0
    nml['dr_1'] = dr1
    nml['bound_sep'] = bound_sep
    nml['lim_sep'] = 5

    grid_params = aurora.grids_utils.create_radial_grid(nml)
    rvol_grid, pro_grid, qpr_grid, prox_aparm = grid_params

    rhop_grid = interp1d(_rvol, rhop, fill_value='extrapolate')(rvol_grid)
    rhop_grid[0] = 0.0
    
    return rvol_grid*1e-2, rhop_grid

    

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


def plot_check_fits(res, geqdsk, Te_min=10., num_SP=None):

    Rsep = aurora.rad_coord_transform(1.0, 'rhop', 'Rmid', geqdsk)

    fig, ax = plt.subplots(1,3, sharex='col')
        
    raw = res['raw']
    fit = res['fit']    

    ne_ts_mask = raw['ne_rhop'] > np.min(fit['rhop'])
    ne_sp_mask = raw['ne_rhop'] > np.min(fit['rhop']) 
    if num_SP['ne'] == 0: 
        ne_ts_mask[:] = True
        ne_sp_mask[:] = False
    else:
        ne_ts_mask[-num_SP['ne']:] = False
        ne_sp_mask[:-num_SP['ne']] = False
    ax[0].errorbar(res['raw']['ne_rhop'][ne_ts_mask], res['raw']['ne'][ne_ts_mask], res['raw']['ne_unc'][ne_ts_mask], fmt='o', lw=2, c='b')
    ax[0].errorbar(res['raw']['ne_rhop'][ne_sp_mask], res['raw']['ne'][ne_sp_mask], res['raw']['ne_unc'][ne_sp_mask], fmt='x', lw=2, c='b')
    ax[0].plot(res['fit']['rhop'], res['fit']['ne'], c='b')
    ax[0].set_ylabel(r'$n_e$')

    Te_ts_mask = raw['Te_rhop'] > np.min(fit['rhop'])
    Te_sp_mask = raw['Te_rhop'] > np.min(fit['rhop']) 
    if num_SP['Te'] == 0: 
        Te_ts_mask[:] = True
        Te_sp_mask[:] = False
    else:
        Te_sp_mask[:-num_SP['Te']] = False
        Te_sp_mask[:-num_SP['Te']] = False
    ax[1].errorbar(res['raw']['Te_rhop'][Te_ts_mask], res['raw']['Te'][Te_ts_mask], res['raw']['Te_unc'][Te_ts_mask], fmt='o', c='r', lw=2)
    ax[1].errorbar(res['raw']['Te_rhop'][Te_sp_mask], res['raw']['Te'][Te_sp_mask], res['raw']['Te_unc'][Te_sp_mask], fmt='x', c='r', lw=2)
    ax[1].plot(res['fit']['rhop'], res['fit']['Te'], c='r')
    ax[1].set_ylabel(r'$T_e$')

    kinmin_val = np.min(res['fit']['rhop'])
    kinmax_val = np.max(res['fit']['rhop'])

    nemin_val = np.minimum(np.nanmin(res['raw']['ne']), np.nanmin(res['fit']['ne'])) - 3e13
    nemax_val = np.nanmax(res['fit']['ne']) + 1e14    
    Temin_val = np.minimum(np.nanmin(res['raw']['Te']), np.nanmin(res['fit']['Te'])) - 50   
    Temax_val = np.nanmax(res['fit']['Te']) + 200

    ax[0].set_xlim([kinmin_val - 0.1, kinmax_val])
    ax[0].set_ylim([nemin_val, nemax_val])
    
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
    #shot = 1070816013
    #tmin = 1.0
    #tmax = 1.5

    # test L-mode
    shot = 1000830024
    tmin = 0.6
    tmax = 0.7
        

    ############
    ne_min = 1e12 # cm^{-3}
    Te_min = 10. #eV

    force_to_zero = True # helps constrain fits if there's not good SOL coverage
    num_mc = 5 # to estimate fitting error
 
    gfiles_loc = '/home/millerma/lya/gfiles/' # WILL WANT TO MODIFY THIS TO GRAB USERNAME OF WHOEVER IS RUNNING

    # PFS not sure if we actually need the geqdsk tbh - will have to check on this
    geqdsk = cmod_tools.get_geqdsk_cmod(shot,(tmin+tmax)/2.*1e3, gfiles_loc=gfiles_loc)

    import time
    start_time = time.time()

    # this is the call to grab TS data and fit it
    kp_out = cmod_tools.get_cmod_kin_profs(shot, tmin, tmax, geqdsk = geqdsk, probes=['A','F'],
                                           apply_final_sep_stretch=True, force_to_zero=force_to_zero,
                                           frac_err=False, num_mc=num_mc, core_ts_mult=False, edge_ts_mult=False) 
    f_ne, f_Te, f_pe, p_ne, p_Te, p_pe, num_SP = kp_out

    kp_time = time.time()
    print('Time for fits: ', kp_time - start_time)

    time = (tmin+tmax)/2

    # used to output all the data - probably can streamline this
    res = assemble_fit_into_dict(shot, tmin, tmax, f_ne, f_Te, f_pe, 
                                    p_ne, p_Te, p_pe, geqdsk=geqdsk, 
                                    SOL_exp_decay=False)

    res_raw = assemble_raw_into_dict(shot, tmin, tmax, 
                                        p_ne, p_Te, p_pe, geqdsk=geqdsk, 
                                        SOL_exp_decay=False)

    res.update(res_raw)



    ##### PLOT RESULTS #####
    
    plot_check_fits(res, geqdsk, Te_min=Te_min, num_SP=num_SP)

    # make some plots to check uncertainties - use dictionaries to plot
    kp_dict = {'ne':{}, 'grad_ne':{}, 'Te':{}, 'grad_Te':{}, 'pe':{}, 'grad_pe':{}}
    kp_dict['ne']['var'], kp_dict['ne']['std'], kp_dict['grad_ne']['var'], kp_dict['grad_ne']['std'] = f_ne.y, f_ne.fit_std, -f_ne.grad_y, f_ne.grad_fit_std
    kp_dict['Te']['var'], kp_dict['Te']['std'], kp_dict['grad_Te']['var'], kp_dict['grad_Te']['std'] = f_Te.y, f_Te.fit_std, -f_Te.grad_y, f_Te.grad_fit_std
    kp_dict['pe']['var'], kp_dict['pe']['std'], kp_dict['grad_pe']['var'], kp_dict['grad_pe']['std'] = f_pe.y, f_pe.fit_std, -f_pe.grad_y, f_pe.grad_fit_std
    kp_dict['rhop'] = f_ne.x # can grab .x for any parameter (f_ne, f_Te, f_pe) - should all be the same

    ''' not too sure why this is here but gonna leave it in case it helps visualize the fits?
    # compute std using full distribution
    
    from scipy.stats import norm    
    ne_stats = np.array([norm.fit(f_ne.fit_dist_prefilt[:,jj]) for jj in range(len(f_ne.x))])
    grad_ne_stats = np.array([norm.fit(f_ne.grad_fit_dist_prefilt[:,jj]) for jj in range(len(f_ne.x))])
    Te_stats = np.array([norm.fit(f_Te.fit_dist_prefilt[:,jj]) for jj in range(len(f_Te.x))])
    grad_Te_stats = np.array([norm.fit(f_Te.grad_fit_dist_prefilt[:,jj]) for jj in range(len(f_Te.x))])

    kp_dict['Te']['var'] = kp_dict['ne']['var']
    kp_dict['Te']['std'] = ne_stats[:,1]
    
    kp_dict['grad_Te']['var'] = kp_dict['grad_ne']['var']
    kp_dict['grad_Te']['std'] = grad_ne_stats[:,1]
    '''

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


