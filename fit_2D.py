# Routines written primarily by Francesco Sciortino

import MDSplus
import numpy as np
import copy, sys, pickle as pkl
if '/home/sciortino/quickfit3' not in sys.path:
    sys.path.append('/home/sciortino/quickfit3')
if '/home/sciortino/tools3' not in sys.path:
    sys.path.append('/home/sciortino/tools3')
if '/home/sciortino/tools3/neutrals' not in sys.path:
    sys.path.append('/home/sciortino/tools3/neutrals')
if '/home/sciortino/usr/python3modules/gptools3' not in sys.path:
    sys.path.append('/home/sciortino/usr/python3modules/gptools3')
if '/home/sciortino/usr/python3modules/profiletools3' not in sys.path:
    sys.path.append('/home/sciortino/usr/python3modules/profiletools3')
if '/home/sciortino/usr/python3modules/eqtools3' not in sys.path:
    sys.path.append('/home/sciortino/usr/python3modules/eqtools3')
if '/home/sciortino/usr/python3modules/TRIPPy3' not in sys.path:
    sys.path.append('/home/sciortino/usr/python3modules/TRIPPy3')

#import eqtools
from grid_map import map2grid # QUICKFIT
from peak_detect import peakdetect
from scipy.interpolate import RectBivariateSpline, interp1d
import save_profiletools_data
import quickfit_db
import matplotlib.pyplot as plt
from IPython import embed
import cmod_tools
import aurora
from scipy.constants import e as q_electron
import twopoint_model

import matplotlib as mpl
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16
mpl.rcParams['axes.labelsize']=16
#mpl.rc('text',usetex=True)    # uncomment for pretty (but slow) plots (doesn't seem to work on mfews08)

ls_cycle = aurora.get_ls_cycle()
c_cycle = aurora.get_color_cycle()


def fit_2D_profs(shot,tmin=None,tmax=None,ne_zero_edge = False, Te_zero_edge = True,
                 eps_x_Te=None, eps_t_Te=None, eps_x_ne=None, eps_t_ne=None,
                 stretch_lcfs=False, time_indpt_profs=False,
                 pedestal_rho_ne=0.95, pedestal_rho_Te=0.95,
                 nr_pts=121, roa_max=1.1, Te_LCFS_eV=75.0, get_2pt_model_Teu=True,
                 get_Ti=False):
    '''Function for quick time-dependent kinetic profile fitting. 

    INPUTS
    shot : C-Mod shot number, e.g. 1101014019
    tmin : lower bound of the time window [s]
    tmax : upper bound of the time window [s]
    ne_zero_edge : bool, if True force ne to be 0 at the edge bound
    Te_zero_edge : bool, if True force Te to be 0 at the edge bound
    eps_x_Te : float, [0,1]; parameter for radial Te profile smoothness
    eps_t_Te : float, [0,1]; parameter for temporal Te profile smoothnss
    eps_x_ne : float, [0,1]; parameter for radial ne profile smoothness
    eps_t_ne : float, [0,1]; parameter for temporal ne profile smoothnss
    stretch_lcfs : bool, if True stretch radial coordinate such that Te(rho=1)=Te_LCFS_eV or the value 
        given by the 2-point model if get_2pt_model_Teu=True. 
    time_indpt_profs : bool, if True time average all profiles
    pedestal_rho_(ne,Te) : value at which pedestal is expected for ne and Te (separately). 
        If set to large value (e.g. rho>2) this will have no practical effect. Default is rho=1.0.
    nr_pts : int, number of radial points in fitted profiles. Default is 121.
    roa_max : maximum value of r/a to fit. Default is 1.1.
    Te_LCFS_eV : electron temperature expected at the LCFS (only used if stretch_lcfs=True). Default is 75eV. 
    get_Ti : if True, attempt to fetch Ti data from Hirex-Sr and form a full profile with edge TS.

    OUTPUTS:
    out : array containing the following fields:
         r_vec, (t_vec,) ne, ne_std, Te, Te_std = out
         where t_vec is only present if a time average is not requested.
    p_ne : object containing experimental data and uncertainties for ne
    p_Te : object containing experimental data and uncertainties for Te

    NB: ne in 10^20 m^-3; Te in keV    

    MWE:
    
    import fit_2D, matplotlib.pyplot as plt
    shot = 1070511010
    tmin=0.8; tmax=1.2
    out,p_ne,p_Te = fit_2D.fit_2D_profs(shot,tmin,tmax, time_indpt_profs=True)
    r_vec, ne, ne_std, Te, Te_std = out
    p_ne.plot_data(); plt.plot(r_vec, ne); plt.tight_layout()
    p_Te.plot_data(); plt.plot(r_vec, Te); plt.tight_layout()
    '''

    try:
        # get info specific to shot:
        _out = quickfit_db.quickfit_database(shot)
        t_min, t_max, lookahead, delta, max_ne_err, max_Te_err, _pedestal_rho,\
            _eps_x_Te, _eps_t_Te, _eps_x_ne, _eps_t_ne, plot_times, reinforce_ped_f = _out
    except:
        # from WMC database
        t_min,t_max = 0.9, 1.1 #1.2, 1.3
        plot_times=[1.0,1.05]
        lookahead, delta, max_ne_err, max_Te_err, _pedestal_rho = 30, 0.1, 0.5, 0.5, None
        _eps_x_Te, _eps_t_Te, _eps_x_ne, _eps_t_ne = 0.45, 0.7, 0.5, 0.5
        reinforce_ped_f = 5
        
    if pedestal_rho_ne is None:
        # pedestal_rho was not passed as an argument, use defaults
        pedestal_rho_ne = copy.deepcopy(_pedestal_rho)
    if pedestal_rho_Te is None:
        pedestal_rho_Te = copy.deepcopy(_pedestal_rho)

    if eps_x_Te is None: eps_x_Te = copy.deepcopy(_eps_x_Te)
    if eps_t_Te is None: eps_t_Te = copy.deepcopy(_eps_t_Te)
    if eps_x_ne is None: eps_x_ne = copy.deepcopy(_eps_x_ne)
    if eps_t_ne is None: eps_t_ne = copy.deepcopy(_eps_t_ne)    
    
    if tmin is not None:
        # replace
        t_min = copy.deepcopy(tmin)
    if tmax is not None:
        # replace
        t_max = copy.deepcopy(tmax)
    
    dt_ne = 1e-3
    dt_Te = 1e-4   # small values make fit take longer
        
    # Run fit for some time before and after requested times (cut at the end)
    t_min-=0.02
    t_max+=0.02

    # get sawtooth times
    t_sawteeth = get_sawtooth_times(shot,t_min,t_max, lookahead, delta)

    # get ne,Te data
    p_ne, p_Te = save_profiletools_data.get_ne_Te(shot, [t_min,t_max], ntimes=None,  #ntimes=None--> no GPC downsampling
                           reinforce_ped_f=reinforce_ped_f,reinforce_x=0.9) 

    # clean up profs
    min_ne_err=0.005 #0.01;
    min_Te_err=0.005 #0.01 #0.001  # increasing this to 0.05 will miss pedestal/SOL points
    p_ne,p_Te = clean_profs(shot,p_ne, p_Te, min_ne_err, max_ne_err, min_Te_err, max_Te_err)

    # try:
    #     # attempt to fetch ASP and FSP data if available
    #     p_ne_p, p_Te_p = save_profiletools_data.fetch_edge_probes(shot, time, rhop_min=0.995, rhop_max=1.05)
    #     p_ne.add_profile(p_ne_p)
    #     p_Te.add_profile(p_Te_p)

    #     print(f'Successfully fetched ASP data for shot {shot}')

    #     # Since we have SOL data, prevent QUICKFIT from enforcing 0's at the wall
    #     ne_zero_edge = False
    #     Te_zero_edge = False        
    # except Exception:
    #     # probe data not available
    #     pass
    
    if get_Ti:
        p_Ti = get_Ti_data(shot, t_min, t_max, tht=1, p_Te=p_Te)

    p_ne.y[p_ne.y<=0.] = 0.01  # 10^18 m^-3
    p_Te.y[p_Te.y<=0.] = 0.01 # 10 eV


    # TEMPORARY
    #extra_X = np.ones((10,2))
    #extra_X[:,0] = (t_max+t_min)/2. * np.ones(10)
    #extra_X[:,1] = np.linspace(1.01, 1.03,10) #np.ones(10)* np.max(p_ne.X[:,1])+0.01

    #p_ne.add_data(extra_X, np.ones(10)*np.min(p_ne.y), err_y=np.ones(10)*np.ones(10)*np.min(p_ne.err_y))
    
    # now fit:
    res = run_quickfit(p_ne, p_Te, t_sawteeth, dt_ne, dt_Te,
                                  ne_zero_edge, Te_zero_edge, pedestal_rho_ne,pedestal_rho_Te,
                                  eps_x_ne, eps_t_ne, eps_x_Te, eps_t_Te, nr_pts, roa_max,
                                  p_Ti=p_Ti if get_Ti else None)
    if get_Ti:
        ne_arr, Te_arr, Ti_arr = res
    else:
        ne_arr, Te_arr = res
        
    #ne, ne_u, ne_d, ne_t, ne_r = ne_arr
    #Te, Te_u, Te_d, Te_t, Te_r = Te_arr
    #Ti, Ti_u, Ti_d, Ti_t, Ti_r = Ti_arr
   
    if stretch_lcfs and get_2pt_model_Teu:
        # get 2-point model prediction from
        ne_r = ne_arr[-1]
        r_vec = ne_r[0,:]
        ne = np.mean(ne_arr[0], axis=0)
        Te = np.mean(Te_arr[0], axis=0)
        Te_lcfs_eV = Teu_2pt_model(shot,t_min,t_max,ne,Te,r_vec)
    else:
        # use given value
        Te_lcfs_eV = copy.deepcopy(Te_LCFS_eV)

    if stretch_lcfs:
        if Te_lcfs_eV<75.0:
            print('Te_lcfs_eV<75.0 is hardly believable... Setting it to 75 eV now')
            Te_lcfs_eV=75.0
        print('Set Te [eV] at the LCFS: ', Te_lcfs_eV)
    
    if time_indpt_profs:
        # average profiles in time
        _out = get_time_ave_profs(ne_arr,Te_arr)

        # time average experimental data as well
        p_ne.time_average(weighted=True)
        p_Te.time_average(weighted=True)

        # not sure why time_average seems to give small err_y even though it was large enough before averaging...
        p_ne.err_y[p_ne.err_y<min_ne_err] = min_ne_err
        p_Te.err_y[p_Te.err_y<min_Te_err] = min_Te_err

        # get_time_ave_profs returns ne_u and ne_d (etc.) rather than std's
        r_vec,ne, ne_u,ne_d,Te,Te_u,Te_d = _out
        ne_std = (ne_u - ne_d)/2.
        Te_std = (Te_u - Te_d)/2.

        # set minimum of 3 eV and 10eV minimum uncertainty
        Te_std[Te<1e-2] = 1e-2
        Te[Te<3e-3] = 3e-3
        
        if stretch_lcfs:            
            # stretch time-indpt profiles to have Te_u eV at LCFS
            out = stretch_time_indpt_profs(*_out, Te_LCFS=Te_lcfs_eV)
        else:
            out = [r_vec, ne[0,:], ne_std[0,:], Te[0,:], Te_std[0]]  # only return 1 (time-indpt) slice
    else:
        # time dependent
        if stretch_lcfs:
            # ensure that ne and Te profiles are on the same r,t grids
            _ne, _ne_u, _ne_d, _ne_t_, _ne_r_ = ne_arr
            _Te, _Te_u, _Te_d, _Te_t_, _Te_r_ = Te_arr

            _ne_t = _ne_t_[:,0]
            _ne_r = _ne_r_[0,:]
            _Te_t = _Te_t_[:,0]
            _Te_r = _Te_r_[0,:]
            
            r_vec = np.linspace(np.min([_ne_r.min(),_Te_r.min()]),np.max([_ne_r.max(),_Te_r.max()]),
                                np.max([len(_ne_r),len(_Te_r)]))
            t_vec = np.linspace(np.min([_ne_t.min(),_Te_t.min()]),np.max([_ne_t.max(),_Te_t.max()]),
                                np.max([len(_ne_t),len(_Te_t)]))
            ne = RectBivariateSpline(_ne_t,_ne_r,_ne)(t_vec,r_vec)
            ne_u = RectBivariateSpline(_ne_t,_ne_r,_ne_u)(t_vec,r_vec)
            ne_d = RectBivariateSpline(_ne_t,_ne_r,_ne_d)(t_vec,r_vec)
            Te = RectBivariateSpline(_Te_t,_Te_r,_Te)(t_vec,r_vec)
            Te_u = RectBivariateSpline(_Te_t,_Te_r,_Te_u)(t_vec,r_vec)
            Te_d = RectBivariateSpline(_Te_t,_Te_r,_Te_d)(t_vec,r_vec)
            
            # Now stretch profiles such that we have 75 eV at the LCFS
            ne,ne_u,ne_d,Te,Te_u,Te_d = stretch_profs(   # NB: stretch_prof works with time-dept profs
                t_vec,r_vec, Te, Te_u, Te_d, ne, ne_u, ne_d,
                Te_LCFS=Te_lcfs_eV
            )
            
            # standard deviations
            ne_std = (ne_u-ne_d)/2.
            Te_std = (Te_u-Te_d)/2.
            
            # set minimum of 3 eV and equal minimum uncertainty
            Te_std[Te<3e-3] = 3e-3
            Te[Te<3e-3] = 3e-3
            
            out = [r_vec, t_vec, ne, ne_std, Te, Te_std]  # NB: extra output t_vec


    if get_Ti:
        # Ti is always obtained as time independent from Hirex-Sr
        # don't stretch since we don't know relationship of Ti and Te at the LCFS really...

        Ti, Ti_u, Ti_d, Ti_t, Ti_r = Ti_arr
        Ti_std = (Ti_u - Ti_d)/2.
        Ti_std[Ti<3e-3] = 3e-3
        Ti[Ti<3e-3] = 3e-3

        if time_indpt_profs:
            out.append(np.mean(Ti,axis=0))
            out.append(np.mean(Ti_std,axis=0))  # approx
        else:
            out.append(Ti)
            out.append(Ti_std)
        return out, p_ne, p_Te, p_Ti
    else:       
        
        # ne in 10^20 m^-3; Te in keV
        return out, p_ne, p_Te  # r_vec, (t_vec,) ne, ne_std, Te, Te_std = out
    

def get_Ti_data(shot, t_min, t_max, tht=1, p_Te=None):
    ''' get Ti profile (merged with ETS at the edge, mixed A & B branches as appropriate) '''
    
    try:
        p_Ti = load_hirex_profs.get_final_prof(shot, t_min,t_max,tht=tht, plot_raw=False)
    except:
        # try alternative THT's:
        try:
            p_Ti = load_hirex_profs.get_final_prof(shot, t_min,t_max,tht=1, plot_raw=False)
        except:
            try:
                p_Ti = load_hirex_profs.get_final_prof(shot, t_min,t_max,tht=2, plot_raw=False)
            except:
                # give up
                p_Ti = copy.deepcopy(p_Te)
                p_Ti.y *= np.nan # Ti profiles are not available

    return p_Ti



# PFS this is the wrapper for the 2-point model
def Teu_2pt_model(shot,tmin,tmax, geqdsk=None, pressure_opt = 3, lambdaq_opt=1, rhop_vec=None, ne=None, Te=None):
    '''
    Get 2-point model prediction for Te at the LCFS.

    Parameters
    ----------
    shot : C-Mod shot number
    tmin, tmax : time window in seconds
    geqdsk : dictionary containing the processed geqdsk file
    pressure_opt: int, choice of method to get the volume-averaged plasma pressure used
    rhop_vec : 1D arr, sqrt(norm.pol.flux) vector for ne and Te profiles, only used if pressure_opt=2    
    ne : 1D arr, electron density profile in units of 1e20m^-3, only used if pressure_opt=2
    Te : 1D arr, electron temperature profile in units of keV, only used if pressure_opt=2

    Returns
    -------
    Tu_eV : 2-point model prediction for the electron temperature at the LCFS
    '''
    if geqdsk is None:
        geqdsk = cmod_tools.get_geqdsk_cmod(
                    shot, time*1e3, gfiles_loc = '/home/sciortino/EFIT/lya_gfiles/')

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
        
    # Several options to find vol-average pressure for Brunner scaling:
    if pressure_opt==1:
        # Use W_mhd rather than plasma pressure:
        vol =  geqdsk['fluxSurfaces']['geo']['vol']
        W_mhd = cmod_tools.get_CMOD_var(var='Wmhd',shot=shot, tmin=tmin, tmax=tmax, plot=False)
        p_Pa_vol_avg = 2./3. * W_mhd / vol[-1]

    elif pressure_opt==2:
        # find volume average within LCF from ne and Te profiles
        p_Pa = 2 * (ne*1e20) * (Te*1e3*q_electron)  # taking pe=pi
        indLCFS = np.argmin(np.abs(rhop_vec-1.0))
        p_Pa_vol_avg = aurora.vol_average(p_Pa[:indLCFS], rhop_vec[:indLCFS], geqdsk=geqdsk)[-1]

    elif pressure_opt==3:  # default for C-Mod, since Brunner himself used this formula
        # use tor beta to extract vol-average pressure
        BTaxis = np.abs(cmod_tools.get_CMOD_var(var='Bt',shot=shot, tmin=tmin, tmax=tmax, plot=False))
        betat = cmod_tools.get_CMOD_var(var='betat',shot=shot, tmin=tmin, tmax=tmax, plot=False)
        p_Pa_vol_avg = (betat/100)*BTaxis**2.0/(2.0*4.0*np.pi*1e-7)   # formula used by D.Brunner

    else:
        raise ValueError('Undefined option for volume-averaged pressure')


    # B fields at the LFS LCFS midplane
    Rlcfs = aurora.rad_coord_transform(1.0, 'r/a', 'Rmid', geqdsk)
    #Rlcfs = eq.rho2rho('r/a', 'Rmid', 1, time)

    R_midplane = geqdsk['fluxSurfaces']['midplane']['R']
    Bp_midplane = geqdsk['fluxSurfaces']['midplane']['Bp']
    Bt_midplane = geqdsk['fluxSurfaces']['midplane']['Bt']
    Bp = np.abs(interp1d(R_midplane, Bp_midplane)(Rlcfs))
    Bt = np.abs(interp1d(R_midplane, Bt_midplane)(Rlcfs))

    if lambdaq_opt==1:
        # now get 2-point model prediction
        mod = twopoint_model.two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0,  # dummy value of ne at the sepatrix, not used for Tu_eV calculation
            lam_q_model='Brunner'
        )
    if lambdaq_opt==2:
        # now get 2-point model prediction
        mod = twopoint_model.two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0,  # dummy value of ne at the sepatrix, not used for Tu_eV calculation
            lam_q_model='Eich'
        )
    elif lambdaq_opt==3:
        # use lambdaT calculated from TS points to get lambda_q using lambda_q = 2/7*lambda_T
        mod = twopoint_model.two_point_model(
            0.69, 0.22,
            Psol, Bp, Bt, q95, p_Pa_vol_avg,
            1.0, 
            lam_q_model='lam_T'
        )

    return mod.Tu_eV, mod.lam_q_mm
    

def get_sawtooth_times(shot, t_min, t_max, lookahead=30, delta=0.1, return_data=False):
    ''' Get sawtooth times for a given shot by finding peaks in ECE signal. 

    `lookahead` and `delta` are parameters to find the peaks. 
    Varying them will increase/decrease sensitivity to ECE signal variations and 
    identify more/less events.
    '''
    eleTree = MDSplus.Tree('electrons',shot)
    nodeTe0 = eleTree.getNode('\electrons::gpc_te1')
    timeTe0 = nodeTe0.dim_of().data()
    ind1=np.searchsorted(timeTe0,t_min)
    ind2=np.searchsorted(timeTe0,t_max)
    time = timeTe0[ind1:ind2]
    Te0 = nodeTe0.data()[ind1:ind2]

    a,b = peakdetect(Te0,x_axis=time,lookahead=lookahead,delta=delta)
    # MODIFY lookahead (~100) to get useful number of peaks
    t_saw_max = [m[0] for m in a]
    Te_max_val =  np.asarray([m[1] for m in a])
    t_saw_min =[ m[0] for m in b]
    Te_min_val = np.asarray([m[1] for m in b])

    t_sawteeth = sorted(np.concatenate((t_saw_max,t_saw_min)))
    #t_sawteeth = np.concatenate(([t_min],t_sawteeth,[t_max]))  # include initial and final points

    if return_data:
        return t_sawteeth, time, Te0,t_saw_max,Te_max_val, t_saw_min,Te_min_val
    else:
        return t_sawteeth





def clean_profs(shot,p_ne, p_Te, min_ne_err, max_ne_err, min_Te_err, max_Te_err):
    ''' Clean up ne and Te profiles. 
    This is a rather arbitrary (and maybe not quite robust) method which is built by identifying 
    issues in certain shots. 
    '''
    # Eliminate TS points at r/a=[0.375,0.385] since they are faulty (check for every shot)
    p_Te.remove_points(np.logical_and(p_Te.X[:,1]>0.35, p_Te.X[:,1]<0.385))
    p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.35, p_ne.X[:,1]<0.385))

    p_Te.remove_points(np.logical_and(p_Te.X[:,1]>0.872, p_Te.X[:,1]<0.884))
    #p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.872, p_ne.X[:,1]<0.884))
    p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.86, p_ne.X[:,1]<0.884))

    p_ne.remove_points(np.logical_or(p_ne.err_y>max_ne_err, p_ne.err_y<min_ne_err))
    p_Te.remove_points(np.logical_or(p_Te.err_y>max_Te_err, p_Te.err_y<min_Te_err))

    # eliminate Te points that are too small or too large to be correct
    p_Te.remove_points(np.logical_and(p_Te.X[:,1]<0.98, p_Te.y<0.075))
    p_Te.remove_points(np.logical_and(p_Te.X[:,1]>1.01, p_Te.y>0.3))


    if shot==1101014011:
        # adjustment for substituted ETS
        p_Te.remove_points(np.logical_and(p_Te.X[:,1]<0.97, p_Te.y<0.2))

    if shot==1101014012:
        # hand-picked outliers
        p_Te.remove_points(np.logical_and(p_Te.X[:,1]>0.88, p_Te.y>0.97))

    if shot == 1101014030:
        p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.325, p_ne.X[:,1]<0.38))

    if shot==1120914029:
        # broken channels
        p_ne.remove_points(p_ne.y>1.5)
        p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.85, p_ne.y>1.12))
        p_ne.remove_points(np.logical_and(p_ne.X[:,1]<0.6, p_ne.y<0.85))

    if shot == 1120914036:
        p_ne.remove_points(np.logical_and(p_ne.X[:,1]<0.6, p_ne.y<1.0))
        p_Te.remove_points(p_Te.y>5.5)
        p_Te.remove_points(np.logical_and(p_Te.X[:,1]<0.6, p_Te.y<1.0))


    if shot == 1140729021:
        p_ne.remove_points(np.logical_and(p_ne.X[:,1]<0.92, p_ne.y<1.75))
        p_ne.remove_points(np.logical_and(p_ne.X[:,1]>0.25, p_ne.y>3.1))

    return p_ne,p_Te



# =========================================
def run_quickfit(p_ne, p_Te, t_sawteeth=None, dt_ne=1e-3, dt_Te=1e-4,
                 ne_zero_edge=False, Te_zero_edge=True, pedestal_rho_ne=1.0,pedestal_rho_Te=1.0,
                 eps_x_ne=0.5, eps_t_ne=1.0, eps_x_Te=0.45, eps_t_Te=0.7,
                 nr_pts=121, roa_max=1.1, p_Ti=None):
    ''' Run QUICKFIT given profiletools objects for ne and Te. 
    If t_sawteeth is not None, then this routines avoids smoothing over
    sawteeth. 

    If p_Ti is given, also fits Ti in the same way as Te.
    '''

    sawteeth = t_sawteeth
    elms= None
    elm_phase=None

    ### ne: 
    time_ne = p_ne.X[:,0]
    roa_ne = p_ne.X[:,1]
    y_ne = p_ne.y
    err_y_ne = p_ne.err_y
    P_ne = np.ones_like(y_ne, dtype=bool)   # point index --- only useful for nonlocal measurements
    W_ne = np.ones_like(y_ne, dtype=float)    # weight --- only useful for nonlocal measurements

    ### Now Te:
    time_Te = p_Te.X[:,0]
    roa_Te = p_Te.X[:,1]
    y_Te = p_Te.y
    err_y_Te = p_Te.err_y
    P_Te = np.ones_like(y_Te, dtype=bool)   # point index --- only useful for nonlocal measurements
    W_Te = np.ones_like(y_Te, dtype=float)    # weight --- only useful for nonlocal measurements

    # Fit and exclude outliers:
    ne_arr, Te_arr = get_quickfit(roa_ne,time_ne,y_ne, err_y_ne, P_ne, W_ne,dt_ne,
                                  roa_Te,time_Te,y_Te, err_y_Te, P_Te, W_Te,dt_Te,
                                  ne_zero_edge,Te_zero_edge,
                                  sawteeth, elms,elm_phase, pedestal_rho_ne, pedestal_rho_Te,
                                  eps_x_ne, eps_t_ne, eps_x_Te, eps_t_Te,nr_pts,roa_max)

    res = [ne_arr, Te_arr]
    if p_Ti is not None:
        time_Ti = p_Ti.X[:,0]
        roa_Ti = p_Ti.X[:,1]
        y_Ti = p_Ti.y
        err_y_Ti = p_Ti.err_y
        P_Ti = np.ones_like(y_Ti, dtype=bool)   # point index --- only useful for nonlocal measurements
        W_Ti = np.ones_like(y_Ti, dtype=float)    # weight --- only useful for nonlocal measurements

        # fix time resolution of Ti to be low:
        dt_Ti=10e-3 
        Ti_arr = get_Ti_quickfit(roa_Ti, time_Ti, y_Ti, err_y_Ti, P_Ti, W_Ti, dt_Ti,
                    Te_zero_edge, sawteeth, elms, elm_phase, pedestal_rho_Te,
                    eps_x_Te, eps_t_Te, nr_pts, roa_max)

        res.append( Ti_arr )
        
    return res

    
def get_quickfit(roa_ne,time_ne,y_ne, err_y_ne, P_ne, W_ne,dt_ne,
                 roa_Te,time_Te,y_Te, err_y_Te, P_Te, W_Te,dt_Te,
                 ne_zero_edge,Te_zero_edge,
                 sawteeth, elms,elm_phase, pedestal_rho_ne,pedestal_rho_Te,
                 eps_x_ne, eps_t_ne, eps_x_Te, eps_t_Te, nr_pts=121, roa_max=1.1):

    transformation = lambda x: np.log(np.maximum(x,0)/.1+1),\
                     lambda x:np.maximum(np.exp(x)-1,1.e-6)*.1,\
                     lambda x:1/(.1+np.maximum(0, x))  # transformations to better fit pedestal and SOL
        
    # ne
    MG_ne = map2grid(roa_ne,time_ne,y_ne,err_y_ne, P_ne, W_ne,nr_pts,dt_ne,r_max=roa_max)  #FS mod
    
    MG_ne.PrepareCalculation(
        #transformation=transformation,
        zero_edge=ne_zero_edge,
        core_discontinuties=sawteeth,edge_discontinuties= elms,
        robust_fit=False,elm_phase=elm_phase,pedestal_rho=pedestal_rho_ne)
    
    MG_ne.PreCalculate()
    ne, ne_u, ne_d, ne_t, ne_r = MG_ne.Calculate(eps_x_ne, eps_t_ne)

    # Te
    MG_Te = map2grid(roa_Te,time_Te,y_Te,err_y_Te, P_Te, W_Te,nr_pts,dt_Te,r_max=roa_max)  #FS mod
    
    MG_Te.PrepareCalculation(
        #transformation=transformation,
        zero_edge=Te_zero_edge,
        core_discontinuties=sawteeth,edge_discontinuties= elms,
        robust_fit=False,elm_phase=elm_phase,pedestal_rho=pedestal_rho_Te)
    
    MG_Te.PreCalculate()
    
    Te, Te_u, Te_d, Te_t, Te_r = MG_Te.Calculate(eps_x_Te, eps_t_Te)

    return (ne, ne_u, ne_d, ne_t, ne_r),(Te, Te_u, Te_d, Te_t, Te_r)
    

def get_Ti_quickfit(roa_Ti,time_Ti,y_Ti, err_y_Ti, P_Ti, W_Ti,dt_Ti,
                    Ti_zero_edge,sawteeth, elms,elm_phase, pedestal_rho_Ti,
                    eps_x_Ti, eps_t_Ti, nr_pts=121, roa_max=1.1):
    ''' Quickfit fitting of Ti '''
    MG_Ti = map2grid(roa_Ti,time_Ti,y_Ti,err_y_Ti, P_Ti, W_Ti,nr_pts,dt_Ti,r_max=roa_max)  #FS mod
    MG_Ti.PrepareCalculation(zero_edge=Ti_zero_edge, 
                             core_discontinuties=sawteeth,edge_discontinuties= elms,
                             robust_fit=False,elm_phase=elm_phase,pedestal_rho=pedestal_rho_Ti)
    MG_Ti.PreCalculate()
    
    Ti, Ti_u, Ti_d, Ti_t, Ti_r = MG_Ti.Calculate(eps_x_Ti, eps_t_Ti)

    return (Ti, Ti_u, Ti_d, Ti_t, Ti_r)

###########################################
#
#
#       Plotting methods
#
#
##############################################

def plot_3D_ne(roa_ne,time_ne,y_ne, ne_r,ne_t,ne, plot_raw=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if plot_raw:
        ax.scatter(roa_ne, time_ne, y_ne.T, c='r')
    ax.plot_surface(ne_r.T, ne_t.T, ne.T, rstride=10, cstride=10, alpha=0.5)  #explicit strides avoid annoying 3D spikes...
    ax.set_ylabel('time [s]', labelpad=10)
    ax.set_xlabel('r/a', labelpad=10)
    ax.set_zlabel(r'$n_e$ [$\times 10^{20} m^{-3}$]', labelpad=10)
    plt.tight_layout()


def plot_3D_Te(roa_Te,time_Te,y_Te, Te_r,Te_t,Te,plot_raw=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if plot_raw:
        ax.scatter(roa_Te, time_Te, y_Te.T, c='r')
    ax.plot_surface(Te_r.T,Te_t.T, Te.T, rstride=10, cstride=10, alpha=0.5)
    ax.set_ylabel('time [s]', labelpad=10)
    ax.set_xlabel('r/a', labelpad=10)
    ax.set_zlabel(r'$T_e$ [keV]', labelpad=10)

    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)
    plt.tight_layout()
    
# =============
def plot_Te_slider():
    # Slider plot for Te -- sliding in time
    from slider_scatter_plot import slider_plot # qt5 issue in py3
    
    f = plt.figure()
    gs = mplgs.GridSpec(2, 1, height_ratios=[10, 1])
    a_plot = f.add_subplot(gs[0, :])
    a_slider = f.add_subplot(gs[1, :])

    slider_plot(radial_vec_Te,time_vec_Te,
            np.expand_dims(Te.T,axis=0),
            np.expand_dims(old_div((Te_u.T - Te_d.T),2.),axis=0),
            xlabel='r/a', ylabel='time [s]',
            zlabel='',
            labels=['$T_e$'], plot_sum=False,
            axs = (f,a_plot,a_slider))

    slider_plot(radial_vec_Te,time_vec_Te,
            np.expand_dims(Te_expt.T,axis=0),
            np.expand_dims(Te_expt_unc.T,axis=0),
            xlabel='r/a', ylabel='time [s]',
            zlabel='',
            labels=['$T_e$'], plot_sum=False,
            axs = (f,a_plot,a_slider))

    '''
    # Slider plot for Te -- sliding in radius --- interpolation is clearly poor...
    f = plt.figure()
    gs = mplgs.GridSpec(2, 1, height_ratios=[10, 1])
    a_plot = f.add_subplot(gs[0, :])
    a_slider = f.add_subplot(gs[1, :])
        
    slider_plot(time_vec_Te,radial_vec_Te,
            np.expand_dims(Te,axis=0),
            np.expand_dims((Te_u - Te_d)/2.,axis=0),
            xlabel='time [s]', ylabel='r/a',
            zlabel='',
            labels=['$T_e$'], plot_sum=False,
            axs = (f,a_plot,a_slider))

    slider_plot(time_vec_Te,radial_vec_Te,
            np.expand_dims(Te_expt,axis=0),
            np.expand_dims(Te_expt_unc,axis=0),
            xlabel='time [s]', ylabel='r/a',
            zlabel='',
            labels=['$T_e$'], plot_sum=False,
            axs = (f,a_plot,a_slider))
    '''

    
def plot_static_Te(t_exp_binned_Te, time_vec_Te, radial_vec_Te, Te_u, Te_d, Te, roa_Te, indt_Te, y_Te, err_y_Te, static_time = [1.23], plot_raw=True, edge_focus=False):
    # Te --- Static plot

    if edge_focus:
        # only plot data in the edge region
        indr = np.argmin(np.abs(radial_vec_Te - 0.9))
        radial_vec_Te = radial_vec_Te[indr:]
        Te_u = Te_u[:,indr:]
        Te_d = Te_d[:,indr:]
        Te = Te[:,indr:]
        # all points listed in a single vector for expt data
        y_Te = y_Te[roa_Te>0.9]
        err_y_Te = err_y_Te[roa_Te>0.9]
        # don't sub-digitize Te points at the edge
        indt_Te = indt_Te[roa_Te>0.9]
        roa_Te = roa_Te[roa_Te>0.9]
        
    plt.figure()
    for stime in static_time:
        cc = next(c_cycle)
        
        tind_exp = np.argmin(np.abs(t_exp_binned_Te - stime))
        tind_fit = np.argmin(np.abs(time_vec_Te - stime))

        plt.fill_between(radial_vec_Te, Te_u[tind_fit,:],Te_d[tind_fit,:], alpha=0.2, color=cc)
        plt.plot(radial_vec_Te, Te[tind_fit,:],color=cc, label='t={} s'.format(stime))
        if plot_raw:
            plt.gca().errorbar(roa_Te[indt_Te==tind_exp], y_Te[indt_Te==tind_exp], err_y_Te[indt_Te==tind_exp],fmt='.',color=cc)
        plt.xlabel('r/a')
        plt.ylabel('$T_e$ $[keV]$')
        plt.legend()
    
# =============
# =============
def plot_ne_slider(radial_vec_ne, time_vec_ne, ne,ne_u,ne_d,ne_expt,ne_expt_unc):
    # Slider plot for ne -- sliding in time

    from slider_scatter_plot import slider_plot # qt5 issue in py3
    
    f = plt.figure()
    gs = mplgs.GridSpec(2, 1, height_ratios=[10, 1])
    a_plot = f.add_subplot(gs[0, :])
    a_slider = f.add_subplot(gs[1, :])   
    
    slider_plot(radial_vec_ne,time_vec_ne,
                np.expand_dims(ne.T,axis=0),
                np.expand_dims((ne_u.T - ne_d.T)/2.,axis=0),
                xlabel='r/a', ylabel='time [s]',
                zlabel='',
                labels=['$n_e$'], plot_sum=False,
                axs = (f,a_plot,a_slider))
    
    slider_plot(radial_vec_ne,time_vec_ne,
                np.expand_dims(ne_expt.T,axis=0),
                np.expand_dims(ne_expt_unc.T,axis=0),
                xlabel='r/a', ylabel='time [s]',
                zlabel='',
                labels=['$n_e$'], plot_sum=False,
                axs = (f,a_plot,a_slider))


def plot_static_ne(t_exp_binned_ne,time_vec_ne, radial_vec_ne,ne_u,ne_d,ne,roa_ne,indt_ne,y_ne,err_y_ne,static_time = [1.23], plot_raw=True):
    # ne --- Static plot
    plt.figure()
    
    for stime in static_time:
        cc = next(c_cycle)
        # plot multiple time slices/fits
        tind_exp = np.argmin(np.abs(t_exp_binned_ne -stime))
        tind_fit = np.argmin(np.abs(time_vec_ne - stime))
    
        plt.fill_between(radial_vec_ne, ne_u[tind_fit,:],ne_d[tind_fit,:], alpha=0.2, color=cc)
        plt.plot(radial_vec_ne, ne[tind_fit,:], color=cc)
        if plot_raw:
            plt.errorbar(roa_ne[indt_ne==tind_exp], y_ne[indt_ne==tind_exp], err_y_ne[indt_ne==tind_exp],fmt='.', color=cc)
        plt.xlabel('r/a')
        plt.ylabel('$n_e$ $[m^{-3}]$')
        

def compare_Te0(shot, time,Te0,t_saw_max,Te_max_val,t_saw_min,Te_min_val,Te_t,Te):

    plt.figure()
    plt.plot(time,Te0)
    plt.plot(t_saw_max,Te_max_val, 'ro')
    plt.plot(t_saw_min,Te_min_val, 'go')
    plt.title("shot: %d"%shot)

    plt.plot(Te_t[:,0], Te[:,0])
    plt.xlabel('time [s]')
    plt.ylabel('$T_e$ $[keV]$')



def stretch_profs_new(time_vec, r_vec, Te, ne, Te_LCFS=75.0):
    '''
    Stretch in x direction to match chosen temperature (in eV) at LCFS.
    Note that ne and Te must be on the same radial and time bases!
    '''
    TeShifted = copy.deepcopy(Te); neShifted = copy.deepcopy(ne); 
    
    # ensure a temperature of Te_LCFS eV at each time slice
    for ti,tt in enumerate(time_vec):
        x_of_TeSep = interp1d(TeShifted[ti,:], r_vec, bounds_error=False)(Te_LCFS*1e-3)
        xShifted = r_vec/x_of_TeSep
        TeShifted[ti,:] = interp1d(xShifted, TeShifted[ti,:], bounds_error=False)(r_vec)
        neShifted[ti,:] = interp1d(xShifted, ne[ti,:], bounds_error=False)(r_vec)

        # without extrapolation, some values at the edge may be set to nan. Set them to boundary value:
        whnan = np.isnan(TeShifted)
        if np.sum(whnan):
            TeShifted[whnan] = TeShifted[~whnan][-1]
        whnan = np.isnan(neShifted)
        if np.sum(whnan):
            neShifted[whnan] = neShifted[~whnan][-1]

    return neShifted, TeShifted


# PFS: this is the function I use to shift profiles
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


def stretch_profs(time_vec,r_vec, Te, Te_u, Te_d, ne, ne_u, ne_d, Te_LCFS=75.0):
    '''
    Stretch in x direction to match chosen temperature (in eV) at LCFS.
    Note that ne and Te must be on the same radial and time bases!
    '''

    TeShifted = copy.deepcopy(Te); neShifted = copy.deepcopy(ne); 
    TeShifted_u = copy.deepcopy(Te_u); neShifted_u = copy.deepcopy(ne_u)
    TeShifted_d = copy.deepcopy(Te_d); neShifted_d = copy.deepcopy(ne_d)
    
    # ensure a temperature of 75 eV at each time slice
    for ti,tt in enumerate(time_vec):
        x_of_TeSep = interp1d(TeShifted[ti,:], r_vec, bounds_error=False)(Te_LCFS*1e-3)

        # store x_of_TeSep in a temporary file for easy reading
        with open('x_of_TeSep_tmp.pkl','wb') as f:
            pkl.dump(x_of_TeSep, f)
        print(f'Te of LCFS found at x={x_of_TeSep}')
        xShifted = r_vec/x_of_TeSep
        TeShifted[ti,:] = interp1d(xShifted, TeShifted[ti,:], bounds_error=False)(r_vec)
        TeShifted_u[ti,:] = interp1d(xShifted, TeShifted_u[ti,:], bounds_error=False)(r_vec)
        TeShifted_d[ti,:] = interp1d(xShifted, TeShifted_d[ti,:], bounds_error=False)(r_vec)
        neShifted[ti,:] = interp1d(xShifted, neShifted[ti,:], bounds_error=False)(r_vec)
        neShifted_u[ti,:] = interp1d(xShifted, neShifted_u[ti,:], bounds_error=False)(r_vec)
        neShifted_d[ti,:] = interp1d(xShifted, neShifted_d[ti,:], bounds_error=False)(r_vec)        

        # without extrapolation, some values at the edge may be set to nan. Set them to boundary value:
        whnan = np.isnan(TeShifted)
        if np.sum(whnan):
            TeShifted[whnan] = TeShifted[~whnan][-1]
        whnan = np.isnan(neShifted)
        if np.sum(whnan):
            neShifted[whnan] = neShifted[~whnan][-1]

    return neShifted, neShifted_u, neShifted_d, TeShifted, TeShifted_u, TeShifted_d



def get_time_ave_profs(ne_arr, Te_arr):
    ''' Time average fitted kinetic profiles '''
    ne, ne_u, ne_d, ne_t, ne_r = ne_arr
    Te, Te_u, Te_d, Te_t, Te_r = Te_arr

    # all slices are identical:
    ne_r = ne_r[0,:]
    Te_r = Te_r[0,:]
    ne_t = ne_t[:,0]
    Te_t = Te_t[:,0]

    # average through given time window
    ne_av = np.mean(ne, axis=0)
    ne_u_av = np.mean(ne_u, axis=0)
    ne_d_av = np.mean(ne_d, axis=0)
    Te_av = np.mean(Te, axis=0)
    Te_u_av = np.mean(Te_u,axis=0)
    Te_d_av = np.mean(Te_d,axis=0)

    # Interpolate ne and Te onto the same radial grid
    r_vec = np.linspace(0.0,np.minimum(np.max(ne_r),np.max(Te_r)), np.maximum(len(ne_r),len(Te_r)))
    ne = np.atleast_2d(interp1d(ne_r, ne_av)(r_vec))
    Te = np.atleast_2d(interp1d(Te_r, Te_av)(r_vec))
    ne_u = np.atleast_2d(interp1d(ne_r, ne_u_av)(r_vec))
    ne_d = np.atleast_2d(interp1d(ne_r, ne_d_av)(r_vec))
    Te_u = np.atleast_2d(interp1d(Te_r, Te_u_av)(r_vec))
    Te_d = np.atleast_2d(interp1d(ne_r, Te_d_av)(r_vec))

    return r_vec,ne, ne_u,ne_d, Te,Te_u,Te_d




def stretch_time_indpt_profs_new(r_vec,ne,Te,Te_LCFS=75.0):

    # Now stretch profiles such that we have 75 eV at the LCFS
    ne, Te = stretch_profs_new([1.], r_vec, Te, ne, Te_LCFS=Te_LCFS) # NB: stretch_prof works with time-dept profs
    
    ne_av=ne[0,:]
    Te_av=Te[0,:]

    # set minimum of 3 eV
    Te_av[Te_av<3e-3] = 3e-3
    
    return [ne_av, Te_av]



def stretch_time_indpt_profs(r_vec,ne, ne_u,ne_d, Te,Te_u,Te_d,Te_LCFS=75.0):

    # Now stretch profiles such that we have 75 eV at the LCFS
    ne,ne_u,ne_d,Te,Te_u,Te_d = stretch_profs(   # NB: stretch_prof works with time-dept profs
        [1.0],r_vec, Te, Te_u, Te_d, ne, ne_u, ne_d, Te_LCFS=Te_LCFS
    )
    ne_av=ne[0,:]; ne_u_av=ne_u[0,:]; ne_d_av=ne_d[0,:]
    Te_av=Te[0,:]; Te_u_av=Te_u[0,:]; Te_d_av=Te_d[0,:]

    # standard deviations
    ne_std_av = (ne_u_av-ne_d_av)/2.
    Te_std_av = (Te_u_av-Te_d_av)/2.

    # set minimum of 3 eV and equal minimum uncertainty
    Te_std_av[Te_av<3e-3] = 3e-3
    Te_av[Te_av<3e-3] = 3e-3
    
    return [r_vec, ne_av, ne_std_av, Te_av, Te_std_av]



def stretch_profs_time_indpt_new(ne_arr,Te_arr, Te_LCFS=75.0):
    ''' Wrapper of stretch_profs to produce time-indepedent profile fits that meet the Te=Te_LCFS eV
    condition at the LCFS.  '''

    r_vec,ne, ne_u,ne_d,Te,Te_u,Te_d = get_time_ave_profs(ne_arr,Te_arr)
    
    ne_av, Te_av = stretch_time_indpt_profs_new(r_vec,ne, Te, Te_LCFS=Te_LCFS)

    # standard deviations (don't stretch upper and lower bounds; that will force lower and upper bounds to go through
    ne_std_av = (ne_u-ne_d)/2.
    Te_std_av = (Te_u-Te_d)/2.
    
    return [r_vec, ne_av, ne_std_av]


def stretch_profs_time_indpt(ne_arr,Te_arr, Te_LCFS=75.0):
    ''' Wrapper of stretch_profs to produce time-indepedent profile fits that meet the Te=Te_LCFS eV
    condition at the LCFS.  '''

    r_vec,ne, ne_u,ne_d,Te,Te_u,Te_d = get_time_ave_profs(ne_arr,Te_arr)
    
    return stretch_time_indpt_profs(r_vec,ne, ne_u,ne_d,Te,Te_u,Te_d)

