# Written by Francesco Sciortino, modified by Andres Miller

import matplotlib.pyplot as plt
plt.ion()
import numpy as np, copy
import pickle as pkl
import os, scipy
from scipy.interpolate import interp1d, RectBivariateSpline
import matplotlib.ticker as mtick
from IPython import embed
import single_shot, cmod_tools
import power_balance as pb
import data_access as da
from scipy.optimize import curve_fit
from omfit_classes.omfit_mds import OMFITmdsValue
import time as _time

################################################################

# set variables to create database in your own directory 
data_loc = '/home/millerma/thomson_separatrix/shotlists/'
db_filestem = 'test_db.txt'
regime = 'any'
each_thomson = True

##########################################################

rhop_loc=1.0 #0.99 #1.0
rhop_ped_range = [0.95,0.99] # range of rhop that is considered "pedestal" for gradient scale lengths
rhop_vec = np.linspace(0.94, 1.1, 100)

rhop_sol = 1.01 # rhop value at which SOL quantities shall we evaluated

# load list of shots and times
tables = {}
tables[regime] = np.loadtxt(data_loc+db_filestem+'.txt', skiprows=2, usecols=(1,2,3))

mds_vars = ['Bt','Bp','betat','Ip','nebar','P_RF','P_rad_diode','P_rad_main','p_D2','q95','Wmhd','dWdt',\
            'Lgap', 'Rgap', 'kappa', 'ssep', 'Udelta', 'Ldelta',\
            'P_oh','li','p_E_BOT_MKS','p_B_BOT_MKS', 'p_F_CRYO_MKS', 'p_G_SIDE_RAT'] #, 'P_tot','P_sol']

var_list = ['ne_prof','ne_prof_unc','Te_prof','Te_prof_unc','pe_prof','pe_prof_unc',\
            'grad_ne_prof','grad_ne_prof_unc','grad_Te_prof','grad_Te_prof_unc','grad_pe_prof','grad_pe_prof_unc',\
            'ne_raw','ne_raw_unc','Te_raw','Te_raw_unc','pe_raw','pe_raw_unc',\
            'ne_fit_coefs','ne_mean_coefs','Te_fit_coefs','Te_mean_coefs','pe_fit_coefs','pe_mean_coefs',\
            'area_LCFS','R_prof','rhop_prof','R_raw','rhop_raw',\
            'ne_R','Te_R','pe_R','ne_rhop','Te_rhop','pe_rhop',\
            'shot','tmin','tmax','f_gw','n_m3_avg','T_eV_avg','p_Pa_avg','gas_rate','gas_fuel','gas_cum',\
            'gradB_drift_up', 'favorable','regime','widx','CRYO_ON','R_geo','a_geo','R_sep','lam_q','Bp_OMP','Bt_OMP','ts_mult_factor',\
            'outer_rho_fmp_m','outer_ne_fmp_cm3','outer_Te_fmp_eV','outer_Js_fmp_Acm2',\
            'inner_rho_fmp_m','inner_ne_fmp_cm3','inner_Te_fmp_eV','inner_Js_fmp_Acm2',\
            'upper_rho_fmp_m','upper_ne_fmp_cm3','upper_Te_fmp_eV','upper_Js_fmp_Acm2']


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def main_db():

    windows={}
    win = 0
    for regime, table in tables.items(): 
        
        # sort the shot list if not already - probably not strictly necessary
        sortshot = np.argsort(tables[regime][:,0]) # sort it
        tables[regime] = tables[regime][sortshot]
    
        compute_range = np.arange(tables[regime].shape[0])
        windows[regime] = {}

        for ii in compute_range:
            shot = int(tables[regime][ii,0])

            if each_thomson:

                # get thomson time points
                # import MDSplus
                # electrons = MDSplus.Tree('electrons', shot)

                thomson_times = np.linspace(0,2,100)#electrons.getNode('\\electrons::top.yag_edgets.results:ne').dim_of(0).data()

                thomson_start = round(tables[regime][ii,1],3)
                thomson_end = round(tables[regime][ii,2],3)
                mask_shot = (thomson_times >= thomson_start) & (thomson_times <= thomson_end)

                for tt in thomson_times[mask_shot]:

                    tmin = round(tt - 1e-4, 4) # just to be safe
                    tmax = round(tt + 1e-4, 4) # just to be safe

                    if shot not in windows[regime]:
                        windows[regime][shot] = {}
                        n = 0

                    windows[regime][shot][n] = [tmin, tmax]
                    n += 1
                    win += 1

            else:

                tmin = round(tables[regime][ii,1],3)
                tmax = round(tables[regime][ii,2],3)

                if shot not in windows[regime]:
                    windows[regime][shot] = {}
                    n = 0

                windows[regime][shot][n] = [tmin, tmax]
                n += 1
                win += 1

    print('Number of windows in database:',win)

    res = {}
    for var in var_list + mds_vars:
        res[var] = []

    db_exists = False
    try:
        with open(db_filestem+'.pkl', 'wb') as f:
            res = pkl.load(f)
        db_exists = True
        print('Using existing database file')    

    # if db exists, entries will likley be stored as arrays
        for field in res:
            res[field] = list(res[field])

    except Exception:
        print('No database file exists; creating new one')    

    res = run_db(res, windows)

    # turn them into arrays to store them
    for field in res:
        res[field] = np.array(res[field])
    
    num_windows_to_store = len(res['shot'])
    print('Requested {} windows - storing {} windows'.format(win, num_windows_to_store-num_windows_db))

    # store results
    with open(db_filestem+'.pkl', 'wb') as f:
        pkl.dump(res,f)    



def run_db(res, windows, plot_kin=False, user_check=False):
    # create database on all (or num_windows) 100-ms-long time windows

    curr_window = 0
    ftz = True
    t0 = _time.time()

    for regime in windows:
        for shot in windows[regime]:
            for widx in windows[regime][shot]:
              
                tmin = windows[regime][shot][widx][0]
                tmax = windows[regime][shot][widx][1]

                time = (tmax+tmin)/2.
                print(shot, regime, widx, tmin, tmax)

                try: 
                    mult_factor = cmod_tools.get_ts_tci_ratio(shot, tmin, tmax, plot=False)
                    print('TS multiplication factor is {:.2f}'.format(mult_factor))
                    core_ts_mult, edge_ts_mult = True, True

                except:
                    print('Some stupid thing happened, skipping')
                    mult_factor = np.nan
                    core_ts_mult, edge_ts_mult = False, False

                try:
                    kp_out = cmod_tools.get_cmod_kin_profs(shot,tmin,tmax,
                                                           apply_final_sep_stretch=False, force_to_zero=ftz,
                                                           frac_err=False, num_mc=5, 
                                                           core_ts_mult=False,
                                                           edge_ts_mult=False)

                    f_ne, f_Te, f_pe, p_ne, p_Te, p_pe  = kp_out
 
                    if plot_kin:
                    
                        fig,ax = plt.subplots()
                        ax.plot(f_ne.x, f_ne.y, 'k-')
                        ax.errorbar(p_ne.X, p_ne.y, p_ne.err_y, fmt='.')
                        ax.set_xlabel('r/a')
                        ax.set_ylabel('n_e')

                        fig,ax = plt.subplots()
                        ax.plot(f_Te.x, f_Te.y, 'k-')
                        ax.errorbar(p_Te.X, p_Te.y, p_Te.err_y, fmt='.')
                        ax.set_xlabel('r/a')
                        ax.set_ylabel('T_e')
                    
                    if user_check:
                        input("Press Enter to continue...")
                        embed()

                    # exclude shots that don't have enough ne,Te data points near LCFS over the chosen interval
                    maskped = np.logical_and(p_ne.X[:,0]>0.90, p_ne.X[:,0]<1.01)
                    maskSOL = p_ne.X[:,0]>1

                    print('Data points in pedestal: {}'.format(np.sum(maskped)))
                    print('Data points in SOL: {}'.format(np.sum(maskSOL)))
                    
                    print(f'All good for shot {shot}, tmin={tmin}, tmax={tmax}!')
                except Exception as e:
                    print('Exception: ')
                    print(e)
                    import traceback; print(traceback.format_exc())
                    print(f'Could not get ne,Te profiles for shot {shot}, tmin={tmin}, tmax={tmax}. Skipping.')
                    continue

                if f_Te.y[-1]>100e-3 or f_Te.y[0]<0.5 or f_ne.y[0]<0.5:
                    # if fits give any of these conditions, something must have silently failed
                    print('Fits for shot {shot}, tmin={tmin}, tmax={tmax} seem wrong. Skipping.')
                    #continue - dont need to kill the process, just flag it

                #####
                res['regime'].append(regime)
                res['shot'].append(shot)
                res['widx'].append(widx)                
                res['tmin'].append(tmin)
                res['tmax'].append(tmax)
                res['ts_mult_factor'].append(mult_factor)
                
                # collect quantities from MDS
                for var in mds_vars:
                    try:
                        if var=='dWdt': # this is pretty dependent on the time slice you give it so give it a time slice
                            ddata = da.get_CMOD_var(var=var,shot=shot,tmin=tmin,tmax=tmax,return_time=False)
                            val = np.mean(ddata)
                        else:
                            t,ddata = da.get_CMOD_var(var=var,shot=shot,return_time=True)
                            if var=='ssep':
                                # ~1 for USN, -1 for LSN, 0 are DN. Must carefully eliminate values of ~40 because they are spurious
                                _val = ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))]
                                mask_ssep = np.logical_and(_val<3, _val>-3)
                                val = np.mean(_val[mask_ssep]) if len(_val[mask_ssep]) else np.nan

                                # cases with np.nan are typically inner-wall limited plasmas!
                            else:
                                # simple mean
                                if ddata is not None and np.any(~np.isnan(ddata)):
                                    val= np.mean(ddata[np.argmin(np.abs(t-tmin)):np.argmin(np.abs(t-tmax))])
                                else:
                                    val = np.nan
                        res[var].append(val)
                    except Exception as e:
                        print(e)
                        res[var].append(np.nan)

                # determine grad-B drift direction and divertor location
                gradB_drift_up = False if res['Bt'][-1]<0 else True
                res['gradB_drift_up'].append(gradB_drift_up)
                # define USN as having ssep>0.1 and LSN as having ssep<-0.1
                if np.isnan(res['ssep'][-1]):  # IWL plasmas
                    res['favorable'].append(np.nan)
                elif -0.1<res['ssep'][-1]<0.1:
                    res['favorable'].append(0.)
                elif (gradB_drift_up and res['ssep'][-1]>0.1) or (gradB_drift_up==False and res['ssep'][-1]<-0.1):
                    res['favorable'].append(1.)
                else:
                    res['favorable'].append(-1.)
                        
                # get gas-puff rate and cumulative gas fueling
                try:
                    gas_time, gas_tot = cmod_tools.get_CMOD_gas_fueling(shot, plot=False) #Torr-l
                
                    # get an average gas amount over time window
                    itmin = np.argmin(np.abs(gas_time-tmin))
                    itmax = np.argmin(np.abs(gas_time-tmax))
                    res['gas_fuel'].append(np.mean(gas_tot[itmin:itmax]))
                
                    # get rate at which gas is being added on average during time window
                    gas_rate = smooth(np.gradient(smooth(gas_tot,31), gas_time),31)  # additional smoothing (x2)
                    res['gas_rate'].append(np.mean(gas_rate[itmin:itmax]))
                
                    # cumulative total gas up to time in the middle of the time window
                    res['gas_cum'].append(np.sum(gas_tot[:int((itmin+itmax)/2)]))       

                except:
                    print('Could not load data for fueling')
                    res['gas_fuel'].append(np.nan)
                    res['gas_rate'].append(np.nan)
                    res['gas_cum'].append(np.nan)
        
                # recalculate lam_q to add to database
                lam_T_nl = 1
                Te_sep_eV, lam_q_mm = pb.Teu_2pt_model(shot, tmin, tmax, lambdaq_opt=1)
                res['lam_q'].append(lam_q_mm)
                    
                # get efit info - these are important for separatrix OS, but do not know how to do this with eqtools
                try:
                    # load magnetic geometry
                    geqdsk, gfile_name = cmod_tools.get_geqdsk_cmod(
                             shot, time*1e3, gfiles_loc = '/home/millerma/lya/gfiles/', return_fname=True)
                
                    res['R_geo'].append(geqdsk['fluxSurfaces']['geo']['R'][-1]) # indexes LCFS
                    res['a_geo'].append(geqdsk['fluxSurfaces']['geo']['a'][-1]) # LCFS
                    
                    # 'areao' output from aeqdsk doesn't seem to give area of LCFS in m^2.... use clearer output in geqdsk
                    res['area_LCFS'].append(geqdsk['fluxSurfaces']['geo']['surfArea'][-1])
                
                except:
                    res['R_geo'].append(np.nan)
                    res['a_geo'].append(np.nan)
                    res['area_LCFS'].append(np.nan)
    
                try: # EFIT20 only exists for shots from certain years
                    e = da.CModEFITTree(int(shot), tree='EFIT20', length_unit='m')
                except:
                    e = da.CModEFITTree(int(shot), tree='analysis', length_unit='m')
                time = (tmin + tmax)/2

                res['R_sep'].append(e.rho2rho('sqrtpsinorm', 'Rmid', 1, time))

                # request for Bpol at the midplane
                Bp_OMP = np.abs(e.rz2BZ(res['R_sep'][-1], 0, time))
                Bt_OMP = np.abs(e.rz2BT(res['R_sep'][-1], 0, time))
                
                res['Bp_OMP'].append(Bp_OMP)
                res['Bt_OMP'].append(Bt_OMP)

                sres = single_shot.assemble_into_dict(shot, tmin, tmax, 
                                                        f_ne, f_Te, f_pe,
                                                        p_ne, p_Te, p_pe)
                lyares_fit = sres['fit']
                lyares_raw = sres['raw']

                res['ne_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['ne'],bounds_error=False)(rhop_vec))
                res['ne_prof_unc'].append(interp1d(lyares_fit['rhop'],lyares_fit['ne_unc'],bounds_error=False)(rhop_vec))
                res['Te_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['Te'],bounds_error=False)(rhop_vec))
                res['Te_prof_unc'].append(interp1d(lyares_fit['rhop'],lyares_fit['Te_unc'],bounds_error=False)(rhop_vec))
                res['pe_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['pe'],bounds_error=False)(rhop_vec))
                res['pe_prof_unc'].append(interp1d(lyares_fit['rhop'],lyares_fit['pe_unc'],bounds_error=False)(rhop_vec))
                
                res['grad_ne_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['grad_ne'],bounds_error=False)(rhop_vec))
                res['grad_ne_prof_unc'].append(interp1d(lyares_fit['rhop'],lyares_fit['grad_ne_unc'],bounds_error=False)(rhop_vec))
                res['grad_Te_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['grad_Te'],bounds_error=False)(rhop_vec))
                res['grad_Te_prof_unc'].append(interp1d(lyares_fit['rhop'],lyares_fit['grad_Te_unc'],bounds_error=False)(rhop_vec))
                res['grad_pe_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['grad_pe'],bounds_error=False)(rhop_vec))
                res['grad_pe_prof_unc'].append(interp1d(lyares_fit['rhop'],lyares_fit['grad_pe_unc'],bounds_error=False)(rhop_vec))
                
                res['R_prof'].append(interp1d(lyares_fit['rhop'],lyares_fit['R'],bounds_error=False)(rhop_vec))
                res['rhop_prof'].append(rhop_vec)
                
                # raw data
                res['ne_R'].append(lyares_raw['ne_R'])
                res['ne_rhop'].append(lyares_raw['ne_rhop'])
                res['ne_raw'].append(lyares_raw['ne'])
                res['ne_raw_unc'].append(lyares_raw['ne_unc'])
    
                res['Te_R'].append(lyares_raw['Te_R'])
                res['Te_rhop'].append(lyares_raw['Te_rhop'])
                res['Te_raw'].append(lyares_raw['Te'])
                res['Te_raw_unc'].append(lyares_raw['Te_unc'])
                
                res['pe_R'].append(lyares_raw['pe_R'])
                res['pe_rhop'].append(lyares_raw['pe_rhop'])
                res['pe_raw'].append(lyares_raw['pe'])
                res['pe_raw_unc'].append(lyares_raw['pe_unc'])
   
                # these are the ones that will be used for neutral inferences 
                res['R_raw'].append(lyares_raw['R'])
                res['rhop_raw'].append(lyares_raw['rhop'])

                # store coefficients for fits
                res['ne_fit_coefs'].append(f_ne.popt[0])
                res['ne_mean_coefs'].append(f_ne.popt[1])
                res['Te_fit_coefs'].append(f_Te.popt[0])
                res['Te_mean_coefs'].append(f_Te.popt[1])
                res['pe_fit_coefs'].append(f_pe.popt[0])
                res['pe_mean_coefs'].append(f_pe.popt[1])

                # identify which shots had the cryopump cold/operating:
                try:
                    import MDSplus
                    edge = MDSplus.Tree('edge',shot)
                    node = edge.getNode('\edge::top.cryopump:message')
                    
                    if node.data() is None:
                        res['CRYO_ON'].append(False)
                    else:
                        res['CRYO_ON'].append(node.data()[0]=='CRYOPUMP ON')
                except:
                    res['CRYO_ON'].append(False)
        
        t1 = _time.time()
        print(f'Time to create database: {t1 - t0} s')

    return res

if __name__ == '__main__':
    res = main_db()

