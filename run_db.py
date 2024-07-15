# Written by Francesco Sciortino, modified by Andres Miller

import matplotlib.pyplot as plt
plt.ion()
import numpy as np, copy
import pickle as pkl
import os, scipy
from scipy.interpolate import interp1d, RectBivariateSpline
import matplotlib.ticker as mtick
from IPython import embed
import single_shot, cmod_tools, fit_2D
from scipy.optimize import curve_fit
from twopoint_model import two_point_model
from omfit_classes.omfit_mds import OMFITmdsValue
import time as _time

################################################################

# set variables to create database in your own directory 
data_loc = '/home/millerma/itpa_lmode/shotlists/'
db_filestem = 'jerry5_lmode_db'
regime = 'lmode'

##########################################################

rhop_loc=1.0 #0.99 #1.0
rhop_ped_range = [0.95,0.99] # range of rhop that is considered "pedestal" for gradient scale lengths
rhop_vec = np.linspace(0.94, 1.1, 100)

rhop_sol = 1.01 # rhop value at which SOL quantities shall we evaluated

# load list of shots and times
tables = {}
tables[regime] = np.loadtxt(data_loc+db_filestem+'.pkl', skiprows=2, usecols=(1,2,3))

mds_vars = ['Bt','Bp','betat','Ip','nebar','P_RF','P_rad_diode','P_rad_main','p_D2','q95','Wmhd','dWdt',\
            'Lgap', 'Rgap', 'kappa', 'ssep', 'Udelta', 'Ldelta',\
            'P_oh','li','p_E_BOT_MKS','p_B_BOT_MKS', 'p_F_CRYO_MKS', 'p_G_SIDE_RAT'] #, 'P_tot','P_sol']

var_list = ['ne_prof','ne_prof_unc','Te_prof','Te_prof_unc','pe_prof','pe_prof_unc',\
            'grad_ne_prof','grad_ne_prof_unc','grad_Te_prof','grad_Te_prof_unc','grad_pe_prof','grad_pe_prof_unc',\
            'ne_raw','ne_raw_unc','Te_raw','Te_raw_unc','pe_raw','pe_raw_unc',\
            'ne_fit_coefs','ne_mean_coefs','Te_fit_coefs','Te_mean_coefs','pe_fit_coefs','pe_mean_coefs',\
            'area_LCFS','R_prof','rvol_prof','rhop_prof','R_raw','R_raw_sp','rvol_raw','rhop_raw',\
            'ne_R','Te_R','pe_R','ne_rvol','Te_rvol','pe_rvol','ne_rhop','Te_rhop','pe_rhop',\
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
        
        tables[regime] = tables[regime]
        sortshot = np.argsort(tables[regime][:,0]) # sort it
        tables[regime] = tables[regime][sortshot]
    
        compute_range = np.arange(tables[regime].shape[0])
        windows[regime] = {}

        for ii in compute_range:
            shot = int(tables[regime][ii,0])
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

    num_windows = win
    db_exists = False
    try:
        with open(db_filestem+'.pkl', 'wb') as f:
            res = pkl.load(f)
        db_exists = True
        print('Using existing database file')    

    # if db exists, entries will likley be stored as arrays
        for field in res: #['nn_prof','S_ion_prof_unc','ne_prof','Te_prof','nn_prof_unc','S_ion_prof_unc','ne_prof_unc','Te_prof_unc','R_prof','rhop_prof']:
                          #['nn_raw','S_ion_raw','ne_raw','Te_raw','nn_raw_unc','S_ion_raw_unc','ne_raw_unc','Te_raw_unc','R_raw','rhop_raw']:
            res[field] = list(res[field])

    except Exception:
        print('No database file exists; creating new one')    

    
    # if dont want to run whole database, only run for some windows
    windows_in = {}
    curr_win = 0

    num_windows_db = len(res['shot']) if db_exists else 0
    shot_id = []
    for ii in range(num_windows_db):
        shot_str = res['shot'][ii].astype(str)
        tmin_str = res['tmin'][ii].astype(str)
        tmax_str = res['tmax'][ii].astype(str)
        shot_id.append(shot_str + tmin_str + tmax_str)

    for regime in windows:
        for shot in windows[regime]:
            for widx in windows[regime][shot]:
            
                if curr_win < num_windows:

                    if db_exists:                  

                        new_shot_str = str(shot)
                        new_tmin_str = str(windows[regime][shot][widx][0])
                        new_tmax_str = str(windows[regime][shot][widx][1])
                        curr_str = new_shot_str + new_tmin_str + new_tmax_str
 
                        if curr_str not in shot_id:
  
                        # check if time window already in database

                            if regime not in windows_in: windows_in[regime] = {} 
                            if shot not in windows_in[regime]: windows_in[regime][shot] = {}
                    
                            windows_in[regime][shot][widx] = windows[regime][shot][widx]
                            curr_win += 1

                    else: 
                        if regime not in windows_in: windows_in[regime] = {} 
                        if shot not in windows_in[regime]: windows_in[regime][shot] = {}
                        windows_in[regime][shot][widx] = windows[regime][shot][widx]
                        curr_win += 1

    from IPython import embed; embed()

    res = run_db(res, windows_in)

    # turn them into arrays to store them
    for field in res: #['nn_prof','S_ion_prof_unc','ne_prof','Te_prof','nn_prof_unc','S_ion_prof_unc','ne_prof_unc','Te_prof_unc','R_prof','rhop_prof']:
                      #['nn_raw','S_ion_raw','ne_raw','Te_raw','nn_raw_unc','S_ion_raw_unc','ne_raw_unc','Te_raw_unc','R_raw','rhop_raw']:
        res[field] = np.array(res[field])
    
    num_windows_to_store = len(res['shot'])
    print('Requested {} windows - storing {} windows'.format(curr_win, num_windows_to_store-num_windows_db))

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

                # load magnetic geometry
                geqdsk, gfile_name = cmod_tools.get_geqdsk_cmod(
                            shot, time*1e3, gfiles_loc = '/home/millerma/lya/gfiles/', return_fname=True)
                
                try:
                    kp_out = cmod_tools.get_cmod_kin_profs(shot,tmin,tmax, geqdsk = geqdsk, probes=['A','F'],
                                                           apply_final_sep_stretch=False, force_to_zero=ftz,
                                                           #frac_err=True)
                                                           frac_err=False,num_mc=2, 
                                                           core_ts_mult=core_ts_mult, core_ts_factor=mult_factor,
                                                           edge_ts_mult=edge_ts_mult, edge_ts_factor=mult_factor) #CHECK BEST WAY TO STORE MEAN COEFFICIENTS
                    f_ne, f_Te, f_pe, p_ne, p_Te, p_pe, num_SP = kp_out
 
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
                    
                    #input("Press Enter to continue...")
                    #embed()

                    # exclude shots that don't have enough ne,Te data points near LCFS over the chosen interval
                    maskped = np.logical_and(p_ne.X[:,0]>0.90, p_ne.X[:,0]<1.01)
                    maskSOL = p_ne.X[:,0]>1

                    print('Data points in pedestal: {}'.format(np.sum(maskped)))
                    print('Data points in SOL: {}'.format(np.sum(maskSOL)))

                    #if len(p_Te.X[maskLCFS])<6:
                    #    raise ValueError(f'Not enough data kinetic data in the pedestal!')
                    
                    # check if there exist any data point in the SOL
                    #if np.all(p_ne.X[:,0]<0.99):
                    #    print(f'No SOL ne data points inside of r/a=0.99 in shot {shot}!')
                    #    #raise ValueError(f'No SOL ne data points inside of r/a=0.99 in shot {shot}!')
                    #if np.all(p_Te.X[:,0]<0.99):
                    #    print(f'No SOL Te data points inside of r/a=0.99 in shot {shot}!')


                    
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
                    if var=='dWdt': # this is pretty dependent on the time slice you give it so give it a time slice
                        ddata = cmod_tools.get_CMOD_var(var=var,shot=shot,tmin=tmin,tmax=tmax,return_time=False)
                        val = np.mean(ddata)
                    else:
                        t,ddata = cmod_tools.get_CMOD_var(var=var,shot=shot,return_time=True)
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

                    
                # 'areao' output from aeqdsk doesn't seem to give area of LCFS in m^2.... use clearer output in geqdsk
                res['area_LCFS'].append(geqdsk['fluxSurfaces']['geo']['surfArea'][-1])
                
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
                Te_sep_eV, lam_q_mm = fit_2D.Teu_2pt_model(shot, tmin, tmax, lam_T_nl, geqdsk, pressure_opt = 3, lambdaq_opt=1)
                res['lam_q'].append(lam_q_mm)
                    
                # where is separatrix
                res['R_geo'].append(geqdsk['fluxSurfaces']['geo']['R'][-1]) # indexes LCFS
                res['a_geo'].append(geqdsk['fluxSurfaces']['geo']['a'][-1]) # LCFS
                res['R_sep'].append(aurora.rad_coord_transform(1, 'rhop', 'Rmid', geqdsk))

                # request for Bpol at the midplane

                Bp_OMP, Bt_OMP = cmod_tools.get_B_RZ(shot, tmin, tmax, R=res['R_sep'][-1], Z=0, gfile_name=gfile_name)
                res['Bp_OMP'].append(Bp_OMP)
                res['Bt_OMP'].append(Bt_OMP)


                ''' 
                ######## Shift profiles to align them with 2-point model prediction for Te at the LCFS
                # total input power
                RF_efficiency=0.8
                P_tot = RF_efficiency * res['P_RF'][-1] + res['P_oh'][-1]   # MW
                
                # power through LCFS  (NB: strictly speaking we should also consider dW/dt, but this must be small in steady-state)
                P_sol = np.clip(P_tot - res['P_rad'][-1],1e-10,None)    # ensure P_sol>=0 (or slightly more, to prevent 0 division)
                
                # find volume-averaged pressure for Brunner scaling from betat, as Brunner does himself
                BTaxis = np.abs(lyman_data.get_CMOD_var(var='Bt',shot=shot, tmin=tmin, tmax=tmax, plot=False))
                betat = lyman_data.get_CMOD_var(var='betat',shot=shot, tmin=tmin, tmax=tmax, plot=False)
                p_Pa_vol_avg = (res['betat'][-1]/100)*res['Bt'][-1]**2.0/(2.0*4.0*np.pi*1e-7)   # formula used by D.Brunner
                
                model = two_point_model(0.69, 0.22, P_sol, res['Bp'][-1], res['Bt'][-1], res['q95'][-1],
                                        p_Pa_vol_avg, 1.0) # dummy ne_sep at the end, not used for Tu_eV calculation
                
                # set a minimum Te at the LCFS in case some issue occurred in 2-pt model calculation
                Tu_eV = model.Tu_eV if model.Tu_eV>40.0 else 60.0   # assume 60 eV if calculation failed
                
                roa_of_TeSep = interp1d(_Te, roa_kp, bounds_error=False, fill_value=1)(Tu_eV*1e-3)  # keV
                
                #roaShifted = np.maximum(roa_kp + (1 - roa_of_TeSep),0)
                roaShifted = roa_kp/roa_of_TeSep
                Te = interp1d(roaShifted, _Te, bounds_error=False, fill_value=(_Te[0],None))(roa_kp)  # keV
                grad_Te = interp1d(roaShifted, _grad_Te, bounds_error=False, fill_value=(_grad_Te[0],None))(roa_kp)  # keV
                ne = interp1d(roaShifted, _ne, bounds_error=False, fill_value=(_ne[0],None))(roa_kp)  # 10^20 m^-3
                grad_ne = interp1d(roaShifted, _grad_ne, bounds_error=False, fill_value=(_grad_ne[0],None))(roa_kp)  # 10^20 m^-3
                
                # shift uncertainties in the same way -- not strictly rigorous, but decent
                Te_std = interp1d(roaShifted, _Te_std, bounds_error=False, fill_value=(_Te_std[0],None))(roa_kp)  # keV
                grad_Te_std = interp1d(roaShifted, _grad_Te_std, bounds_error=False, fill_value=(_grad_Te_std[0],None))(roa_kp)  # keV
                ne_std = interp1d(roaShifted, _ne_std, bounds_error=False, fill_value=(_ne_std[0],None))(roa_kp)  # 10^20 m^-3
                grad_ne_std = interp1d(roaShifted, _grad_ne_std, bounds_error=False, fill_value=(_grad_ne_std[0],None))(roa_kp)  # 10^20 m^-3
                
                # without extrapolation, some values at the edge may be set to nan. Set them to boundary value:
                Te[np.isnan(Te)] = Te[~np.isnan(Te)][-1] # keV
                ne[np.isnan(ne)] = ne[~np.isnan(ne)][-1] # 10^20 m^-3
                
                # try to shift TS profiles using TeSep
                p_ne.X += 1 - roa_of_TeSep
                p_Te.X += 1 - roa_of_TeSep
                '''

                ######
                
                # gather profiles (inputs in keV, & 10^20m^-3; outputs in eV & cm^-3)
                #single_res = single_case(shot,tmin,tmax, roa_kp, ne, ne_std, Te, Te_std,
                #                         p_ne, p_Te, SOL_exp_decay=True, decay_from_LCFS=False)
                #
                # results in cm^-3 and eV:
                #R, roa, rhop, ne_prof,ne_prof_unc, Te_prof, Te_prof_unc,\
                #    N1_prof,N1_prof_unc, emiss_prof, emiss_prof_unc, emiss_min, Te_min = single_res

                ### only get these profiles if I know there's lyman-alpha 
                # alternately, do a try except, but for now let's ignore for 
                # sep activity

                
                sres = single_shot.assemble_fit_into_dict(shot, tmin, tmax, f_ne, f_Te, f_pe, p_ne, p_Te, p_pe, geqdsk=geqdsk, SOL_exp_decay=False)
                lyares_fit = sres['fit']

                res_raw = single_shot.assemble_raw_into_dict(shot, tmin, tmax, p_ne, p_Te, p_pe, geqdsk=geqdsk, SOL_exp_decay=False)
                    
                sres.update(res_raw)
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
                res['rvol_prof'].append(interp1d(lyares_fit['rvol'],lyares_fit['R'],bounds_error=False)(rhop_vec))
                res['rhop_prof'].append(rhop_vec)
                
                # raw data
                res['ne_R'].append(lyares_raw['ne_R'])
                res['ne_rhop'].append(lyares_raw['ne_rhop'])
                res['ne_rvol'].append(lyares_raw['ne_rvol'])
                res['ne_raw'].append(lyares_raw['ne'])
                res['ne_raw_unc'].append(lyares_raw['ne_unc'])
    
                res['Te_R'].append(lyares_raw['Te_R'])
                res['Te_rhop'].append(lyares_raw['Te_rhop'])
                res['Te_rvol'].append(lyares_raw['Te_rvol'])
                res['Te_raw'].append(lyares_raw['Te'])
                res['Te_raw_unc'].append(lyares_raw['Te_unc'])
                
                res['pe_R'].append(lyares_raw['pe_R'])
                res['pe_rhop'].append(lyares_raw['pe_rhop'])
                res['pe_rvol'].append(lyares_raw['pe_rvol'])
                res['pe_raw'].append(lyares_raw['pe'])
                res['pe_raw_unc'].append(lyares_raw['pe_unc'])
   
                # these are the ones that will be used for neutral inferences 
                res['R_raw'].append(lyares_raw['R'])
                res['rvol_raw'].append(lyares_raw['rvol'])
                res['rhop_raw'].append(lyares_raw['rhop'])

                sp_mask = np.full(lyares_raw['R'].shape, True); sp_mask[:-num_SP['ne']] = False
                res['R_raw_sp'].append(lyares_raw['R'][sp_mask])

                # store coefficients for fits
                res['ne_fit_coefs'].append(f_ne.popt[0])
                res['ne_mean_coefs'].append(f_ne.popt[1])
                res['Te_fit_coefs'].append(f_Te.popt[0])
                res['Te_mean_coefs'].append(f_Te.popt[1])
                res['pe_fit_coefs'].append(f_pe.popt[0])
                res['pe_mean_coefs'].append(f_pe.popt[1])


                # identify which shots had the cryopump cold/operating:
                node = OMFITmdsValue(server='CMOD',shot=shot,treename='EDGE', TDI='\EDGE::TOP.CRYOPUMP:MESSAGE')
                if node.data() is None:
                    res['CRYO_ON'].append(False)
                else:
                    res['CRYO_ON'].append(node.data()[0]=='CRYOPUMP ON')
        
                # Compute volume averages
                Ip_MA = np.abs(res['Ip'][-1])/1e6
                # may want to check on this later - not sure if I trust it
                try:
                    res['f_gw'].append( cmod_tools.get_Greenwald_frac(shot,tmin,tmax, f_ne.x, f_ne.y, Ip_MA, geqdsk=geqdsk) )
                except:
                    # try again, some issue with omfit_omfit
                    res['f_gw'].append( cmod_tools.get_Greenwald_frac(shot,tmin,tmax, f_ne.x, f_ne.y, Ip_MA, geqdsk=geqdsk) )
                    
                # store final volume-averaged pressure [Pa], using 2-point model corrected profiles
                p_Pa_avg, n_m3_avg, T_eV_avg = cmod_tools.get_vol_avg(shot,time, rhop_vec, lyares_fit['ne'], lyares_fit['Te'], geqdsk=geqdsk, quantities=['p','n','T'])
                res['p_Pa_avg'].append( p_Pa_avg )
                res['n_m3_avg'].append( n_m3_avg )
                res['T_eV_avg'].append( T_eV_avg )

                # simple detachment indicator: pressure on divertor probes vs. separatrix pressure
                locs = ['outer', 'inner', 'upper']
                
                for loc in locs:
                    try:
                        _out_probes = cmod_tools.load_fmp_neTe(res['shot'][-1], tmin, tmax, loc=loc, get_max=False)
                        
                        if loc == 'upper':
                            neo_fmp, rhoo_fmp, Teo_fmp, _, Jso_fmp, _, nei_fmp, rhoi_fmp, Tei_fmp, _, Jsi_fmp, _ = _out_probes                
                            # for now let's just combine the upper stuff
                            rho_fmp = np.hstack((rhoo_fmp, rhoi_fmp))        
                            ne_fmp = np.hstack((neo_fmp, nei_fmp))        
                            Te_fmp = np.hstack((Teo_fmp, Tei_fmp))        
                            Js_fmp = np.hstack((Jso_fmp, Jsi_fmp))        
 
                        else:
                            ne_fmp, rho_fmp, Te_fmp, _, Js_fmp, _ = _out_probes                        

                        res[f'{loc}_rho_fmp_m'].append( rho_fmp )
                        res[f'{loc}_ne_fmp_cm3'].append( ne_fmp )
                        res[f'{loc}_Te_fmp_eV'].append( Te_fmp )
                        res[f'{loc}_Js_fmp_Acm2'].append( Js_fmp )

                    except Exception as e:
                        print(e)
                        # in some cases, only 1 data point seems to be available from FM probes and that breaks interpolation
                        res[f'{loc}_rho_fmp_m'].append( np.nan )
                        res[f'{loc}_ne_fmp_cm3'].append( np.nan )
                        res[f'{loc}_Te_fmp_eV'].append( np.nan )
                        res[f'{loc}_Js_fmp_Acm2'].append( np.nan )

        t1 = _time.time()
        print(f'Time to create database: {t1 - t0} s')

    return res

if __name__ == '__main__':
    res = main_db()

