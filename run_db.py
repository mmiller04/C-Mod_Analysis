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

# load ADAS H ionization rates
import aurora
atom_data = aurora.get_atom_data('H',['scd']) #'scd93_h.dat')


################################################################

# set variables to create database in your own directory 
data_loc = '/home/millerma/itpa_lmode/shotlists/'
#lyadb_filename = 'lmode07_db.pkl'
lyadb_filename = 'jerry5_lmode_db.pkl'
#lyadb_filename = 'brian5_lmode_db.pkl'

##########################################################

rhop_loc=1.0 #0.99 #1.0
rhop_ped_range = [0.95,0.99] # range of rhop that is considered "pedestal" for gradient scale lengths
rhop_vec = np.linspace(0.94, 1.1, 100)

rhop_sol = 1.01 # rhop value at which SOL quantities shall we evaluated

# load list of shots and times
tables = {}

# to use brian's edge database
#tables['brian_db'] = np.loadtxt(data_loc+'ts_edgedb_overlap3b.txt', skiprows=2,usecols=(1,2,3))
tables['lmode'] = np.loadtxt(data_loc+'jerry_db_lmodes5.txt', skiprows=2,usecols=(1,2,3))
#tables['hmode'] = np.loadtxt(data_loc+'jerry_db_edas2.txt', skiprows=2,usecols=(1,2,3))

# for dan's lambdaq database
#tables['dan_db'] = np.loadtxt(data_loc+'dans_shots.txt'.format(label), skiprows=2,usecols=(1,2,3))

# for seans db
#tables['sean_db'] = np.loadtxt(data_loc+'seans_shots_50ms.txt'.format(label), skiprows=2,usecols=(1,2,3))

#tables['lmode07'] = np.loadtxt(data_loc+'lyman_lmode_fy07_0320.txt', skiprows=2,usecols=(1,2,3))
#tables['lmode08'] = np.loadtxt(data_loc+'lyman_lmode_fy08_0320.txt', skiprows=2,usecols=(1,2,3))

name_map = {'hmode':'H-mode','lmode':'RF-heated L-mode','ohmic':'Ohmic L-mode','imode':'I-mode'}


# judged by eye, from filterscope signals:
ELMy = {1080304029:[0.9,1.4], 1080213027: [0.9,1.3], 1080213021: [0.85,0.95], # interval goes to 1.2
        1080213020: [0.85, 1.25], # interval goes to 1.5
        1080213018: [1.0, 1.2], # interval goes to 1.45
        1080213014: [1.15, 1.35], # interval goes from 0.95 to 1.45
        }

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


'''
for regime in windows:
    if regime=='ohmic':   #!!!!!!!
        continue
    
    for sidx, shot in enumerate(windows[regime]):
        
        for widx in windows[regime][shot]:

            tmin = windows[regime][shot][widx][0]
            tmax = windows[regime][shot][widx][1]

            time = (tmax+tmin)/2.
            print('----------------------')
            print(sidx, shot, regime, widx, tmin, tmax)

            # load magnetic geometry
            geqdsk = lyman_data.get_geqdsk_cmod(
                shot, time*1e3, gfiles_loc = '/home/sciortino/EFIT/lya_gfiles/')

            try:
                kp_out = lyman_data.get_cmod_kin_profs(shot,tmin,tmax, geqdsk = geqdsk,
                                                       apply_final_sep_stretch=True, force_to_zero=True)
            except Exception as e: # all types of exceptions
                print(e)
                print(f'Likely no edge Thomson data for shot {shot}, tmin={tmin}, tmax={tmax}. Exclude!')
                continue
            roa_kp, _ne, _ne_std, _Te, _Te_std, p_ne, p_Te, kpnum = kp_out #1e20 m^-3 and keV
            
            # check if there exist any data point in the SOL
            if np.all(p_ne.X[:,0]<1.0):
                print(f'No SOL ne data points in shot {shot}, tmin={tmin}, tmax={tmax}!')
                continue
            if np.all(p_Te.X[:,0]<1.0):
                print(f'No SOL Te data points in shot {shot}, tmin={tmin}, tmax={tmax}!')
                continue
            
            # exclude shots that don't have enough ne,Te data points near LCFS over the chosen interval
            maskLCFS = np.logical_and(p_Te.X[:,0]>0.95, p_Te.X[:,0]<1.01)
            if len(p_Te.X[maskLCFS])<6:
                print(f'Not enough data kinetic data in the pedestal for shot {shot}, tmin={tmin}, tmax={tmax}!')
                continue
            
            print(f'Acceptable data quality for shot {shot}, tmin={tmin}, tmax={tmax}')
'''
######
'''
# check which shots are double null
for regime, table in tables.items():
    compute_range = np.arange(tables[regime].shape[0])
    
    for ii in compute_range:
        shot = int(tables[regime][ii,0])
        tmin = round(tables[regime][ii,1],2)
        tmax = round(tables[regime][ii,2],2)
        time = (tmax+tmin)/2.

        for ijk in np.arange(10): # 10 attempts
            try:
                geqdsk = lyman_data.get_geqdsk_cmod(
                    shot, time*1e3, gfiles_loc = '/home/sciortino/EFIT/lya_gfiles/')
                geqdsk.load()  # for safety
                geqdsk['fluxSurfaces'].load()  # for safety
                break
            except:
                pass

        snode = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:ssep')
        id0 = np.argmin(np.abs(snode.dim_of(0) - tmin))
        id1 = np.argmin(np.abs(snode.dim_of(0) - tmax))
        ssep = snode.data()[id0:id1]

        rnode = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:rseps')
        znode = OMFITmdsValue(server='CMOD', shot=shot, treename='analysis',TDI='\EFIT_AEQDSK:zseps')
        print('ssep: ',ssep)
        plt.figure(); geqdsk.plot(only2D=True)
        plt.title(f'{ii}, {regime}, {shot},{tmin},{tmax}')
        input("Press Enter to continue...")

'''
'''
compute_range = np.arange(tables['hmode'].shape[0])

for ii in compute_range:
    shot = int(tables['hmode'][ii,0])
    tmin = round(tables['hmode'][ii,1],2)
    tmax = round(tables['hmode'][ii,2],2)
    time = (tmax+tmin)/2.

    t,data = lyman_data.get_CMOD_var(var='h_alpha',shot=shot, plot=True)
    plt.gca().set_title(f'{ii}, hmode, {shot},{tmin},{tmax}')
'''    
        
'''
# check which shots are double null
for regime, table in tables.items():
    compute_range = np.arange(tables[regime].shape[0])
    
    for ii in compute_range:
        shot = int(tables[regime][ii,0])
        tmin = round(tables[regime][ii,1],2)
        tmax = round(tables[regime][ii,2],2)
        time = (tmax+tmin)/2.

        for ijk in np.arange(10): # 10 attempts
            try:
                geqdsk = lyman_data.get_geqdsk_cmod(
                    shot, time*1e3, gfiles_loc = '/home/sciortino/EFIT/lya_gfiles/')
                geqdsk.load()  # for safety
                geqdsk['fluxSurfaces'].load()  # for safety
                break
            except:
                pass
            
        plt.figure(); geqdsk.plot(only2D=True)
        plt.title(f'{ii}, {regime}, {shot},{tmin},{tmax}')
        input("Press Enter to continue...")
'''


        
# If needed, delete all gfiles on disk to re-load/save them from MDS+
#import glob, os
#for case in tables:
#    for shot in tables[case][:,0]:
#        for filename in glob.glob(my_gfiles_dir+f'g{shot}*'):
#            print('Removed ', filename)
#            os.remove(filename)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


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


def main_db():

    #cut_regime = 5 

    # partition all good shots into 100 ms time windows
    windows={}
    win = 0
    for regime, table in tables.items(): 
        
        #tables[regime] = tables[regime][:cut_regime] # cut to number wanted
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
 
            # use this later if end up changing window size
            #tdiff = np.round((tmax-tmin),2)       
            #if tdiff >= 0.1:
            #    num_win=int(np.floor(tdiff/0.1))
            #
            #    for n in np.arange(num_win):
            #        windows[regime][shot][n] = [np.round(tmin+0.1*n,5), np.round(tmin+0.1*(n+1),5)]
            #    win+=num_win

    print('Number of windows in database:',win)

    res = {}
    for var in var_list + mds_vars:
        res[var] = []

    #num_windows = 5
    num_windows = win
    db_exists = False
    try:
        with open(lyadb_filename, 'wb') as f:
        #with open('lya_db_rest_07_08_1016.pkl', 'wb') as f:
            res = pkl.load(f)
        db_exists = True
        print('Using existing database file')    

    # if db exists, entries will be stored as arrays
        for field in res: #['nn_prof','S_ion_prof_unc','ne_prof','Te_prof','nn_prof_unc','S_ion_prof_unc','ne_prof_unc','Te_prof_unc','R_prof','rhop_prof']:
                          #['nn_raw','S_ion_raw','ne_raw','Te_raw','nn_raw_unc','S_ion_raw_unc','ne_raw_unc','Te_raw_unc','R_raw','rhop_raw']:
            res[field] = list(res[field])

    except Exception:
        print('No database file exists; creating new one')    

    
    # if dont want to run whole database, only run for some windows
    windows_in = {}
    curr_win = 0
 
#    while curr_win < num_windows:
#        for regime, table in tables.items():
#            compute_range = np.arange(tables[regime].shape[0])
#            windows_in[regime] = {}
#
#            for ii in compute_range:
#
#                shot = int(tables[regime][ii,0])
#                tmin = round(tables[regime][ii,1],2)
#                tmax = round(tables[regime][ii,2],2)
#
#                windows[regime][shot] = {}
#
#                if tmax-tmin > 0.1:

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
                        #in_regime = np.where(np.array(res['regime']) == regime)[0] 
                        #in_shot = np.where(np.array(res['shot'])[in_regime] == shot)[0]
                    
                        #if widx not in np.array(res['widx'])[in_shot]:  
                            if regime not in windows_in: windows_in[regime] = {} 
                            if shot not in windows_in[regime]: windows_in[regime][shot] = {}
                    
                            windows_in[regime][shot][widx] = windows[regime][shot][widx]
                            curr_win += 1

                    else: 
                        if regime not in windows_in: windows_in[regime] = {} 
                        if shot not in windows_in[regime]: windows_in[regime][shot] = {}
                        windows_in[regime][shot][widx] = windows[regime][shot][widx]
                        curr_win += 1

    res = run_db(res, windows_in)

    # turn them into arrays to store them
    for field in res: #['nn_prof','S_ion_prof_unc','ne_prof','Te_prof','nn_prof_unc','S_ion_prof_unc','ne_prof_unc','Te_prof_unc','R_prof','rhop_prof']:
                      #['nn_raw','S_ion_raw','ne_raw','Te_raw','nn_raw_unc','S_ion_raw_unc','ne_raw_unc','Te_raw_unc','R_raw','rhop_raw']:
        res[field] = np.array(res[field])
    
    num_windows_to_store = len(res['shot'])
    print('Requested {} windows - storing {} windows'.format(curr_win, num_windows_to_store-num_windows_db))
    # store results
    with open(lyadb_filename, 'wb') as f:
#    with open('lya_db_rest_07_08_1016.pkl', 'wb') as f:
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
                

    
#######################################
#
#             POSTPROCESSING
#
#######################################

def post_processing(res):
    data = {}
    for key in ['shot','nn_prof','nn_prof_unc','nn_loc','nn_loc_unc','nn_by_ne_loc','nn_by_ne_loc_unc',
                'Ln_ped_mm','Ln_ped_mm_unc', 'tmin','tmax','widx','regime', 'Ln_ped_exp_mm','Ln_ped_exp_mm_unc',
                'ne_loc','ne_loc_unc', 'ne_sol', 'ne_sol_unc', 'Te_sol','Te_sol_unc','CRYO_ON',
                #'grad_ne', 'grad_ne_unc', 'grad_Te','grad_Te_unc', 
                'Lne', 'LTe', 'Lne_unc','LTe_unc','Deff','Veff','Deff_unc','Veff_unc',
                'P_oh','P_tot','P_sol','P_sol/A','f_gw','gas_rate','gas_fuel','gas_cum', 'area_LCFS']+\
                mds_vars+\
                ['Gamma_D','ioniz_out','Gamma_D_unc','ioniz_out_unc','ne_sep','ne_sep_unc',
                'Te_sep', 'Te_sep_unc', 'lam_q','q_par', 'p_Pa_avg', 'detachment',
                'gradB_drift_up', 'favorable', 'p_E_BOT_MKS', 'p_B_BOT_MKS', 'p_F_CRYO_MKS',
                'l_e','l_cx_cold','l_cx_hot','nu_star']: #,'l_cx_cold_unc','l_cx_hot_unc','nu_star_unc']:
        data[key] = []
    
    for ii in np.arange(len(res['R_prof'])):   # loop over shots

        nn_by_ne_prof = res['nn_prof'][ii]/res['ne_prof'][ii]
        nn_by_ne_prof_unc = np.sqrt((res['nn_prof_unc'][ii]/res['ne_prof'][ii])**2+
                                    (res['nn_prof'][ii]/res['ne_prof'][ii]**2)**2*res['ne_prof_unc'][ii]**2)  #

        # exclude points where there might be nan or else interpolation at rhop_loc will give nans...
        mask = ~np.isnan(res['nn_prof'][ii])
        mask_ped = (rhop_vec>rhop_ped_range[0]) & (rhop_vec<rhop_ped_range[1]) & mask
    
        # Choose whether to extrapolate (should not be abused! Only appropriate near the LCFS, where we always have data)
        fill_value='extrapolate' #None    #doesn't seem to matter in practice?

        nn_loc = interp1d(rhop_vec[mask], res['nn_prof'][ii][mask],
                          bounds_error=False, fill_value=fill_value)(rhop_loc)
        nn_loc_unc = interp1d(rhop_vec[mask], res['nn_prof_unc'][ii][mask],
                              bounds_error=False, fill_value=fill_value)(rhop_loc)
        nn_by_ne_loc = interp1d(rhop_vec[mask], nn_by_ne_prof[mask],
                                bounds_error=False, fill_value=fill_value)(rhop_loc) 
        nn_by_ne_loc_unc = interp1d(rhop_vec[mask], nn_by_ne_prof_unc[mask],
                                    bounds_error=False, fill_value=fill_value)(rhop_loc)

        # interpolate some more useful variables
        R_loc = interp1d(rhop_vec[mask], res['R_prof'][ii][mask],
                         bounds_error=False, fill_value=fill_value)(rhop_loc)
        R_lcfs = interp1d(rhop_vec[mask], res['R_prof'][ii][mask],
                          bounds_error=False, fill_value=fill_value)(1.0)
        R_sol = interp1d(rhop_vec[mask], res['R_prof'][ii][mask],
                         bounds_error=False, fill_value=fill_value)(rhop_sol)
    
        ne_loc = interp1d(rhop_vec[mask], res['ne_prof'][ii][mask],
                          bounds_error=False, fill_value=fill_value)(rhop_loc) 
        ne_loc_unc = interp1d(rhop_vec[mask], res['ne_prof_unc'][ii][mask],
                              bounds_error=False, fill_value=fill_value)(rhop_loc)

        # evaluations in the SOL
        ne_sol = interp1d(rhop_vec[mask], res['ne_prof'][ii][mask],
                          bounds_error=False, fill_value=fill_value)(rhop_sol) 
        ne_sol_unc = interp1d(rhop_vec[mask], res['ne_prof_unc'][ii][mask],
                              bounds_error=False, fill_value=fill_value)(rhop_sol)
        Te_sol = interp1d(rhop_vec[mask], res['Te_prof'][ii][mask],
                          bounds_error=False, fill_value=fill_value)(rhop_sol)
        Te_sol_unc = interp1d(rhop_vec[mask], res['Te_prof_unc'][ii][mask],
                              bounds_error=False, fill_value=fill_value)(rhop_sol)    
        if rhop_loc==1.0:
            ne_sep = ne_loc
            ne_sep_unc = ne_loc_unc
        else:
            ne_sep = interp1d(rhop_vec[mask], res['ne_prof'][ii][mask],
                              bounds_error=False, fill_value=fill_value)(1.0) 
            ne_sep_unc = interp1d(rhop_vec[mask], res['ne_prof_unc'][ii][mask],
                                  bounds_error=False, fill_value=fill_value)(1.0)

        # set some minimum uncertainties
        nn_loc_unc[nn_loc_unc<0.2*nn_loc] = 0.2*nn_loc
        nn_by_ne_loc_unc[nn_by_ne_loc_unc<0.2*nn_by_ne_loc] = 0.2*nn_by_ne_loc
        ne_sep_unc[ne_sep_unc<0.2*ne_sep] = 0.2*ne_sep
        ne_sol_unc[ne_sol_unc<0.2*ne_sol] = 0.2*ne_sol
        Te_sol_unc[Te_sol_unc<0.2*Te_sol] = 0.2*Te_sol
        
        # check that we have some points in the pedestal; if not, there must be some issue with the Lya mapping
        # e.g. shot 1080416025, t=[0.9, 1.1], imode
        if len(rhop_vec[mask_ped])<5:
            # less than 5 points in the pedestal, not acceptable
            continue
        
        if (0.98<rhop_loc<1.01) and (nn_by_ne_loc>1.0 or nn_by_ne_loc<1e-6):
            # if nn/ne > 0.1 or <1e-6 near the separatrix, assume that something's wrong
            continue

        if nn_loc_unc/nn_loc>1.:
            # estimated uncertainty makes measurement not useful
            continue    

        # check if p_D2 value is reasonable. If not, skip case
        if np.abs(res['p_D2'][ii])>100:
            continue

        # After these few checks have passed, start saving variables
        data['shot'].append(res['shot'][ii])
        data['tmin'].append(res['tmin'][ii])
        data['tmax'].append(res['tmax'][ii])

        if res['regime'][ii]=='hmode':
            if res['shot'][ii] in ELMy:
                # check if ELMy or EDA H-mode
                if res['tmin'][ii]>=ELMy[res['shot'][ii]][0] and res['tmax'][ii]<=ELMy[res['shot'][ii]][1]:
                    # ELMy
                    data['regime'].append('ELMy')
                else:
                    # EDA
                    data['regime'].append('EDA')
            else:
                # EDA
                data['regime'].append('EDA')            
        else:
            data['regime'].append(res['regime'][ii])
                
        data['widx'].append(res['widx'][ii])

        data['nn_loc'].append(nn_loc)
        data['nn_loc_unc'].append(nn_loc_unc)
        data['nn_by_ne_loc'].append(nn_by_ne_loc)
        data['nn_by_ne_loc_unc'].append(nn_by_ne_loc_unc)
        
        for var in mds_vars:
            #data[var].append(np.abs(res[var][ii]))
            data[var].append(res[var][ii])

        data['ne_loc'].append(ne_loc)
        data['ne_loc_unc'].append(ne_loc_unc)
        data['ne_sep'].append(ne_sep)
        data['ne_sep_unc'].append(ne_sep_unc)
        data['ne_sol'].append(ne_sol)
        data['ne_sol_unc'].append(ne_sol_unc)
        data['Te_sol'].append(Te_sol)
        data['Te_sol_unc'].append(Te_sol_unc)

        
        ###########################################
        # Ionization rate profiles
        ###########################################
        SSS = aurora.interp_atom_prof(
            atom_data['scd'], np.log10(res['ne_prof'][ii][mask]), np.log10(res['Te_prof'][ii][mask]),
            log_val=False, x_multiply=False)[:,0]   # ionization rate in units of cm^3/s if x_multiply=False
        
        ioniz = res['ne_prof'][ii][mask]*res['nn_prof'][ii][mask]*SSS
        ioniz_unc = np.sqrt(res['nn_prof'][ii][mask]**2 * res['ne_prof_unc'][ii][mask]**2+
                            res['nn_prof_unc'][ii][mask]**2 * res['ne_prof'][ii][mask]**2)*SSS   
            
        data['Ln_ped_exp_mm'].append(res['Ln_ped_exp_mm'][ii] if res['Ln_ped_exp_mm'][ii]>0 else 10) # arbitrary
        data['Ln_ped_exp_mm_unc'].append(res['Ln_ped_exp_mm_unc'][ii] if res['Ln_ped_exp_mm_unc'][ii]>0 else 100)
        data['Ln_ped_mm'].append(res['Ln_ped_mm'][ii] if res['Ln_ped_mm'][ii]>0 else 10)
        data['Ln_ped_mm_unc'].append(res['Ln_ped_mm_unc'][ii] if res['Ln_ped_mm_unc'][ii]>0 else 100)

        #########################
        # total input power
        RF_efficiency=0.8
        data['P_tot'].append(RF_efficiency * res['P_RF'][ii] + res['P_oh'][ii])   # MW

        # power through LCFS
        # (NB: strictly speaking we should also consider dW/dt, but this must be small in steady-state windows)
        P_sol = np.clip(data['P_tot'][-1] - res['P_rad'][ii],1e-10,None)    # ensure P_sol>=0
        data['P_sol'].append(P_sol)  

        data['area_LCFS'].append(res['area_LCFS'][ii])
        data['P_sol/A'].append(data['P_sol'][-1]/(res['area_LCFS'][ii])) 
        data['f_gw'].append(res['f_gw'][ii])
        data['gas_rate'].append(res['gas_rate'][ii])
        data['gas_fuel'].append(res['gas_fuel'][ii] if res['gas_fuel'][ii]>0 else 1e-10) # -ve or 0 values break regression
        data['gas_cum'].append(res['gas_cum'][ii] if res['gas_cum'][ii]>0 else 1e-10)

        # geometry/field configuration
        data['gradB_drift_up'].append(1 if res['gradB_drift_up'][ii] else -1)
        data['favorable'].append(float(res['favorable'][ii])) # +1 for USN, 0 for DB, -1 for LSN --- ssep gives continuous scale

        # formulae from Hutchinson, PSFC RR 1995
        data['l_e'].append(1.45e-4* Te_sol**2/(ne_sol/1e14))  # m
        data['nu_star'].append(np.pi*R_sol*data['q95'][-1]/(data['l_e'][-1]))     # all in [m] 
        data['l_cx_cold'].append( 2.4e-2*(Te_sol/2.0)**(-0.33)*(ne_sol/1e14)**(-1.))   # m-->mm
        data['l_cx_hot'].append(1.4e-2 * (Te_sol/2.0)**(0.17)*(ne_sol/1e14)**(-1.))  #m-->mm

        # Also store entire radial profiles:
        data['nn_prof'].append(res['nn_prof'][ii])
        data['nn_prof_unc'].append(res['nn_prof_unc'][ii])

        # calculate \int_{r=0}^{LCFS} ioniz dR
        # NB: ioniz is in units of cm^-3 s^-1 --> multiply by 1e6 to change to m^-3 s^-1
        iLCFS = np.nanargmin(np.abs(res['R_prof'][ii] - R_lcfs))
        Gamma_D_prof = scipy.integrate.cumtrapz(ioniz[:iLCFS]*1e6, res['R_prof'][ii][mask][:iLCFS], initial=0.0)
        data['Gamma_D'].append( Gamma_D_prof[-1] )  # particles/(m^2 s^1)
        Gamma_D_prof_unc = scipy.integrate.cumtrapz(ioniz_unc[:iLCFS]*1e6, res['R_prof'][ii][mask][:iLCFS], initial=0.0)
        data['Gamma_D_unc'].append( Gamma_D_prof_unc[-1] )

        _ioniz_out = ioniz[iLCFS:]*1e6
        _ioniz_out_unc = ioniz_unc[iLCFS:]*1e6
        _R_out = res['R_prof'][ii][mask][iLCFS:]
        data['ioniz_out'].append( scipy.integrate.simps(_ioniz_out[~np.isnan(_ioniz_out)],
                                                        _R_out[~np.isnan(_ioniz_out)]) )
        data['ioniz_out_unc'].append( scipy.integrate.simps(_ioniz_out_unc[~np.isnan(_ioniz_out)],
                                                            _R_out[~np.isnan(_ioniz_out)]) )

        # 2-point model results using Brunner lambda_q scaling - repeat from above
        data['p_Pa_avg'].append( res['p_Pa_avg'][ii] )
        model = two_point_model(0.69, 0.22, data['P_sol'][-1], data['Bp'][-1],
                                data['Bt'][-1], data['q95'][-1], data['p_Pa_avg'][-1],
                                ne_sep*1e6) # ne-sep in m^-3  # ne_sep not actually used unless we ask for ne-target

        data['Te_sep'].append(model.Tu_eV if model.Tu_eV>40.0 else 60.0)   # assume 60 eV if calculation failed)
        data['lam_q'].append(model.lam_q_mm_brunner) # mm
        data['q_par'].append(model.q_par_MW_m2)  # MW/m^2

        # arbitrary uncertainty on Te_sep from 2-point model -- statistical unc cannot represent model inadequacy
        data['Te_sep_unc'].append(20.) # 20 eV

        # Call it detached if pressure on divertor is less than 0.2 times the separatrix pressure
        data['detachment'].append(res['ne_fmp_cm3'][ii]*res['Te_fmp_eV'][ii]/(ne_sep*model.Tu_eV))  # ne_sep already in cm^-3
        #data['detached'].append(1 if det_metric<0.2 else 0)

        data['CRYO_ON'].append(res['CRYO_ON'][ii])

        # calculate profiles of Deff and Veff
        _ne_m3_grads = np.abs(np.gradient(res['ne_prof'][ii][mask]*1e6, res['R_prof'][ii][mask])[:iLCFS])   + 1e-33  # add small number to avoid 0 division
        _ne_m3_vals = (res['ne_prof'][ii][mask]*1e6)[:iLCFS]    + 1e-33  # add small number to avoid 0 division
        _Te_eV_grads = np.abs(np.gradient(res['Te_prof'][ii][mask], res['R_prof'][ii][mask])[:iLCFS])    + 1e-33  # add small number to avoid 0 division
        _Te_eV_vals = (res['Te_prof'][ii][mask])[:iLCFS]    + 1e-33  # add small number to avoid 0 division
        
        Deff_prof = Gamma_D_prof/_ne_m3_grads
        Veff_prof = Gamma_D_prof/_ne_m3_vals

        # take 20% uncertainty in density, 30% in density gradient
        Deff_prof_unc = np.sqrt((1./_ne_m3_grads)**2 * Gamma_D_prof_unc**2 + (Gamma_D_prof/_ne_m3_grads**2)**2 * (_ne_m3_grads*0.3)**2)
        Veff_prof_unc = np.sqrt((1./_ne_m3_vals)**2 * Gamma_D_prof_unc**2 + (Gamma_D_prof/_ne_m3_vals**2)**2 * (_ne_m3_vals*0.2)**2)

        Lne_prof = _ne_m3_vals/_ne_m3_grads
        LTe_prof = _Te_eV_vals/_Te_eV_grads
        
        # store only Deff and Veff at rhop=0.99 for the moment (TODO: look at profiles)
        data['Deff'].append( interp1d(rhop_vec[mask][:iLCFS], Deff_prof, bounds_error=False, fill_value=fill_value)(0.99) )  # m^2/s
        data['Veff'].append( interp1d(rhop_vec[mask][:iLCFS], Deff_prof, bounds_error=False, fill_value=fill_value)(0.99) )  # m/s
        
        data['Deff_unc'].append( interp1d(rhop_vec[mask][:iLCFS], Deff_prof_unc, bounds_error=False, fill_value=fill_value)(0.99) )  # m^2/s
        data['Veff_unc'].append( interp1d(rhop_vec[mask][:iLCFS], Deff_prof_unc, bounds_error=False, fill_value=fill_value)(0.99) )  # m/s
            
        data['Lne'].append( interp1d(rhop_vec[mask][:iLCFS], Lne_prof, bounds_error=False, fill_value=fill_value)(0.99) *1e3) # mm
        data['LTe'].append( interp1d(rhop_vec[mask][:iLCFS], LTe_prof, bounds_error=False, fill_value=fill_value)(0.99) *1e3) # mm

        # arbitrarily set uncertainties on gradient scale lengths to be 50%
        data['Lne_unc'].append(data['Lne'][-1]*0.5)
        data['LTe_unc'].append(data['LTe'][-1]*0.5)
        
        '''
        # quantities related to ne and Te gradients
        # taking just max gradient seems to be very noisy...
        xx = (res['R'][ii][mask]-res['R'][ii][mask][-1])*100  # cm, same as 
        popt, pcov = scipy.optimize.curve_fit(lambda xx,a,b: a+b*xx,
                                              xx,
                                              res['ne_prof'][ii][mask]/1e14,
                                              sigma = res['ne_prof_unc'][ii][mask]/1e14,
                                              p0=(1., -1.))
        if popt[1]==1:
            embed()
            
        #bounds = [(1,-1000), (1000,0)])  # mins, maxs -- helps avoiding overflow
        data['grad_ne'].append( np.abs(popt[1])*1e14 *1e8)  # cm^-4 --> m^-4
        #data['grad_ne_unc'].append( np.sqrt(pcov[1,1])*1e14 )  #  cm^-4
        data['grad_ne_unc'].append( 0.2 * data['grad_ne'][-1])  # arbitrary 20% uncertainty, more reasonable estimate
        
        popt, pcov = scipy.optimize.curve_fit(lambda xx,a,b: a+b*xx,
                                              xx,
                                              res['Te_prof'][ii][mask_ped],
                                              sigma = res['Te_prof_unc'][ii][mask_ped],
                                              p0=(200., -100.))

        #bounds = [(1,-1000), (1000,0)])  # mins, maxs -- helps avoiding overflow
        data['grad_Te'].append( np.abs(popt[1]) *1e2 ) # eV/cm --> eV/m
        #data['grad_Te_unc'].append( np.sqrt(pcov[1,1]) *1e2) # eV/cm --> eV/m
        data['grad_Te_unc'].append( 0.2 * data['grad_Te'][-1])  # arbitrary 20% uncertainty, more reasonable estimate
        
        # gradient scale lengths
        data['Lne'].append((data['ne_sep'][-1]*1e6)/data['grad_ne'][-1] *1e3)  # m -->mm
        data['LTe'].append(data['Te_sep'][-1]/data['grad_Te'][-1]*1e3)  # m -->mm
        '''

    
    for key in data:
        data[key] = np.array(data[key])

    # temporary: store results
    #with open('lya_db_fs.pkl', 'wb') as f:
    #    pkl.dump(res,f)

    # mask out some problematic conditions
    mmask1 = np.logical_or(data['p_E_BOT_MKS']<0, data['p_B_BOT_MKS']<0)
    mmask2 = np.logical_or(data['p_D2']<0, data['p_F_CRYO_MKS']<0)
    mmask = np.logical_or(mmask1, mmask2)
    for key in data:
        data[key] = np.array(data[key])[~mmask]
        
    # Ignore points with nn_loc greater than 1e11 cm^-3 -- they seem to be scattered and have no trends
    mask_nn = data['nn_loc']>1e-3*1e14  # only a few points
    for key in data:
        data[key] = np.array(data[key])[~mask_nn]

    # eliminate cases where no gas at all appears to have been injected
    mask_gas = data['gas_fuel']<1e-3  # only a few points
    for key in data:
        data[key] = np.array(data[key])[~mask_gas]

        
    # eliminate IWL plasmas for the moment, since the localization of the LCFS appears inaccurate
    mask_IWL = np.isnan(data['ssep'])
    for key in data:
        data[key] = np.array(data[key])[~mask_IWL]
        

    # consider only shots where the amount of injected gas is clear -- say, above 2 Torr-l
    # above 2 Torr-l, there is an approximately linear relation between gas_fuel and p_D2
    mask_gas = data['gas_fuel']<2.  # only a few points
    for key in data:
        data[key] = np.array(data[key])[~mask_gas]

    ##############################################

    plot_vars = ['Bt','Bp','Ip','ne_sep','P_sol/A','P_rad','q95','Wmhd','p_D2'] #,'Wmhd','P_oh']

    fig0,axs0 = plt.subplots(3,3, figsize=(15,8), num=555)
    ax0 = axs0.flatten()
    fig1,axs1 = plt.subplots(3,3, figsize=(15,8), num=666)
    ax1 = axs1.flatten()
    fig2,axs2 = plt.subplots(3,3, figsize=(15,8), num=777)
    ax2 = axs2.flatten()
    axs = [ax0,ax1,ax2]

    c_reg = {'ohmic':'g','lmode':'b','hmode':'r','imode':'k'}

    for regime, col in c_reg.items():
        mask = data['regime']==regime
        
        # plot nn at rhop=rhop_loc
        for ii,key in enumerate(plot_vars):
            ax0[ii].errorbar(data[key][mask], data['nn_loc'][mask], data['nn_loc_unc'][mask], alpha=0.5,
                             fmt='.', c=col)

        # plot nn/ne at rhop=rhop_loc
        for ii,key in enumerate(plot_vars):
            ax1[ii].errorbar(data[key][mask], data['nn_by_ne_loc'][mask], data['nn_by_ne_loc_unc'][mask],
                             fmt='.',c=col, alpha=0.5)
            ax1[ii].set_ylim([None,1e-1])

        # plot L0 at rhop=rhop_loc
        for ii,key in enumerate(plot_vars):
            ax2[ii].errorbar(data[key][mask], data['Ln_ped_mm'][mask], data['Ln_ped_mm_unc'][mask],
                             fmt='.', c=col, alpha=0.5)
            ax2[ii].set_ylim([None,100])

    fig0.suptitle(fr'$N_1$ [$cm^{{-3}}$] at $\rho_p={rhop_loc}$', fontsize=20)
    fig1.suptitle(fr'$N_1/n_e$ at $\rho_p={rhop_loc}$', fontsize=20)
    fig2.suptitle(fr'$L_0$ [mm] ($\rho_p={rhop_loc}$)', fontsize=20)

    for xx in np.arange(len(axs[:3])):
        axs[xx][0].set_xlabel(r'$B_T$ [T]')
        axs[xx][1].set_xlabel(r'$B_p$ [T]')
        axs[xx][2].set_xlabel(r'$I_p$ [A]')
        axs[xx][3].set_xlabel(r'$n_{e,sep}$ [$cm^{-3}$]')
        axs[xx][4].set_xlabel(r'$P_{sol}/A$ [MW]') # axs[xx][4].set_xlabel(r'$P_{RF}$ [MW]')
        axs[xx][5].set_xlabel(r'$P_{rad}$ [MW]') 
        axs[xx][6].set_xlabel(r'$q_{95}$')
        axs[xx][7].set_xlabel(r'$W_{MHD}$ [J]')
        axs[xx][8].set_xlabel(r'$p_{D2}$ [mTorr]')
        
    fig0.tight_layout(); fig0.subplots_adjust(top=0.88)

    for ii in np.arange(len(ax0)):
        #axs[xx][ii].set_ylim([1e-5,1e-2])
        axs[xx][ii].set_yscale('log')
        
    for ii in np.arange(len(ax1)):
        ax1[ii].set_ylim([1e-5,1e-2])
        ax1[ii].set_yscale('log')

    fig1.tight_layout(); fig1.subplots_adjust(top=0.88)

    for ii in np.arange(len(ax2)):
        ax2[ii].set_ylim([2.,1e2])
        #ax2[ii].set_yscale('log')
        
    fig2.tight_layout(); fig2.subplots_adjust(top=0.88)


    ##### Physical length scales and nu_star ####
    fig3,ax3 = plt.subplots(3,4, figsize=(15,8), num=888, sharex='col')

    for regime,col in c_reg.items():

        mask = data['regime']==regime
        
        # plot nn, nn/ne and Ln0 at rhop=rhop_loc vs. Hutchinson variables
        for ii, key1 in enumerate(['l_e','nu_star','l_cx_cold','l_cx_hot']):
            ax3[0,ii].errorbar(data[key1][mask], data['nn_loc'][mask], data['nn_loc_unc'][mask],
                               fmt='.', c=col, alpha=0.5)
            ax3[1,ii].errorbar(data[key1][mask], data['nn_by_ne_loc'][mask], data['nn_by_ne_loc_unc'][mask],
                               fmt='.', c=col, alpha=0.5)
            ax3[2,ii].errorbar(data[key1][mask], data['Ln_ped_exp_mm'][mask], data['Ln_ped_exp_mm_unc'][mask],
                               fmt='.', c=col, alpha=0.5)

            [ax3[jj,ii].grid('on') for jj in np.arange(ax3.shape[0])]
            
    ax3[0,0].set_ylabel(r'$n_n$ [$cm^{-3}$]')
    ax3[1,0].set_ylabel(r'$n_n/n_e$')
    ax3[2,0].set_ylabel(r'$L_{n_n}$ [$mm$]')
    for ii in np.arange(ax3.shape[0]):
        for jj in np.arange(ax3.shape[1]):
            ax3[ii,jj].set_yscale('log')

    [ax3[0,ii].set_xscale('log') for ii in np.arange(ax3.shape[1])]

    ax3[-1,0].set_xlabel(r'$l_e$ [m]')
    ax3[-1,1].set_xlabel(r'$\nu_*$')
    ax3[-1,2].set_xlabel(r'$l_{cx}$ (cold)')
    ax3[-1,3].set_xlabel(r'$l_{cx}$ (hot)')

    fig3.tight_layout()

    ######


    #### Dummy figure for legend
    figg = plt.figure()
    for regime,col in c_reg.items():
       plt.errorbar([],[],[],c=col, label=name_map[regime])
    plt.legend(loc='best',fontsize=30,framealpha=0.0).set_draggable(True)
    plt.gca().axis('off')



def regressions():

    ############################################
    #
    #             regressions
    #
    ############################################

    import regression_tools as rt

    c_reg = {'ohmic':'g','lmode':'b','hmode':'r','imode':'k'}

    # Test regressions
    #mask = data['regime']=='hmode'
    mask = np.ones_like(data['regime'], dtype=bool) # to select all
    mask_fav = data['favorable']==1
    mask_IWL = np.isnan(data['ssep'])
    mask_LSN = data['ssep']<-0.1   # only LSN
    mask_USN = data['ssep']>0.1     # only USN
    mask_DN = np.logical_and(-0.1<data['ssep'], data['ssep']<0.1)   # select double nulls
    mask_cryo = data['CRYO_ON']==True #False
    mask_cleanLn = data['Ln_ped_exp_mm']<50.0
    mask_h = np.logical_or(data['regime']=='EDA',data['regime']=='ELMy') # to select all H-modes
    mask_l = np.logical_or(data['regime']=='ohmic', data['regime']=='lmode') # to select all L-modes
    mask_ohmic = data['regime'] == 'ohmic'
    mask_rfl = data['regime'] =='lmode'


    #rd = rt.subselect_database(data, mask)

    #color_var = 'favorable' #'cryo_on' #'favorable' #'gradB_drift_up' # 

    '''

    # Demonstration that C-Mod is in a main-chamber recycling regime: midplane pressure is not closely related to divertor pressure
    #rt.run_regression(['p_D2'], rd, y=rd['p_F_CRYO_MKS'], mask=rd['favorable']>0,
    #                   y_unc = np.zeros_like(rd['p_F_CRYO_MKS'])+0.2, color_var='cryo_on',
    #                   y_label='p_{D2,F-cryo}', rhop_loc=rhop_loc, plot_cov=False)

    rt.run_regression(['p_F_CRYO_MKS'], rd, y=rd['p_D2'], mask=rd['favorable']>0,
                       y_unc = np.zeros_like(rd['p_D2'])+0.05, color_var='cryo_on',
                       y_label='p_{D2,mid}', rhop_loc=rhop_loc, plot_cov=False)

    #rt.run_regression(['p_D2'], rd, y=rd['p_F_CRYO_MKS'], mask=rd['favorable']<0,
    #                   y_unc = np.zeros_like(rd['p_F_CRYO_MKS'])+0.2, color_var='cryo_on',
    #                   y_label='p_{D2,F-cryo}', rhop_loc=rhop_loc, plot_cov=False)

    rt.run_regression(['p_F_CRYO_MKS'], rd, y=rd['p_D2'], mask=rd['favorable']<0,
                       y_unc = np.zeros_like(rd['p_D2'])+0.05, color_var='cryo_on',
                       y_label='p_{D2,mid}', rhop_loc=rhop_loc, plot_cov=False)


    #rt.run_regression(['p_D2'], rd, y=rd['p_B_BOT_MKS'], mask=rd['favorable']>0,
    #                   y_unc = np.zeros_like(rd['p_B_BOT_MKS'])+0.2, color_var='cryo_on',
    #                   y_label='p_{D2,F-cryo}', rhop_loc=rhop_loc, plot_cov=False)

    rt.run_regression(['p_B_BOT_MKS'], rd, y=rd['p_D2'], mask=rd['favorable']>0,
                       y_unc = np.zeros_like(rd['p_D2'])+0.05,  color_var='cryo_on',
                       y_label='p_{D2,mid}', rhop_loc=rhop_loc, plot_cov=False)


    #rt.run_regression(['p_D2'], rd, y=rd['p_B_BOT_MKS'], mask=rd['favorable']<0,
    #                   y_unc = np.zeros_like(rd['p_B_BOT_MKS'])+0.2, color_var='cryo_on',
    #                   y_label='p_{D2,F-cryo}', rhop_loc=rhop_loc, plot_cov=False)

    rt.run_regression(['p_B_BOT_MKS'], rd, y=rd['p_D2'], mask=rd['favorable']<0,
                       y_unc = np.zeros_like(rd['p_D2'])+0.05, color_var='cryo_on',
                       y_label='p_{D2,mid}', rhop_loc=rhop_loc, plot_cov=False)


    '''


    # use constant uncertainty, not a function of pressure magnitude!
    #rt.run_regression(['p_B_BOT_MKS'], rd, y=rd['p_E_BOT_MKS'], #y_unc = rd['p_E_BOT_MKS']*0.01, #0.5,
    #               y_unc = np.ones_like(rd['p_E_BOT_MKS'])*0.0001,
    #               color_var='favorable',
    #               y_label='p_{D2,divE}', rhop_loc=rhop_loc, plot_cov=False)



    #### Regressions with physical params'
    #vars_regr = ['p_D2', 'l_e','l_cx_cold','l_cx_hot','nu_star']

    #rt.run_regression(vars_regr, rd, y=rd['nn_loc'], y_unc = rd['nn_loc_unc'], color_var='favorable',
    #                   y_label='n_n', rhop_loc=rhop_loc, plot_cov=False)
    #rt.run_regression(vars_regr, rd, y=rd['Gamma_D'], y_unc = rd['Gamma_D_unc'], y_label='S_{{in}}', color_var='favorable',
    #                   rhop_loc=rhop_loc, plot_cov=False)




    #############
    mod_sel_nnsep=False
    mod_sel_nesep=False
    mod_sel_Ln=False
    mod_sel_Deff=False
    mod_sel_Gammae=False
    mod_sel_Veff=False

    # select all cases
    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))

    ############
    if mod_sel_nnsep:
        # For n_n
        #list_vars = ['Bt','p_D2','ne_sep','q95','p_F_CRYO_MKS','p_E_BOT_MKS','Ip','lam_q','Lne','LTe'] #'p_B_B
        list_vars = ['Bt','gas_fuel','p_D2','Te_sep','p_F_CRYO_MKS','p_E_BOT_MKS','Ip','lam_q'] #'p_B_BO
        regr_var_name='nn_loc'
        var_label = 'n_{n,sep}'
        
        out_nn = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name )
        combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_nn

    ##############
    if mod_sel_nesep:
        # for ne_sep
        #list_vars = ['p_D2','Ip','Wmhd','gas_fuel','gas_cum','P_sol/A','f_gw','q95','Bp','p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        #list_vars = ['nn_loc','Bt','p_D2','Ip','Wmhd','gas_fuel','P_sol/A','q95'] #,'p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        
        # exclude nn from ne_sep regression
        list_vars = ['nebar','Bt','p_D2','Ip','Wmhd','gas_fuel','P_sol/A','q95'] #,'p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        
        regr_var_name='ne_sep'
        var_label='n_{e,sep}'
        
        out_nesep = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name)
        combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_nesep
        

    ############
    if mod_sel_Ln:
        # for Ln
        method='exp_' #''
        regr_var_name='Ln_ped_'+method+'mm'
        var_label='L_{n_n}'
        
        # check all possible combinations of the following variables:
        #list_vars = ['p_D2','Ip','Wmhd','gas_fuel','gas_cum','P_sol/A','f_gw','q95','Bp', 'p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        #list_vars = ['Te_sep','lam_q','Ip','Wmhd','gas_cum','gas_fuel','P_sol/A','q95'] #'p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        #list_vars = ['Bt','Bp','lam_q','ne_sep','Te_sep','Wmhd','P_sol/A','q95', 'Lne','LTe']
        #list_vars = ['Bt','lam_q','ne_sep','Te_sep','Wmhd','q95', 'Lne','LTe']  # 'P_sol/A'
        list_vars = ['rhostar','Te_sep','Bt','ne_sep','Wmhd','q95', 'Lne','P_sol/A' ]  # '
        
        out_Ln = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name, max_num_combos=2)
        combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_Ln

    ############
    if mod_sel_Gammae:
        # for Gamma_D
        regr_var_name='Gamma_D'
        var_label='\\Gamma_D'
        
        # check all possible combinations of the following variables:
        #list_vars = ['p_D2','Ip','Wmhd','gas_fuel','gas_cum','P_sol/A','f_gw','q95','Bp', 'p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        #list_vars = ['Te_sep','lam_q','Ip','Wmhd','gas_cum','gas_fuel','P_sol/A','q95'] #'p_F_CRYO_MKS','p_E_BOT_MKS'] #,'p_B_BOT_MKS']
        #list_vars = ['Bt','Bp','lam_q','ne_sep','Te_sep','Wmhd','P_sol/A','q95', 'Lne','LTe']
        #list_vars = ['p_D2','ne_sep','Te_sep', 'Bt','lam_q','Wmhd','q95'] #'P_sol/A',
        list_vars = ['p_D2','Te_sep', 'Bt','lam_q','Wmhd','q95','P_sol/A']  # predictable parameters
        
        out_Gammae = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name,max_num_combos=2)
        combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_Gammae


    ############
    if mod_sel_Deff:
        # for Deff
        regr_var_name='Deff'
        var_label='D_{D,eff}'
        
        # check all possible combinations of the following variables:
        #list_vars = ['p_D2','Ip','Wmhd','gas_fuel','gas_cum','P_sol/A','f_gw','q95','Bp', 'p_F_CRYO_MKS','p_E_BOT_MKS'] 
        #list_vars = ['Te_sep','lam_q','Ip','Wmhd','gas_cum','gas_fuel','P_sol/A','q95'] #'p_F_CRYO_MKS','p_E_BOT_MKS'] 
        #list_vars = ['Bt','Bp','lam_q','ne_sep','Te_sep','Wmhd','P_sol/A','q95', 'Lne','LTe']
        #list_vars = ['rhostar','p_D2','Bt','lam_q','ne_sep','Te_sep','Wmhd','q95', 'Lne'] #'P_sol/A',
        list_vars = ['rhostar','Bt','lam_q','Te_sep','Wmhd','q95','P_sol/A']
        
        out_Deff = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name,max_num_combos=2)
        combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_Deff
        

    ######
    if mod_sel_Veff:
        # for Deff
        regr_var_name='Veff'
        var_label='V_{e,eff}'
        
        # check all possible combinations of the following variables:
        #list_vars = ['p_D2','Ip','Wmhd','gas_fuel','gas_cum','P_sol/A','f_gw','q95','Bp', 'p_F_CRYO_MKS','p_E_BOT_MKS']
        #list_vars = ['Te_sep','lam_q','Ip','Wmhd','gas_cum','gas_fuel','P_sol/A','q95'] #'p_F_CRYO_MKS','p_E_BOT_MKS'] 
        #list_vars = ['Bt','Bp','lam_q','ne_sep','Te_sep','Wmhd','P_sol/A','q95', 'Lne','LTe']
        list_vars = ['rhostar','p_D2','Bt','lam_q','gas_fuel','Te_sep','Wmhd','q95', 'Lne'] #'P_sol/A',
        
        out_Veff = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name,max_num_combos=2)
        combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_Veff
        



    ####### Histograms

    import seaborn as sns
    nbins=40

    fig,ax = plt.subplots(1,4, figsize=(14,4))
    sns.histplot(data=rd['Ip'],  kde=False, ax=ax[0], bins=nbins)
    ax[0].set_xlabel(r'$I_p$ [MA]')
    #ax[0].set_yscale('log')
    sns.histplot(data=rd['q95'],  kde=False, ax=ax[1], bins=nbins)
    ax[1].set_xlabel(r'$q_{95}$')
    #ax[1].set_yscale('log')
    #sns.histplot(data=rd['Bt'],  kde=False, ax=ax[1], bins=nbins)
    #ax[1].set_xlabel(r'$B_{T}$ [T]')
    #ax[1].set_yscale('log')
    ax[1].set_ylabel('')
    sns.histplot(data=rd['p_D2'],  kde=False, ax=ax[2], bins=nbins)
    ax[2].set_xlabel(r'$p_{D2,mid}$ [mTorr]')
    ax[2].set_ylabel('')
    sns.histplot(data=rd['P_sol/A'],  kde=False, ax=ax[3], bins=nbins)
    ax[3].set_xlabel(r'$P_{sol}/A$ [$MW/m^2$]')
    ax[3].set_ylabel('')

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()


    ### Another set of histograms 

    # mask out cases with excessively high pe_div/pe_LCFS
    mask_det = rd['detachment']<9.9
    fig,ax = plt.subplots(1,4, figsize=(14,4))
    sns.histplot(data=rd['f_gw'],  kde=False, ax=ax[0], bins=nbins)
    ax[0].set_xlabel(r'$f_{GW}=\langle n_e \rangle/(I_p/(\pi a^2))$')
    sns.histplot(data=rd['ssep'],  kde=False, ax=ax[1], bins=nbins)
    ax[1].set_xlabel(r'$r_{x1}$ - $r_{x2}$ [cm]')
    ax[1].set_ylabel('')
    sns.histplot(data=rd['lam_q'],  kde=False, ax=ax[2], bins=nbins)
    ax[2].set_xlabel(r'$\lambda_q$ [mm]')
    ax[2].set_ylabel('')
    sns.histplot(data=rd['Te_sep'],  kde=False, ax=ax[3], bins=nbins)
    ax[3].set_xlabel(r'$T_{e,sep}$ [eV]')
    ax[3].set_ylabel('')

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()



    #######
    # Simple plots (no regression)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    color_data = rd['Ip'] #rd['l_cx_cold'] #rd['nu_star'] #rd['rhostar'] # rd['Ip']
    color_label = r'$I_p$ [MA]' #r'$l_{cx}$' #r'$\nu_\ast$' #r'$\sqrt{T_e}/B_t$ [$eV^{1/2}/T$] $\sim \rho_*$' #r'$I_p$ [MA]'
    cmap = cm.viridis 
    norm_col = Normalize(vmin=np.nanmin(color_data), vmax=np.nanmax(color_data))
    cols = cmap(norm_col(color_data))


    # Plot of Deff vs. Lne
    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    fig, ax = plt.subplots()
    ax.errorbar(rd['Lne'], rd['Deff'], rd['Deff_unc'], xerr=rd['Lne_unc'],
                fmt='.', ecolor=cols, ms=0.0, alpha=0.5)
    patch = ax.scatter(rd['Lne'], rd['Deff'], alpha=1.0, marker='.',
                       c=color_data, edgecolors=None, cmap=cmap, norm=norm_col)
    plt.colorbar(patch, ax=ax, label=color_label)
    ax.set_xlabel(r'$L_{n_e}$ [$mm$]')
    ax.set_ylabel(r'$D_{D,eff}$ [$m^2/s$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()

    # Plot of Deff vs. Lne for L and H modes
    fig, ax = plt.subplots()
    rd = rt.subselect_database(data, mask_l)
    patch = ax.scatter(rd['Lne'], rd['Deff'], alpha=0.5, marker='o', c='k', edgecolors=None)
    rd = rt.subselect_database(data, np.logical_and(mask_h, data['Lne']<1e2))
    patch = ax.scatter(rd['Lne'], rd['Deff'], alpha=0.5, marker='o', c='r', edgecolors=None)
    ax.set_xlabel(r'$L_{n_e}$ [$mm$]')
    ax.set_ylabel(r'$D_{D,eff}$ [$m^2/s$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()

    # Plot of Veff vs. ne_sep
    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    fig, ax = plt.subplots()
    ax.errorbar(rd['ne_sep'], rd['Veff'], rd['Veff_unc'], xerr=rd['ne_sep_unc'],
                fmt='.', ecolor=cols, ms=0.0, alpha=0.5)
    patch = ax.scatter(rd['ne_sep'], rd['Veff'], alpha=1.0, marker='o',
                       c=color_data, edgecolors=None, cmap=cmap, norm=norm_col)
    plt.colorbar(patch, ax=ax, label=color_label)
    ax.set_xlabel(r'$n_{e,sep}$ [$10^{14}$ $cm^{-3}$]')
    ax.set_ylabel(r'$v_{D,eff}$ [$m/s$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()

    # Plot of p_D2 vs. gas_fuel for L and H modes -- for finite Sgas, some ~linear prop is seen
    fig, ax = plt.subplots()
    rd = rt.subselect_database(data,  ~np.logical_or(data['regime']=='ELMy',data['regime']=='EDA'))
    patch = ax.scatter(rd['gas_fuel'], rd['p_D2'], alpha=0.5, marker='o', c='k', edgecolors=None)
    rd = rt.subselect_database(data,  np.logical_or(data['regime']=='ELMy',data['regime']=='EDA'))
    patch = ax.scatter(rd['gas_fuel'], rd['p_D2'], alpha=0.5, marker='o', c='r', edgecolors=None)
    ax.plot([],[], 'ko', label='L-mode')
    ax.plot([],[], 'ro', label='H-mode')
    ax.legend(loc='best').set_draggable(True)
    ax.set_xlabel(r'$S_{gas}$ [Torr-l]')
    ax.set_ylabel(r'$p_{D2,mid}$ [mTorr]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()


    # Plot of Veff vs. ne_sep for L and H modes
    fig, ax = plt.subplots()
    rd = rt.subselect_database(data, ~np.logical_or(data['regime']=='ELMy',data['regime']=='EDA'))
    patch = ax.scatter(rd['ne_sep'], rd['Veff'], alpha=0.5, marker='o', c='k', edgecolors=None)
    rd = rt.subselect_database(data, np.logical_or(data['regime']=='ELMy',data['regime']=='EDA'))
    patch = ax.scatter(rd['ne_sep'], rd['Veff'], alpha=0.5, marker='o', c='r', edgecolors=None)
    ax.set_xlabel(r'$n_{e,sep}$ [$10^{14}$ $cm^{-3}$]')
    ax.set_ylabel(r'$v_{D,eff}$ [$m/s$]')
    ax.plot([],[], 'ko', label='L-mode')
    ax.plot([],[], 'ro', label='H-mode')
    ax.legend(loc='best').set_draggable(True)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()

    # Plot of Veff vs. rhostar
    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    fig, ax = plt.subplots()
    ax.errorbar(rd['l_cx_hot_sep'], rd['Veff'], rd['Veff_unc'],# xerr=rd['ne_sep_unc'],
                fmt='.', ecolor=cols, ms=0.0, alpha=0.5)
    patch = ax.scatter(rd['l_cx_hot_sep'], rd['Veff'], alpha=1.0, marker='o',
                       c=color_data, edgecolors=None, cmap=cmap, norm=norm_col)
    plt.colorbar(patch, ax=ax, label=color_label)
    ax.set_xlabel(r'$l_{cx}$ [$mm$]')
    ax.set_ylabel(r'$v_{D,eff}$ [$m/s$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()


    # Plot of Gammae vs. p_D2 for fav and unfav directions
    fig, ax = plt.subplots()
    rd = rt.subselect_database(data, mask_USN)
    patch = ax.scatter(rd['p_D2'], rd['Gamma_D'], alpha=0.5, marker='o', c='k', edgecolors=None)
    rd = rt.subselect_database(data, mask_LSN)
    patch = ax.scatter(rd['p_D2'], rd['Gamma_D'], alpha=0.5, marker='o', c='r', edgecolors=None)
    ax.set_xlabel(r'$p_{D2,mid}$ [mTorr]')
    ax.set_ylabel(r'$\Gamma_{e}$ [$10^{20}$ $m^{-2} s^{-1}$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid('on')
    plt.tight_layout()


    ########

    # regime_choices = [['L-mode',mask_l],['H-mode',mask_h]]
    # for case in regime_choices:
    #     regime_choice = case[0]
    #     mask_choice = case[1]
    #     print(f'Regression on Lnn for {regime_choice}')
    #     rd = rt.subselect_database(data, mask_choice)

    #     regr_var_name='Ln_ped_exp_mm'
    #     var_label='L_{n_n}'

    #     #list_vars = ['Ip','rhostar','Te_sep','Bt','ne_sep','Wmhd','q95', 'Lne','P_sol/A' ]  # '
    #     list_vars = ['p_E_BOT_MKS','p_F_CRYO_MKS', 'Bt', 'rhostar','Te_sep','ne_sep','q95', 'Lne' ]  
        
    #     out_Ln = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name, max_num_combos=2)
    #     combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_Ln

        
    #     rt.run_regression(aic_comb_f,
    #                       rd, rd['Ln_ped_exp_mm'], 1e-3*np.ones_like(rd['Ln_ped_exp_mm_unc']),
    #                       color_var='favorable', y_label='L_{n_n}', plot_cov=False, plot=True)


    # regime_choices = [['L-mode',mask_l],['H-mode',mask_h], ['Ohmic L-mode', mask_ohmic], ['RF-heated L-mode',mask_rfl]]    
    # for case in regime_choices:
    #     regime_choice = case[0]
    #     mask_choice = case[1]
    #     print(f'Regression on Veff for {regime_choice}')
    #     rd = rt.subselect_database(data, mask_choice)

    #     regr_var_name='Veff'
    #     var_label='v_{D,eff}'


    #     #rt.run_regression(['ne_sep','p_D2'],
    #     #                  rd, rd['Veff'], rd['Veff_unc'],
    #     #                  color_var='favorable', y_label=r'v_{D,eff}', plot_cov=False, plot=True)
    #     #plt.gcf().suptitle(f'{regime_choice}')

    #     #list_vars = ['p_D2','Ip','Wmhd','gas_fuel','gas_cum','P_sol/A','f_gw','q95','Bp', 'p_F_CRYO_MKS','p_E_BOT_MKS']
    #     #list_vars = ['Te_sep','lam_q','Ip','Wmhd','gas_cum','gas_fuel','P_sol/A','q95'] #'p_F_CRYO_MKS','p_E_BOT_MKS'] 
    #     #list_vars = ['Bt','Bp','lam_q','ne_sep','Te_sep','Wmhd','P_sol/A','q95', 'Lne','LTe']
    #     list_vars = ['rhostar','p_D2','Bt','lam_q','ne_sep','Te_sep','Wmhd','q95', 'Lne'] #'P_sol/A',

    #     out_Veff = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name, max_num_combos=2)
    #     combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_Veff

    #     rt.run_regression(aic_comb_f,
    #                       rd, rd['Veff'], rd['Veff_unc'],
    #                       color_var='favorable', y_label=r'v_{D,eff}', plot_cov=False, plot=True)




    ##### Final, hand-curated regressions

    #rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    rd = rt.subselect_database(data, np.logical_and(np.logical_or(data['regime']=='ohmic',data['regime']=='lmode'), data['CRYO_ON']==False))
    rt.run_regression(['p_D2','p_B_BOT_MKS','p_F_CRYO_MKS'],
                   rd, rd['nn_loc'], rd['nn_loc_unc'],
                   color_var='favorable', y_label='n_{n,sep}', plot_cov=False, plot=True)


    # regression on L-modes, cryo=off, only for p_D2 at midplane -- to compare to Kallenbach
    rd = rt.subselect_database(data, np.logical_and(np.logical_or(data['regime']=='ohmic',data['regime']=='lmode'), data['CRYO_ON']==False))
    rt.run_regression(['p_D2'],
                   rd, rd['nn_loc'], rd['nn_loc_unc'],
                   color_var='gas_fuel', y_label='n_{n,sep}', plot_cov=False, plot=True)

    # regression for ne_sep, all L-modes (cryo on and off)
    rd = rt.subselect_database(data, np.logical_or(data['regime']=='ohmic',data['regime']=='lmode'))
    rt.run_regression(['p_D2','Bt','Ip'],
                  rd, rd['ne_sep'], rd['ne_sep_unc'],
                  color_var='favorable', y_label='n_{e,sep}', plot_cov=False, plot=True)


    # regression for ne_sep, all H-modes -- this is a very poor regression
    rd = rt.subselect_database(data, np.logical_or(data['regime']=='EDA',data['regime']=='ELMy'))
    rt.run_regression(['p_D2','Ip'],   # little Bt variation in H-mode database at present
                  rd, rd['ne_sep'], rd['ne_sep_unc'],
                  color_var='favorable', y_label='n_{e,sep}', plot_cov=False, plot=True)


    # nn vs. p_D2 and ne_sep for the entire database, colored by geometry bias
    #rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    rd = rt.subselect_database(data, np.logical_or(data['regime']=='ohmic',data['regime']=='lmode'))
    #rd = rt.subselect_database(data, np.logical_or(data['regime']=='EDA',data['regime']=='ELMy'))
    rt.run_regression(['p_D2','Te_sep'],
                   rd, rd['nn_loc'], rd['nn_loc_unc'],
                   color_var='favorable', y_label='n_{n,sep}', plot_cov=False, plot=True)





    # nn vs. p_D2 and Te_sep only for cases with cryo off, colored by Sgas
    rd = rt.subselect_database(data, data['CRYO_ON']==False)
    rt.run_regression(['p_D2','Te_sep'],
                   rd, rd['nn_loc'], rd['nn_loc_unc'],
                   color_var='gas_fuel', y_label='n_{n,sep}', plot_cov=False, plot=True)

    # nn vs. p_D2 and ne_sep only for H-mode cases, colored by Sgas
    rd = rt.subselect_database(data, np.logical_or(data['regime']=='EDA',data['regime']=='ELMy'))
    rt.run_regression(['p_D2','Te_sep'],
                   rd, rd['nn_loc'], rd['nn_loc_unc'],
                   color_var='gas_fuel', y_label='n_{n,sep}', plot_cov=False, plot=True)


    '''
    # nn/ne at separatrix for the entire database
    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    rt.run_regression(['p_D2','Ip','q95','Te_sep'],
                      rd, y=rd['nn_loc']/rd['ne_sep'], #1e-3*rd['nn_loc']/rd['ne_sep'],
                      y_unc = np.sqrt((1./rd['ne_sep'])**2*rd['nn_loc_unc']**2+\
                                      #1e-3*np.sqrt((1./rd['ne_sep'])**2*rd['nn_loc_unc']**2+\
                                      (rd['nn_loc']/rd['ne_sep']**2)**2*rd['ne_sep_unc']**2),
                      color_var='favorable',
                      y_label='\\left(\\frac{n_n}{n_e}\\right)_{sep}',
                      rhop_loc=rhop_loc, plot_cov=False, plot=True)

    '''

    # Neutral penetration length in terms of Tesep and Bt
    rd = rt.subselect_database(data, np.logical_and(np.logical_or(data['regime']=='ohmic',
                                                                  data['regime']=='lmode'),
                                                    data['Ln_ped_exp_mm']<47.0))
    rt.run_regression(['Te_sep','Bt'],
                   rd, rd['Ln_ped_exp_mm'], 1e-3*np.ones_like(rd['Ln_ped_exp_mm_unc']),
                   color_var='favorable', y_label='L_{n_n}', plot_cov=True, plot=True)


    # Gamma_D shows effect of fav vs. unfav, but need to exclude p_D2>1 because it seems to mis-behave...
    rd = rt.subselect_database(data, mask_USN) #np.logical_and(data['p_D2']<1, mask_USN))
    rt.run_regression(['p_D2','Bt'],  
                   rd, rd['Gamma_D'], rd['Gamma_D_unc'],
                   color_var='gas_fuel', y_label='\\Gamma_D', plot_cov=False, plot=True)

    # in LSN, there is a q95 dependence, but not in USN...
    rd = rt.subselect_database(data, np.logical_and(data['p_D2']<1, mask_LSN))
    rt.run_regression(['p_D2','q95'],  # possibly without ne_sep
                   rd, rd['Gamma_D'], rd['Gamma_D_unc'],
                   color_var='gas_fuel', y_label='\\Gamma_D', plot_cov=False, plot=True)







    ###
    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    rt.run_regression(['Lne'],  # possibly without p_D2
                   rd, rd['Deff'], rd['Deff_unc'],
                   color_var='favorable', y_label='D_{D,eff}', plot_cov=False, plot=True)


    rd = rt.subselect_database(data,np.ones_like(data['regime'], dtype=bool))
    rt.run_regression(['p_D2','Wmhd'],  # use p_D2 and Wmhd
                   rd, rd['Veff'], rd['Veff_unc'],
                   color_var='gas_fuel', y_label='v_{D,eff}', plot_cov=False, plot=True)




    # # ##### Check linear dependence of S and PEC on ne
    path = '/home/sciortino/atomAI/atomdat_master/adf15/h/pju#h0.dat'

    fileloc = aurora.get_adas_file_loc('scd93_h.dat',filetype='adf11')
    scd = aurora.adas_file(fileloc)

    ne_vec = np.linspace(1e13, 1e15, 100)
    Te_vec = 1e2 * np.ones_like(ne_vec)

    # # ionization rate
    # logS = aurora.interp_atom_prof(atom_data['scd'],np.log10(ne_vec), np.log10(Te_vec), log_val=True, x_multiply=False)[:,0]

    # # Lya PEC

    # log10pec_dict = aurora.read_adf15(path)[1215.2]

    # # evaluate these interpolations on our profiles
    # pec_exc = 10**log10pec_dict['excit'].ev(np.log10(ne_vec), np.log10(Te_vec))


    # ######


    # # Ionization rates are only weakly dependent on density
    # fig,ax = plt.subplots()
    # for idens in np.arange(len(scd.logNe)):
    #     ax.plot(10**scd.logT, 10**scd.data[0,:,idens], label=f'$n_e=10^{{{scd.logNe[idens]}}}$ cm$^{{{-3}}}$')
    # ax.legend(loc='best').set_draggable(True)

    # # Ly-a PEC is completely independent of density
    # fig,ax = plt.subplots()
    # plt.plot(ne_vec, pec_exc)


    # log10pec_dict = aurora.read_adf15(path, plot_lines=[1215.2])[1215.2]


    ###
    log10pec_dict_Lya = aurora.read_adf15(path)[1215.2] # Ly-alpha
    log10pec_dict_Da = aurora.read_adf15(path)[6561.9] # D-alpha
    logTe_vec = np.log10( np.linspace(10, 1000, 100) )
    logne_vec = scd.logNe[2] # *np.ones_like(logTe_vec)
    fig, ax = plt.subplots()
    ax.plot(10**logTe_vec, 10**log10pec_dict_Lya['excit'].ev(logne_vec,logTe_vec), label='PEC Ly-a')
    ax.plot(10**logTe_vec, 10**log10pec_dict_Da['excit'].ev(logne_vec,logTe_vec), label='PEC D-a')
    scd_interp = RectBivariateSpline(scd.logT, scd.logNe, scd.data[0,:,:])
    ax.plot(10**logTe_vec, 10**scd_interp(logTe_vec, logne_vec), label='S')
    ax.legend(loc='best').set_draggable(True)


    fig,ax = plt.subplots()
    ax.plot(10**logTe_vec, 10**scd_interp(logTe_vec, logne_vec)[:,0]/10**log10pec_dict_Lya['excit'].ev( logne_vec, logTe_vec), label='Ly-a')
    #ax.plot(10**logTe_vec, 10**scd_interp(logTe_vec, logne_vec)[:,0]/10**log10pec_dict_Da['excit'].ev( logne_vec, logTe_vec), label='D-a')
    ax.set_xlabel(r'$T_e$ [eV]')
    ax.set_ylabel(r'SXB (S/PEC)')
    ax.text(700,0.9,r'Ly$_\alpha$', fontsize=25, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))
    #ax.text(700,13,r'D$_\alpha$', fontsize=25, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))
    ax.grid(True, ls='--')
    plt.tight_layout()



    fig,ax = plt.subplots()
    for dens in np.linspace(np.log10(3e13), np.log10(1e14),10):
        ax.plot(10**logTe_vec, 10**scd_interp(logTe_vec, dens)[:,0]/10**log10pec_dict_Lya['excit'].ev( dens, logTe_vec), label=f'$n_e = {10**dens:.2g}$ cm$^{-3}$')
    ax.set_xlabel(r'$T_e$ [eV]')
    ax.set_ylabel(r'SXB (S/PEC)')
    ax.text(700,0.9,r'Ly$_\alpha$', fontsize=25, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))
    ax.grid(True, ls='--')
    ax.legend(loc='best').set_draggable(True)
    plt.tight_layout()


    fig,ax = plt.subplots()
    PECs = 10**scd_interp(2, np.linspace(13,14,100))[0,:]/10**log10pec_dict_Lya['excit'].ev( np.linspace(13,14,100), 2)
    ax.plot(np.linspace(13,14,100), PECs/np.min(PECs), label=f'Ly-a')
    ax.set_xlabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_ylabel(r'SXB (S/PEC)')
    ax.text(700,0.9,r'Ly$_\alpha$', fontsize=25, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))
    ax.grid(True, ls='--')
    ax.legend(loc='best').set_draggable(True)
    plt.tight_layout()



    logTe_vec = np.log10( np.linspace(10, 1000, 100) )
    logNe_vec = np.log10(np.linspace(2e13, 2e14,100) )
    PECs = np.zeros((len(logTe_vec), len(scd.logNe)))
    for ii in np.arange(len(scd.logNe)):
        PECs[:,ii] = 10**log10pec_dict_Lya['excit'].ev(scd.logNe[ii], logTe_vec)
    SXB = 10**scd_interp(logTe_vec, scd.logNe)/PECs

    fig,ax = plt.subplots()
    cb = ax.contourf(10**logTe_vec, 10**scd.logNe, SXB.T, label='Ly-a', levels=50)
    plt.colorbar(cb)
    ax.set_xlabel(r'$T_e$ [eV]')
    ax.set_ylabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_title(r'SXB (S/PEC)')
    #ax.text(700,0.9,r'Ly$_\alpha$', fontsize=25, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5)
    ax.grid(True, ls='--')
    plt.tight_layout()



    # fig,ax = plt.subplots()
    # for ii in np.arange(res['ne_prof'].shape[1]):
    #     ax.plot(res['Te_prof'][:,50], 10**scd_interp(logTe_vec, dens)[:,0]/10**log10pec_dict_Lya['excit'].ev( dens, logTe_vec), label=f'Ly-a {10**dens:.2g}')
    # ax.set_xlabel(r'$T_e$ [eV]')
    # ax.set_ylabel(r'SXB (S/PEC)')
    # ax.text(700,0.9,r'Ly$_\alpha$', fontsize=25, bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))
    # ax.grid(True, ls='--')
    # ax.legend(loc='best').set_draggable(True)
    # plt.tight_layout()


        
    ##### New attempt at more meaningful model selection

    # key: limit to lower neutral densities -- higher ones make correlations much weaker, likely inaccurate
    rd = rt.subselect_database(data, data['nn_loc']<1e-3*1e14)

    # key: do not include ne_sep, which leads to circular logic
    list_vars = ['Bt','p_D2','Te_sep','q95','p_F_CRYO_MKS','p_E_BOT_MKS','Ip','lam_q'] #'p_B_BO
    regr_var_name='nn_loc'
    var_label = 'n_{n,sep}'

    #out_nn = rt.model_selection(rd, list_vars, regr_var_name=regr_var_name )
    #combos, r2_comb, bic_comb, aic_comb, fcombos, r2_comb_f, bic_comb_f, aic_comb_f = out_nn

    rt.run_regression(['p_D2','Te_sep'],
                      rd, rd['nn_loc'], rd['nn_loc_unc'],
                      color_var='gas_fuel', y_label='n_{n,sep}', plot_cov=False, plot=True)


if __name__ == '__main__':
    res = main_db()
    post_processing(res)
    regressions()

