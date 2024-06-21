import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/millerma')
import pysepest.pysepest as pysepest

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
