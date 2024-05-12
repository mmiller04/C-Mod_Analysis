'''Routines to produce robust modified tanh fits, particularly suited for pedestal/edge analysis.

sciortino,2021

Routines added by Aaron Rosenthal and Andres Miller

'''

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import interp1d
from scipy import optimize

def mtanh_profile(x_coord, edge=0.08, ped=0.4, core=2.5, expin=1.5, expout=1.5, widthp=0.04, xphalf=None):
    """
     This function generates H-mode-like  density and temperature profiles on the input x_coord grid.

    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate of choice
    edge : float
        Separatrix height
    ped : float
        pedestal height
    core : float
        On-axis profile height
    expin : float
        Inner core exponent for H-mode pedestal profile
    expout : float
        Outer core exponent for H-mode pedestal profile
    widthp : float
        Width of pedestal
    xphalf : float
        Position of tanh

    Returns
    -------
    val : 1D array
        modified tanh profile

    Notes
    -----
    This function is inspired by an OMFIT function, somewhere in the framework (math_utils.py?).
    """

    w_E1 = 0.5 * widthp  # width as defined in eped
    if xphalf is None:
        xphalf = 1.0 - w_E1

    xped = xphalf - w_E1

    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)

    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_E1) - pconst) + edge

    #xpsi = np.linspace(0, 1, rgrid)
    #ones = np.ones(rgrid)
    
    val = 0.5 * a_t * (1.0 - np.tanh((x_coord - xphalf) / w_E1) - pconst) + edge * np.ones_like(x_coord)

    xtoped = x_coord / xped
    for i in range(0, len(x_coord)):
        if xtoped[i] ** expin < 1.0:
            val[i] = val[i] + (core - coretanh) * (1.0 - xtoped[i] ** expin) ** expout

    return val


def mtanh_profile_gradient(x_coord, edge=0.08, ped=0.4, core=2.5, expin=1.5, expout=1.5, widthp=0.04, xphalf=None):
    """
     This function generates H-mode-like  density and temperature profiles on the input x_coord grid.

    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate of choice
    edge : float
        Separatrix height
    ped : float
        pedestal height
    core : float
        On-axis profile height
    expin : float
        Inner core exponent for H-mode pedestal profile
    expout : float
        Outer core exponent for H-mode pedestal profile
    widthp : float
        Width of pedestal
    xphalf : float
        Position of tanh

    Returns
    -------
    val : 1D array
        modified tanh profile

    Notes
    -----
    This function is inspired by an OMFIT function, somewhere in the framework (math_utils.py?).
    """

    # function itself
    w_E1 = 0.5 * widthp  # width as defined in eped
    if xphalf is None:
        xphalf = 1.0 - w_E1

    xped = xphalf - w_E1

    pconst = 1.0 - np.tanh((1.0 - xphalf) / w_E1)
    a_t = 2.0 * (ped - edge) / (1.0 + np.tanh(1.0) - pconst)

    coretanh = 0.5 * a_t * (1.0 - np.tanh(-xphalf / w_E1) - pconst) + edge

    #xpsi = np.linspace(0, 1, rgrid)
    #ones = np.ones(rgrid)
    
    val = 0.5 * a_t * (1.0 - np.tanh((x_coord - xphalf) / w_E1) - pconst) + edge * np.ones_like(x_coord)

    xtoped = x_coord / xped
    for i in range(0, len(x_coord)):
        if xtoped[i] ** expin < 1.0:
            val[i] = val[i] + (core - coretanh) * (1.0 - xtoped[i] ** expin) ** expout


    # gradient of function - analytic
    dval_dx = -a_t / widthp * ( 1 / np.cosh((x_coord - xphalf) / w_E1)**2)
    for i in range(0, len(x_coord)):
        if xtoped[i] ** expin < 1.0:
            dval_dx[i] = dval_dx[i] - (expout * expin * (core - coretanh) * (xtoped[i] ** (expin - 1)) * (1 - xtoped[i] ** expin) ** (expout - 1)) / xped

    return dval_dx


# PFS this is the function I use for the pedestal (includes core and SOL terms)
def Osbourne_Tanh(x,C,reg=None,plot = False):
    """
    adapted from Osborne via Hughes idl script
    tanh function with cubic or quartic inner and linear
    to quadratic outer extensions and 

    INPUTS: 
    c = vector of coefficients defined as such:
    c[0] = pedestal center position
    c[1] = pedestal full width
    c[2] = Pedestal top
    c[3] = Pedestal bottom
    c[4] = inboard slope
    c[5] = inboard quadratic term
    c[6] = inboard cubic term
    c[7] = outboard linear term
    c[8] = outbard quadratic term
    x = x-axis
    """

    if reg is not None:
        for each_coeff in reg:
            C[each_coeff] = 0

    z = 2. * ( C[0] - x ) / C[1]
    P1 = 1. + C[4] * z + C[5] * z**2 + C[6] * z**3
    P2 = 1. + C[7] * z + C[8] * z**2
    E1 = np.exp(z)
    E2 = np.exp(-1.*z)
    F = 0.5 * ( C[2] + C[3] + ( C[2] - C[3] ) * ( P1 * E1 - P2 * E2 ) / ( E1 + E2 ) )

    if plot:
        f, a = plt.subplots(1,1)
        a.plot(x,F)
        plt.show()

    return F


def Osbourne_Tanh_gradient(x,C,reg=None,plot = False):
    """
    adapted from Osborne via Hughes idl script
    tanh function with cubic or quartic inner and linear
    to quadratic outer extensions and 

    INPUTS: 
    c = vector of coefficients defined as such:
    c[0] = pedestal center position
    c[1] = pedestal full width
    c[2] = Pedestal top
    c[3] = Pedestal bottom
    c[4] = inboard slope
    c[5] = inboard quadratic term
    c[6] = inboard cubic term
    c[7] = outboard linear term
    c[8] = outbard quadratic term
    x = x-axis
    """

    if reg is not None:
        for each_coeff in reg:
            C[each_coeff] = 0

    # same as function
    z = 2. * ( C[0] - x ) / C[1]
    P1 = 1. + C[4] * z + C[5] * z**2 + C[6] * z**3
    P2 = 1. + C[7] * z + C[8] * z**2
    E1 = np.exp(z)
    E2 = np.exp(-1.*z)
    F = 0.5 * ( C[2] + C[3] + ( C[2] - C[3] ) * ( P1 * E1 - P2 * E2 ) / ( E1 + E2 ) )

    # gradient of function - analytic
    dz_dx = -2/C[1]
    dP1_dz = C[4] + 2*C[5]*z + 3*C[6]*z**2
    dP2_dz = C[7] + 2*C[8]*z
    #dF_dx = 0.125 * (C[2] - C[3])*dz_dx*(1/np.cosh(z))**2*(2*E1*np.cosh(z)*dP1_dz + 2*P1 - 2*E2*np.cosh(z)*dP2_dz + 2*P2)
    dF_dx = 0.25 * (C[2] - C[3]) * dz_dx * (1/np.cosh(z)) * ((E1*dP1_dz - E2*dP2_dz) + (1/np.cosh(z)) * (P1 + P2))

    if plot:
        f, a = plt.subplots(1,1)
        a.plot(x,F)
        plt.show()

    return dF_dx


def chi_square(x_coord, vals, unc, fit, fitParam, edge_chi=False):
    """calculates reduced chi square from covarience matrix, popt from curve_fit"""
   
    if edge_chi:
        edge_mask = np.logical_and(x_coord > 0.95, x_coord < 0.999)
        #edge_mask = np.logical_and(x_coord > 0.95, x_coord < 1.02)
       
        x_coord = x_coord[edge_mask]
        vals = vals[edge_mask]
        unc = unc[edge_mask]
        fit = fit[edge_mask]

    # calculate residual
    if np.any(unc == None):

        r = np.sum((vals-fit)**2)
        chisq = r/(x_coord.shape[0] - np.sum(np.array(fitParam) != 0))

        return chisq

    else:

        r = np.sum(((vals-fit)/unc)**2)
        chisq = r/(x_coord.shape[0] - np.sum(np.array(fitParam) != 0))

        return chisq



def best_osbourne(x_coord, vals, vals_unc=None, x_out=None,
              guess=[], plot=False,maxfev=2000,bounds = None,reg=None,edge_chi=False, ped_width=None, ne=True):

    """Runs super_fit_osbourne below with 3 different gueses
    runs user can specify None to all 3 guesses using guess = [guess1,guess2,guess3]
    where guess1 = c = [c1,c2,c3...]
    returns the best guess by X^2 value: fit data on x_out, fitParam, fitUnc (1 SD)
    
    Guesses tend to not focus on exponents as those don't cause failure

    Removes all 0 entries from inputs, as this causes fits to fail
    """

    #mask = vals!=0
    #x_coord = x_coord[mask]
    #vals = vals[mask]
    #vals_unc = vals_unc[mask]

    #first setup results
    popt = np.full((3,9),np.inf)
    pcov = np.full((3,9),np.inf)

    xsqr = np.full(3,np.inf)

    if x_out is None:
        res_fit = np.full((3,x_coord.shape[0]),np.inf)
    else:
        res_fit = np.full((3,x_out.shape[0]),np.inf)

    # Now set up guesses

    # Run with no exponents and different positional arguments
    width = 0.06
    xphalf0 = 0.9*(x_coord.min()+x_coord.max()) # assuming good coverage of full profile
    ped_height_guess = vals.mean(0) if ne else vals.mean(0)/1.5

    #guess2 = [xphalf0,width, ped_height_guess, ped_height_guess/5, 0,0,0,0,0]
    guess2 = [1.0,0.03,0.4,0.1,0,0,0,0,0] # modification by MAM - this works better for C-Mod cases I've been looking at

    # Add slopes and more complicated positional guesses 
    # These have been commented out as they did not work as well for C-Mod cases

    '''
    #width = np.diff(x_coord).min()
    #position should be maximum gradient...risky due to uneven spacing
    #ind = np.abs(np.gradient(vals)).argmax()
    #xphalf0 = x_coord[ind]
    #ped_height_guess = vals.mean(0)
    #ped_slope_guess = np.abs((vals[0]-ped_height_guess)/(np.min(x_coord)-np.max(x_coord)))/ped_height_guess #expect positive
    #sol_slope_guess = -ped_slope_guess/10.0
    '''

    width = 0.04
    xphalf0 = 1.02 - width * 0.5
    ped_height_guess = interp1d(x_coord, vals)(x_coord.mean())
    ped_slope_guess = np.abs((vals[0]-ped_height_guess)/(np.min(x_coord)-np.max(x_coord)))/ped_height_guess/2 #expect positive
    sol_slope_guess = -ped_slope_guess/20.0
    #guess = [xphalf0,width, ped_height_guess, vals[-1], ped_slope_guess,0,0,sol_slope_guess,0]
    guess3 = [xphalf0,width, ped_height_guess, vals[-3:].mean(), ped_slope_guess,0,0,sol_slope_guess,0]

    # Load guesses into a list
    guesses = [None,guess2,guess3]
    ped_width_list = [ped_width,None,None]
    if guess!=[]:
        for i in range(len(guess)):
            guesses[i] = guess[i]
    failures = 0

    for i in range(len(guesses)):

        try:
            cFit, cFitParam, cUnc = super_fit_osbourne(x_coord, vals, vals_unc=vals_unc, x_out=x_out,plot=plot,maxfev=maxfev,bounds = bounds,guess = guesses[i],reg=reg,ped_width=ped_width_list[i], ne=ne)
            #cFit, cFitParam, cUnc = super_fit_osbourne(x_coord, vals, vals_unc=vals_unc, x_out=x_out,plot=plot,maxfev=maxfev,bounds = bounds,reg=reg)
            res_fit[i,:] = cFit
            popt[i,:] = cFitParam
            pcov[i,:] = cUnc

            if len(cFit) == len(vals):
                xsqr[i] = chi_square(x_coord,vals, vals_unc,cFit,cFitParam, edge_chi=edge_chi)
            else:
                cFitXSQR  = Osbourne_Tanh(x_coord,cFitParam,reg=reg)
                xsqr[i] = chi_square(x_coord, vals, vals_unc, cFitXSQR, cFitParam, edge_chi=edge_chi)
        
        except Exception as e:
            #print(e)
            #print('guess '+str(i+1)+' failed')
            failures +=1
    """
    #now perform Francesco's fit...
    cFit, cFitParam,cUnc= super_fit(x_coord, vals, vals_unc=None, x_out=x_out, edge_focus=True,
              bounds=None, plot=False)
    res_fit[-1,:] = cFit
    popt[-1,:len(cFitParam)] = cFitParam
    pcov[-1,:len(cUnc)] = cUnc

    xsqr[-1] = chi_square(x_coord,vals, vals_unc,np.asarray(cFit),np.asarray(cFitParam))

    """

    if failures ==3:
        print('Warning: No succesful fits')
        if plot:
            f,a = plt.subplots(1,1)
            a.plot(x_coord,vals)
            plt.show()
    #    import pdb
    #    pdb.set_trace()
    #print('xsqr: '+str(xsqr))

    ind = np.nanargmin(xsqr)

    return res_fit[ind,:],popt[ind,:],pcov[ind,:],xsqr[ind]



def super_fit_osbourne(x_coord, vals, vals_unc=None, x_out=None,
              guess=None, plot=False,maxfev=2000,bounds = None, reg=None, ped_width=None, ne=True):
    '''Fast and complete 1D full-profile fit.
    
    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate on which profile is given
    vals : 1D array
        Profile values to be fitted
    vals_unc : 1D array
        Array containing uncertainties for values. If these are not given, the fit uses
        only the values, i.e. it uses least-squares rather than chi^2 minimization.
    x_out : 1D array
        Desired output coordinate array. If not given, this is set equal to the x_coord array.
    plot : bool
        If True, plot the raw data and the fit. 
    guess : estimate of the fitting parameters guess = c = [c[0],c[1],c[2],...] as specified 
        in returns
    maxfev: nubmer of iterations to converge before aborting

    Returns
    -------
    res_fit : 1D array
        Fitted profile, on the x_out grid (which may be equal to x_coord if x_out was not provided)
    c = vector of coefficients defined as such:
    c[0] = pedestal center position
    c[1] = pedestal full width
    c[2] = Pedestal top
    c[3] = Pedestal bottom
    c[4] = inboard slope
    c[5] = inboard quadratic term
    c[6] = inboard cubic term
    c[7] = outboard linear term
    c[8] = outbard quadratic term

    Notes
    -----
    This is adopted from Jerry Hughes script on c-mod in idl. Typically fails due to guess or
    too low of maxfev
    '''
    
    if isinstance(vals, (int,float)) or isinstance(x_coord, (int, float)):
        raise ValueError('Input profile to super_fit is a scalar! This function requires 1D profiles.')

    def func(x,c0,c1,c2,c3,c4,c5,c6,c7,c8):
        c = np.asarray([c0,c1,c2,c3,c4,c5,c6,c7,c8])
        #print('COEFS',c)
        nval = Osbourne_Tanh(x,c,reg=reg)
        return nval

    # see parameters order in the function docstring
    if guess==None:
        
        # guesses for minimization
        # unfortunately need different guesses for ne,Te - just lower pedestal top and wider pedestal
        
        if ne:
            width = ped_width if ped_width!=None else 0.02
            ped_height_guess = interp1d(x_coord, vals)(x_coord.mean())
            ped_slope_guess = np.abs((vals[0]-ped_height_guess)/(np.min(x_coord)-np.max(x_coord)))/ped_height_guess #expect positive
        
        else:
            width = ped_width if ped_width!=None else 0.1
            ped_height_guess = interp1d(x_coord, vals)(x_coord.mean())/2
            ped_slope_guess = np.abs((vals[0]-ped_height_guess)/(np.min(x_coord)-np.max(x_coord)))/ped_height_guess/5 #expect positive

        xphalf0 = 1 - width * 0.5
        sol_slope_guess = -ped_slope_guess/20.0
        guess = [xphalf0,width, ped_height_guess, vals[-1], ped_slope_guess,0,0,sol_slope_guess,0]
        # guess = [0.985, 0.03, 1.34, 0.1, 0.606, 0, 0, -0.061, 0]
        # actual = [0.979, 0.023, 0.54, 0.022, 0.937, 0.0047, -0.000032, -0.0550, 0.000128]

        #print('GUESS', guess)

    # relatively agnostic bounds that should work in all radial coordinate choices
    if bounds is None:
        #bounds = [(0.8,1.15), (0.01,0.1), (0,None), (0, None),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)]
        bounds = ([0.9,0.005,0,0,-2,-2,-2,-2,-2],[1.05,0.3,1e21,1e21,2,2,2,2,2])
        #bounds = ([0.9,0.005,0,0,-1,-1,-1,-1,-1],[1.1,0.1,1e21,1e21,1,1,1,1,1])

    #print('BOUNDS',bounds)
    
    #print('GUESS',guess)
    # run fit/minimization    
    popt,pcov = optimize.curve_fit(func,x_coord,vals,p0=guess,sigma = vals_unc,bounds = bounds,method = 'trf',maxfev=maxfev) 
    perr = np.sqrt(np.diag(pcov))   
    #print('VALS',popt)

    if x_out is None:
        x_out = x_coord
    res_fit = Osbourne_Tanh(x_out,popt,reg=reg)

    # ensure positivity/minima
    res_fit[res_fit<np.nanmin(vals)] = np.nanmin(vals)
    
    if plot:
        plt.figure()
        if vals_unc is None:
            plt.plot(x_coord, vals, 'r.', label='raw')
        else:
            plt.errorbar(x_coord, vals, vals_unc, fmt='.',c='r', label='raw')
        plt.plot(x_out, res_fit, 'b-', label='fit')
        y1 = Osbourne_Tanh(x_out,popt+perr)
        y2 = Osbourne_Tanh(x_out,popt-perr)

        plt.plot(x_out,y1,'b--')
        plt.plot(x_out,y2,'b--')
        plt.fill_between(x_out,y1,y2,facecolor = 'gray',alpha = 0.15)
        plt.legend()
        plt.show()

    return res_fit, popt,perr


def super_fit(x_coord, vals, vals_unc=None, x_out=None, edge_focus=True,
              bounds=None, plot=False):
    '''Fast and complete 1D full-profile fit.
    
    Parameters
    ----------
    x_coord : 1D array
        Radial coordinate on which profile is given
    vals : 1D array
        Profile values to be fitted
    vals_unc : 1D array
        Array containing uncertainties for values. If these are not given, the fit uses
        only the values, i.e. it uses least-squares rather than chi^2 minimization.
    x_out : 1D array
        Desired output coordinate array. If not given, this is set equal to the x_coord array.
    edge_focus : bool
        If True, the fit takes special care to fit the pedestal and SOL and may give a poor core
        fit. If False, a weaker weight will be assigned to the optimal pedestal match.
    bounds : array of 2-tuple
        Bounds for optimizer. Must be of the right shape! See c array of parameters below.
        If left to None, a default set of bounds will be used.
    plot : bool
        If True, plot the raw data and the fit. 

    Returns
    -------
    res_fit : 1D array
        Fitted profile, on the x_out grid (which may be equal to x_coord if x_out was not provided)
    c : 1D array
        Fitting parameters to the `mtanh_profile` function, in the following order:
        :param edge: (float) separatrix height
        :param ped: (float) pedestal height
        :param core: (float) on-axis profile height
        :param expin: (float) inner core exponent for H-mode pedestal profile
        :param expout (float) outer core exponent for H-mode pedestal profile
        :param widthp: (float) width of pedestal
        :param xphalf: (float) position of tanh

    Notes
    -----
    Note that this function doesn't care about what radial coordinate you pass in x_coord,
    but all fitted parameters will be consistent with the coordinate choice made.    
    '''
    
    if isinstance(vals, (int,float)) or isinstance(x_coord, (int, float)):
        raise ValueError('Input profile to super_fit is a scalar! This function requires 1D profiles.')

    def func(c):
        if any(c < 0):
            return 1e10
        nval = mtanh_profile(x_coord, edge=c[0], ped=c[1], core=c[2], expin=c[3], expout=c[4], widthp=c[5], xphalf=c[6])
        if vals_unc is None:
            cost = np.sqrt(sum(((vals - nval) ** 2 / vals[0] ** 2) * weight_func))
        else:
            cost = np.sqrt(sum(((vals - nval) ** 2 / vals_unc ** 2) * weight_func))
        return cost

    if edge_focus:
        weight_func = ((x_coord > 0.85)*(x_coord<1.1) * x_coord + 0.8)**2 #* x_coord
        #weight_func = ((x_coord > 0.85) * x_coord + 0.001) * x_coord
    else:
        weight_func = 0.1 + x_coord #(x_coord + 0.001) * x_coord

    # guesses for minimization
    width = 0.03
    xphalf0 = 1 - width * 0.5
    ped_height_guess = interp1d(x_coord, vals)([1 - 2 * width])[0]

    # see parameters order in the function docstring
    guess = [vals[-1], ped_height_guess, vals[0], 2.0, 2.0, width, xphalf0]
    #print(guess)

    # relatively agnostic bounds that should work in all radial coordinate choices
    if bounds is None:
        #bounds = [(0, None), (0,None), (0,None), (None,None), (None,None), (0.01,0.05), (0.9,1.1)]
        bounds = [(0, None), (0,None), (0,None), (None,None), (None,None), (0.01,0.1), (0.8,1.1)]

    
    # run fit/minimization  

    res = optimize.minimize(func, guess,method='L-BFGS-B', #'Nelder-Mead',# NM cannot take bounds
                                          bounds=bounds, jac=False)  
    c = list(map(float,res.x))
    err = np.diag(res.hess_inv.todense()) #estimate error

    if x_out is None:
        x_out = x_coord
    res_fit = mtanh_profile(x_out, edge=c[0], ped=c[1], core=c[2], expin=c[3], expout=c[4], widthp=c[5], xphalf=c[6])

    # ensure positivity/minima
    res_fit[res_fit<np.nanmin(vals)] = np.nanmin(vals)
            
    if len(res_fit) == len(vals):
        xsqr = chi_square(x_coord, vals, vals_unc, res_fit, c)
    else:
        res_fit_xsqr  = mtanh_profile(x_coord, edge=c[0], ped=c[1], core=c[2], expin=c[3], expout=c[4], widthp=c[5], xphalf=c[6])
        xsqr = chi_square(x_coord, vals, vals_unc, res_fit_xsqr, c)
    
    if plot:
        plt.figure()
        if vals_unc is None:
            plt.plot(x_coord, vals, 'r.', label='raw')
        else:
            plt.errorbar(x_coord, vals, vals_unc, fmt='.',c='r', label='raw')
        plt.plot(x_out, res_fit, 'b-', label='fit')
        plt.legend()

    return res_fit, c, err, xsqr
