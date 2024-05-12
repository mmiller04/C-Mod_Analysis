#-*-Python-*-
# Created by millerma at 27 Jan 2021  18:57
# modified by sciortino, 2021

# python version of Brian Labombard's get_asp_data.pro and get_fsp_data.pro

import MDSplus as mds
import sys
from omfit_classes.omfit_mds import OMFITmdsValue
from omfit_classes import omfit_mds as om
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
sys.path.append('/home/sciortino/usr/python3modules/profiletools3')
sys.path.append('/home/sciortino/usr/python3modules/eqtools3')
import profiletools
import eqtools
import matplotlib.pyplot as plt
plt.ion()
import aurora
import cmod_tools
from cmod_tools import get_geqdsk_cmod
from IPython import embed
import pymssql

def get_probe_data(shot,time, probe='A'):
    '''Load data structure from one of the C-Mod boundary probes.
    The output is not filtered in any way.

    The rho values returned by this function are the distance from the midplane 
    sepatrix in meters.

    Options: 
    probe = {'A', 'F'}
    '''
    if probe=='A':
        _SP = '\EDGE::TOP.PROBES.ASP'
    elif probe=='F':
        _SP = '\EDGE::TOP.PROBES.FSP_1'
    else:
        raise ValueError('Unrecognized probe name/port!')

    node_string = _SP + ':PLUNGE'

    #node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE', TDI=node_string)
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string) 
    plunge = node.data()
    #plunge_tm = node.dim_of(0)
    plunge_tm = node.dim_of(0).data()
    
    node_string = _SP + '.G_1:Rho'
    #node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE', TDI=node_string)
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)

    raw_rho = node.data()
    #rho_tm = node.dim_of(0)
    rho_tm = node.dim_of(0).data()

    # smooth using poly degree 1
    # window size is 11 to span 0.001 s window (specified in get_asp_data.pro)
    rho = savgol_filter(raw_rho,11,1) 

    node_string = _SP + '.G_1:origin'
    
    #node = OMFITmdsValue(server='CMOD', shot=shot, treename='EDGE', TDI=node_string)
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)

    origin = node.data()

    # Determine plunge times, plunge indices
    rho_max = 0.050

    _out = find_plunges(plunge,rho,rho_max,plunge_tm)
    nplunges,rho_peak,time_peak,indices_peak,nIndices_in,nIndices_out,indices_in,indices_out = _out

    # Find closest plunge, set rho_scan data

    dum = np.min(np.abs(time_peak - time))
    iplunge = np.where(np.abs(time_peak - time) == dum)[0][0]

    rho_scan = np.hstack((rho[indices_in[:nIndices_in[iplunge],iplunge]],
                          rho[indices_out[:nIndices_out[iplunge],iplunge]]))
    rho_scan_tm = np.hstack((rho_tm[indices_in[:nIndices_in[iplunge],iplunge]],
                             rho_tm[indices_out[:nIndices_out[iplunge],iplunge]]))
    min_rho = rho[indices_peak[iplunge]]
    plunge_scan = np.hstack((plunge[indices_in[:nIndices_in[iplunge],iplunge]],
                             plunge[indices_out[:nIndices_out[iplunge],iplunge]]))
    plunge_scan_tm = np.hstack((plunge_tm[indices_in[:nIndices_in[iplunge],iplunge]],
                                plunge_tm[indices_out[:nIndices_out[iplunge],iplunge]]))
    max_plunge = plunge[indices_peak[iplunge]]

    t_start = rho_scan_tm[0]
    t_end = rho_scan_tm[-1]
    t_peak = time_peak[iplunge]

    # Set z_scan, r_scan
    if probe == 'A':
        r_scan = origin[0] - plunge_scan
        z_scan = np.ones(len(r_scan))*origin[1]
    else:
        z_scan = origin[1] + plunge_scan
        r_scan = np.ones(len(z_scan))*origin[0]

    # Read data for each probe, interpolate for rho at times of measurement

    pnodes = ['.p0','.p1','.p2','.p3']
    pname = []
    processed = []
    probe_type = []
    for ip in range(4):
        node_string = _SP+'.G_1'+pnodes[ip]+':NAME'
        #pname_string = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
        node = mds.Tree('EDGE', shot)
        node = node.getNode(node_string)
        pname_string = node.data()
#        pname.append(str(pname_string[0]).strip().upper())
        pname.append(str(pname_string).strip().upper())
        
        node_string = _SP+'.G_1'+pnodes[ip]+':PROCESSED'
#        processed_string = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()[4]
        node = mds.Tree('EDGE', shot)
        node = node.getNode(node_string)
        processed_string = node.data()[4]          
#        processed.append(str(processed_string).strip().upper())
        processed.append(processed_string.decode().strip().upper())

        node_string = _SP+'.G_1'+pnodes[ip]+':PROBE_TYPE'        
#        probe_string = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
        node = mds.Tree('EDGE', shot)
        node = node.getNode(node_string)
        probe_string = node.data()
#        probe_type.append(str(probe_string[0]).strip().upper())
        probe_type.append(str(probe_string).strip().upper())

    psel = np.zeros(4)
    pnum = np.zeros(4)
    
    index = pname.index('NORTH') if 'NORTH' in pname else pname.index('NE')
    pnum[0] = index
    N_name = pname[index]
    
    index = pname.index('SOUTH') if 'SOUTH' in pname else pname.index('SW')
    pnum[1] = index
    S_name = pname[index]
    
    index = pname.index('EAST') if 'EAST' in pname else pname.index('SE')
    pnum[2] = index
    E_name = pname[index]
    
    index = pname.index('WEST') if 'WEST' in pname else pname.index('NW')
    pnum[3] = index
    W_name = pname[index]

    for ip in range(4):
        #if processed[ip] == 'SWEEP' and probe_type[ip] == 'SINGLEC':
        if 'SWEEP' in processed[ip] and 'SINGLEC' in probe_type[ip]:
            psel[ip] = 1
    for ip in range(4):
        if 'SINGLEF' in probe_type[ip]:
        #if probe_type[ip] == 'SINGLEF':
            psel[ip] = 2

    # Now collect data from all probe sides
    data = {}
    for ii,side in enumerate(['N','S','E','W']):
        
        data[side]={}
        data[side]['Ne'] = np.zeros(1)
        data[side]['Te'] = np.zeros(1)
        data[side]['Vf'] = np.zeros(1)
        data[side]['Js'] = np.zeros(1)
        data[side]['rho'] = np.zeros(1)
        data[side]['tbase'] = np.zeros(1)
        data[side]['sig_Ne'] = np.zeros(1)
        data[side]['sig_Te'] = np.zeros(1)
        data[side]['sig_Js'] = np.zeros(1)
        data[side]['radius'] = np.zeros(1)
        data[side]['z'] = np.zeros(1)

        ip = int(pnum[ii])
        if psel[ip] == 1:
            node_string = _SP + '.G_1' + pnodes[ip] + ':OFFSET'
            #offset = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
            node = mds.Tree('EDGE', shot)
            node = node.getNode(node_string)
            offset = node.data()

            node_string = _SP + '.G_1' + pnodes[ip] + ':VALID_FAST:INDICES'
            no_indices = 0
            try: # not sure if this is the best way to do it
                #indices = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
                node = mds.Tree('EDGE', shot)
                node = node.getNode(node_string)
                indices = node.data()
            except:
                node_string = _SP + '.G_1' + pnodes[ip] + ':NE_FAST'
                #work_Ne = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
                node = mds.Tree('EDGE', shot)
                node = node.getNode(node_string)
                work_Ne = node.data()
                indices = range(len(work_Ne))
                no_indices = 1
                
            if len(indices) > 0:
                _out = _load_probe_data(_SP,pnodes,ip,shot,no_indices,indices)
                work_Ne,work_sig_Ne,work_rho,work_rho_tm,work_Is,work_sig_Is,\
                    work_pro_area,work_pro_area_tm,work_Te,work_sig_Te,work_Vf,work_tbase = _out
                
                inrange1 = np.where(work_tbase >= rho_scan_tm[0])[0]
                inrange2 = np.where(work_tbase <= rho_scan_tm[-1])[0]
                inrange = np.intersect1d(inrange1,inrange2)
                count = len(inrange)
                if count > 0:
                    data[side]['Ne'] = work_Ne[inrange]
                    data[side]['Te'] = work_Te[inrange]
                    data[side]['Vf'] = work_Vf[inrange]
                    data[side]['sig_Ne'] = work_sig_Ne[inrange]
                    data[side]['sig_Te'] = work_sig_Te[inrange]
                    data[side]['tbase'] = work_tbase[inrange]
                    data[side]['rho'] = interp1d(work_rho_tm,work_rho)(data[side]['tbase'])
                    data[side]['radius'] = origin[0] + offset[0] + interp1d(plunge_scan_tm,plunge_scan)(data[side]['tbase'])
                    data[side]['z'] = np.ones(len(data[side]['radius']))*(origin[1] + offset[1])
                
                    data[side]['pro_area'] = interp1d(work_pro_area_tm,work_pro_area)(data[side]['tbase'])
                    data[side]['Is'] = work_Is[inrange]
                    data[side]['sig_Is'] = work_sig_Is[inrange]
                    data[side]['Js'] = np.zeros(len(data[side]['Is']))
                    data[side]['sig_Js'] = np.zeros(len(data[side]['Is']))
                    ok = np.where(data[side]['pro_area'] > 0)[0]
                    count = len(ok)
                    if count > 0:
                        data[side]['Js'][ok] = data[side]['Is'][ok]/data[side]['pro_area'][ok]
                        data[side]['sig_Js'][ok] = data[side]['sig_Is'][ok]/data[side]['pro_area'][ok]

        if psel[ip] == 2:
            node_string = _SP + '.G_1' + pnodes[ip] + ':OFFSET'
            #offset = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
            node = mds.Tree('EDGE', shot)
            node = node.getNode(node_string)
            offset = node.data()
            work_Vf,work_tbase = load_vf_data(_SP,pnodes,ip,shot)
            inrange1 = np.where(work_tbase >= rho_scan_tm[0])[0]
            inrange2 = np.where(work_tbase <= rho_scan_tm[-1])[0]
            inrange = np.intersect1d(inrange1,inrange2)
            count = len(inrange)
            if count > 0:
                data[side]['Vf'] = work_Vf[inrange]
                data[side]['tbase'] = work_tbase[inrange]
                data[side]['rho'] = interp1d(rho_scan_tm,rho_scan)(data[side]['tbase'])
                data[side]['radius'] = origin[0] + offset[0] + interp1d(plunge_scan_tm,plunge_scan)(data[side]['tbase'])
                data[side]['z'] = np.ones(len(data[side]['radius']))*(origin[1] + offset[1])


    return t_start,t_peak,t_end,data, plunge_scan,plunge_scan_tm,max_plunge,rho_scan,rho_scan_tm,min_rho



def find_plunges(plunge,rho,rho_max,plunge_tm):
    '''Find when a given probe plunged into the plasma.
    '''
    min_plunge = 0.075
    max_velocity = 5.0
    min_velocity = 0.1
    npts = len(plunge)

    # smooth plunge and rho over a 0.001 sec time window
    dt = plunge_tm - np.roll(plunge_tm,-1)
    nz = np.where(dt != 0.0)
    dt = np.min(abs(dt[nz]))
    nwidth = np.int(np.floor(0.001/dt))

    sm_plunge = boxcar(plunge,nwidth)
    sm_rho = boxcar(rho,nwidth)

    # find magnitude of plunge excursion
    # if dplunge is less than min_plunge meters then probe did not scan

    max_plunge = np.max(sm_plunge)
    dplunge = max_plunge - np.min(sm_plunge)

    if dplunge < min_plunge:
        raise ValueError(f'ERROR in FIND_PLUNGES => Probe did not scan more than {min_plunge} meteres')

    # find indices in sm_plunge where probe was within min_plunge meters of
    # peak insertion

    near_peak = np.where(sm_plunge > (max_plunge-min_plunge))[0]

    # from these points, detect where there are time 'gaps' between
    # data points of over 0.050 seconds

    near_peak_tm = plunge_tm[near_peak]
    near_peak_dtm = np.abs(near_peak_tm - np.roll(near_peak_tm,1))
    igap = np.where(near_peak_dtm > 0.025)[0]
    nplunges = len(igap)

    if nplunges <1:
        raise ValueError('ERROR in FIND_PLUNGES => No time gap greater than 25 ms was found')

    # Now igap contains the indices in time where a scan begins to pass within
    # Min_Plunge meters of max_plunge. Peak insertions for each scan, k, should
    # occur between indices of near_peak(igap(k)) and near_peak(igap(k+1)-1)
    # Extend igap array to have its last element equal to the last index in
    # near_peak plus 1. This way when k=nplunges-1, the quantity igap(k+1)-1
    # will return the index of the last element in near_peak. (This mimics the
    # start of another scan)

    igap = list(igap)
    igap.append(len(near_peak))
    igap = np.array(igap)

    # Now look for nose spikes on the plunge data: We know that the speed of
    # the probe is no more than Max_Velocity meters/second. Therefore it should
    # take at least Turnaround=2*Min_Plunge/Max_Velocity for the probe to travel
    # a total distance of 2*Min_Plunge meters (in and out). If the time span
    # between plunge_tm(near_peak(igap(k+1)-1)) and
    # plunge_tm(near_peak(igap(k))) is less than Turnaround seconds then it must
    # be a noise spike.

    turn_around = 2.0*min_plunge/max_velocity
    for k in range(nplunges-1,-1,-1):
        istart = near_peak[igap[k]]
        iend = near_peak[igap[k+1]-1]
        if (plunge_tm[iend] - plunge_tm[istart]) < turn_around:
            igap[k+1] = -1

    # Eliminate noise spike gaps

    subset = np.where(igap != -1)[0]
    nplunges = len(subset) - 1
    if nplunges < 1:
        raise ValueError('ERROR in noise spike algorithm')
    igap = igap[subset]

    # Now look for peak insertion indices based on plunge (changed from SMRho)

    rho_peak = np.zeros(nplunges)
    time_peak = np.zeros(nplunges)
    indices_peak = np.zeros(nplunges)
    for k in range(nplunges):
        istart = near_peak[igap[k]]
        iend = near_peak[igap[k+1]-1]
        plunge_max = np.max(plunge[istart:iend+1])
        imin = np.where(plunge[istart:iend+1] == plunge_max)[0][0]
        rho_peak[k] = rho[imin+istart] # using sm_rho gives wrong values
        time_peak[k] = plunge_tm[imin+istart]
        indices_peak[k] = imin+istart

    # Compute indices of in-going scans and out-going scans

    nIndices_in = np.zeros(nplunges)
    nIndices_out = np.zeros(nplunges)
    indices_in = np.zeros((npts,nplunges))
    indices_out = np.zeros((npts,nplunges))

    # Compute velocity of scanning probe

    probe_deriv = np.diff(plunge)/np.diff(plunge_tm)
    velocity = savgol_filter(probe_deriv,11,1)

    for k in range(0,nplunges):
        previous_peak = time_peak[k-1] if k > 0 else 0.0
        next_peak = time_peak[k+1] if k < nplunges-1 else np.max(plunge_tm)

        # Going in = where velocity is greater than min_velocity and time after
        # previous peak time and before current one

        condition1 = np.where(plunge_tm > previous_peak)[0]
        condition2 = np.where(plunge_tm <= time_peak[k])[0]
        condition3a = np.where(velocity > min_velocity)[0]
        condition3b = np.where(np.abs(time_peak[k]-plunge_tm) < 0.01)[0]

        condition1_2 = np.intersect1d(condition1,condition2)
        condition3 = np.union1d(condition3a,condition3b)

        in_going = np.intersect1d(condition1_2,condition3)
        count = len(in_going)

        if count > 0:

            # Eliminate gaps in in_going indices
            di = in_going - np.roll(in_going,1)
            igap = np.where(di > 1)[0]
            ngap = len(igap)
            if ngap > 0:
                in_going = in_going[igap[-1]:]
            count = len(in_going)
            iL = in_going[0]
            iR = in_going[-1]
            nIndices_in[k] = iR - iL + 1
            if nIndices_in[k] < 1: # I don't think this is necessary
                indices_in[:,k] = np.zeros(int(npts - nIndices_in[k]))
            else:
                indices_in[:,k] = np.hstack(((np.array(range(int(nIndices_in[k]))))+iL,np.zeros(int(npts - nIndices_in[k]))))

        else:
            indices_in[:,k] = np.zeros(npts) # also not necessary

        # Going out = where velocity is less than -min_velocity and time after
        # current peak time and before next one
        
        condition1 = np.where(plunge_tm >= time_peak[k])[0]
        condition2 = np.where(plunge_tm <= next_peak)[0]
        condition3a = np.where(velocity < -1*min_velocity)[0]
        condition3b = np.where(np.abs(time_peak[k]-plunge_tm) < 0.01)[0]

        condition1_2 = np.intersect1d(condition1,condition2)
        condition3 = np.union1d(condition3a,condition3b)

        out_going = np.intersect1d(condition1_2,condition3)
        count = len(out_going)

        if count > 0:

            # Eliminate gaps in out_going indices

            di = out_going - np.roll(out_going,1)
            igap = np.where(di > 1)[0]
            ngap = len(igap)
            if ngap > 0:
                out_going = out_going[:igap[0]]
            count = len(out_going)
            iL = out_going[0]
            iR = out_going[-1]
            nIndices_out[k] = iR - iL + 1
            if nIndices_out[k] < 1:
                indices_out[:,k] = np.zeros(int(npts - nIndices_out[k]))
            else:
                indices_out[:,k] = np.hstack(((np.array(range(int(nIndices_out[k]))))+iL,np.zeros(int(npts - nIndices_out[k]))))

        else:
            indices_out[:,k] = np.zeros(npts)

    # Restrict indices to where SMRho is less than Rho_max

    pcount = 0
    subset = np.where(sm_rho < rho_max)[0]
    nsubset = len(subset)
    if nsubset < 1:
        raise ValueError('ERROR in FIND_PLUNGES => no data inside rho_max = '+str(rho_max))
    inside_rho_max = np.zeros(len(sm_rho))
    inside_rho_max[subset] = 1

    for k in range(nplunges):
        active_in = np.zeros(len(sm_rho))
        indices_in = indices_in.astype(int)
        nIndices_in = nIndices_in.astype(int)
        if nIndices_in[k] > 1:
            active_in[indices_in[:nIndices_in[k],k]] = 1
        restrict_in1 = np.where(inside_rho_max != 0)
        restrict_in2 = np.where(active_in != 0)
        restrict_in = np.intersect1d(restrict_in1,restrict_in2)
        incount = len(restrict_in)

        active_out = np.zeros(len(sm_rho))
        indices_out = indices_out.astype(int)
        nIndices_out = nIndices_out.astype(int)
        if nIndices_out[k] > 1:
            active_out[indices_out[:nIndices_out[k],k]] = 1
        restrict_out1 = np.where(inside_rho_max != 0)
        restrict_out2 = np.where(active_out != 0)
        restrict_out = np.intersect1d(restrict_out1,restrict_out2)
        outcount = len(restrict_out)

        if incount > 0 and outcount > 0:

            nIndices_in[pcount] = incount
            indices_in[:incount,pcount] = restrict_in
            nIndices_out[pcount] = outcount
            indices_out[:outcount,pcount] = restrict_out

            rho_peak[pcount] = rho_peak[k]
            time_peak[pcount] = time_peak[k]
            indices_peak[pcount] = indices_peak[k]
            pcount += 1

    nplunges = pcount
    if nplunges < 1:
        raise ValueError(f'ERROR in FIND_PLUNGES => no data inside rho_max = {rho_max}')

    # Trim indices arrays

    max_n = np.max(np.hstack((nIndices_in,nIndices_out)))
    trim = range(max_n)
    indices_in = indices_in[trim,:nplunges]
    indices_out = indices_out[trim,:nplunges]
    time_peak = time_peak[:nplunges]
    rho_peak = rho_peak[:nplunges]
    indices_peak = indices_peak[:nplunges].astype(int)

    return nplunges,rho_peak,time_peak,indices_peak,nIndices_in,nIndices_out,indices_in,indices_out


def boxcar(data,window):
    return np.convolve(data,np.ones(window),'valid') / window



def _load_probe_data(_SP,pnodes,ip,shot,no_indices,indices):
    '''Method to call MDS+ nodes for C-Mod ASP or FSP probes.
    '''
    
    node_string = _SP + '.G_1' + pnodes[ip] + ':NE_FAST'
    #work_Ne = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_Ne = node.data()
    # when switching from OMFITmdsValue to MDSplus to get data this doesn't work so for now just setting errors to 1/10 of value
    #if no_indices:
    #    node_string = 'Error_of('+node_string+')'
    #else:
    #    node_string = 'Error_of('+node_string+':DATA)'
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_sig_Ne = node.data()
    #work_sig_Ne = work_sig_Ne[indices]

    node_string = _SP + '.G_1' + pnodes[ip] + ':RHO'
    #work_rho = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_rho = node.data()

    #node_string = 'dim_of(' + _SP + '.G_1' + pnodes[ip] + ':RHO)'
    #work_rho_tm = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    #node = mds.Tree('EDGE', shot)
    #node = node.getNode(node_string)
    #work_rho_tm = node.data()
    work_rho_tm = node.dim_of(0).data()

    node_string = _SP + '.G_1' + pnodes[ip] + ':IS_FAST'
    #work_Is = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_Is = node.data()
    #if no_indices:
    #    node_string = 'Error_of('+node_string+')'
    #else:
    #    node_string = 'Error_of('+node_string+':DATA)'
    #work_sig_Is = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_sig_Is = node.data()
    #work_sig_Is = work_sig_Is[indices]

    node_string = _SP + '.G_1' + pnodes[ip] + ':PRO_AREA'
    #work_pro_area = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_pro_area = node.data()
    #node_string = 'dim_of(' + node_string + ')'
    #work_pro_area_tm = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    #node = mds.Tree('EDGE', shot)
    #node = node.getNode(node_string)
    #work_pro_area_tm = node.data()
    work_pro_area_tm = node.dim_of(0).data()

    node_string = _SP + '.G_1' + pnodes[ip] + ':TE_FAST'
    #work_Te = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_Te = node.data()
    #if no_indices:
    #    node_string = 'Error_of('+node_string+')'
    #else:
    #    node_string = 'Error_of('+node_string+':DATA)'
    #work_sig_Te = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_sig_Te = node.data()
    #work_sig_Te = work_sig_Te[indices]

    node_string = _SP + '.G_1' + pnodes[ip] + ':VF_FAST'
    #work_Vf = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_Vf = node.data()
    #node_string = 'dim_of(' + node_string + ')'
    #work_tbase = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    #node = mds.Tree('EDGE', shot)
    #node = node.getNode(node_string)
    #work_tbase = node.data()
    work_tbase = node.dim_of(0).data()

    return work_Ne,work_sig_Ne,work_rho,work_rho_tm,work_Is,work_sig_Is,\
        work_pro_area,work_pro_area_tm,work_Te,work_sig_Te,work_Vf,work_tbase



def load_vf_data(_SP,pnodes,ip,shot):
    '''Load floating potential from a given C-Mod probe.
    '''
    node_string = _SP + '.G_1' + pnodes[ip] + ':VF_FAST'
    #work_Vf = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    node = mds.Tree('EDGE', shot)
    node = node.getNode(node_string)
    work_Vf = node.data()
    #node_string = 'dim_of(' + node_string + ')'
    #work_tbase = OMFITmdsValue(server='CMOD',shot=shot,treename='edge',TDI=node_string).data()
    #node = mds.Tree('EDGE', shot)
    #node = node.getNode(node_string)
    #work_tbase = node.data()
    work_tbase = node.dim_of(0).data()
    n = len(work_Vf) if len(work_Vf) < len(work_tbase) else len(work_tbase)
    work_Vf = work_Vf[:n]
    work_tbase = work_tbase[:n]

    return work_Vf, work_tbase


def get_clean_data(shot, time, geqdsk=None, probe='A', full_out=False, plot=False):
    '''Simple interface to MDS+ to see if there exist ASP or FSP probe data for a given shot.
    If so, returns data collected nearest to the requested time.
    '''
#    if geqdsk is None:
#        # the geqdsk is used only when using the Aurora radial coordinate transformations
#        geqdsk = lyman_data.get_geqdsk_cmod(
#                    shot, time*1e3, gfiles_loc='/home/millerma/lya/gfiles/')

    asp_out = get_probe_data(shot, time, probe)
    t_start,t_peak,t_end,data,plunge,t_plunge,max_plunge,rho_full,t_rho,min_rho = asp_out

    if plot: fig, ax = plt.subplots(2, figsize=(10, 12), sharex=True)

    #for side in ['N','S','E','W']:   
    #    asp_Ne = savgol_filter(data[side]['Ne'],5,1)
    #    asp_Te = savgol_filter(data[side]['Te'],5,1)
            
        #if plot:
        #ax[0].plot(data[side]['rho'], asp_Ne,'o', label=f'ASP-{side}')
        #ax[1].plot(data[side]['rho'], asp_Te,'o')

    # -----------------------
    # take averages between different probe sides
    # -----------------------
    min_rho = max([data['N']['rho'].min(),data['S']['rho'].min(),data['E']['rho'].min(),data['W']['rho'].min()])
    max_rho = min([data['N']['rho'].max(),data['S']['rho'].max(),data['E']['rho'].max(),data['W']['rho'].max()])
    res = min([len(data['N']['rho']),len(data['S']['rho']),len(data['E']['rho']),len(data['W']['rho'])])
    rho = np.linspace(min_rho,max_rho,res)
    rho[0] += 1e-10 # get some errors sometime for some reason if values are exactly the same

    # Values of rho mapped to midplane already

    # get position of separatrix in m
    R_sep = aurora.rad_coord_transform(np.array(1),'rhop','Rmid',geqdsk)    
    R_0 = geqdsk['RMAXIS']

    # get rhop and roa locations of probe measurements, mapped at the midplane
    rhop = aurora.get_rhop_RZ(R_sep+rho, np.zeros_like(rho), geqdsk) # conversion at the midplane
    roa = (R_sep+rho - R_0)/(R_sep - R_0)

    # uncertainty in r/a probe locations is rather arbitrary...assume 5mm (or get from EFIT?)
    #roa_unc = 2.5e-3/(R_sep - R_0)
    
    ne_arr = np.zeros((len(rho),4))
    Te_arr = np.zeros((len(rho),4))
    Js_arr = np.zeros((len(rho),4))
    Is_arr = np.zeros((len(rho),4))
    Vf_arr = np.zeros((len(rho),4))
    t_arr = np.zeros((len(rho),4))
    
    for ii,side in enumerate(['N','S','E','W']): # for some probe heads, there's only Vf and time
        try:
            ne_arr[:,ii] = interp1d(data[side]['rho'],data[side]['Ne'])(rho)
            Te_arr[:,ii] = interp1d(data[side]['rho'],data[side]['Te'])(rho)
            Js_arr[:,ii] = interp1d(data[side]['rho'],data[side]['Js'])(rho)
            Is_arr[:,ii] = interp1d(data[side]['rho'],data[side]['Is'])(rho)
        except:
            ne_arr[:,ii] = np.full(len(rho), 0)
            Te_arr[:,ii] = np.full(len(rho), 0)
            Js_arr[:,ii] = np.full(len(rho), 0)
            Is_arr[:,ii] = np.full(len(rho), 0)
        # these should hopefully always be there, but if not change later
        Vf_arr[:,ii] = interp1d(data[side]['rho'],data[side]['Vf'])(rho)
        t_arr[:,ii] = interp1d(data[side]['rho'],data[side]['tbase'])(rho)


#    inds_sort = np.argsort(rho_full) 

 #   t_plunge_avg = interp1d(rho_full[inds_sort],t_rho[inds_sort])(rho)

    ne_prof = np.mean(ne_arr, axis=1)
    ne_unc_prof = np.std(ne_arr, axis=1)
    
    Te_prof = np.mean(Te_arr, axis=1)
    Te_unc_prof = np.std(Te_arr, axis=1)
    
    Js_prof = np.mean(Js_arr, axis=1)
    Js_unc_prof = np.std(Js_arr, axis=1)
    
    Is_prof = np.mean(Is_arr, axis=1)
    Is_unc_prof = np.std(Is_arr, axis=1)

    Vf_prof = np.mean(Vf_arr, axis=1)
    Vf_unc_prof = np.std(Vf_arr, axis=1)

    t_prof = np.mean(t_arr, axis=1)

    Vp_prof, Vp_unc_prof = calc_Vp(Vf_prof, Vf_unc_prof, Te_prof, Te_unc_prof)
        
    if plot:
        ax[0].errorbar(roa, ne_prof, ne_unc_prof, fmt='.', c='k', label='ASP')
        ax[1].errorbar(roa, Te_prof, Te_unc_prof, fmt='.', c='k', label='ASP')
        ax[1].set_xlabel(r'r/a')
        ax[0].set_ylabel(r'$n_e$ [$m^{-3}$]')
        ax[1].set_ylabel(r'$T_e$ [$eV$]')
        ax[0].set_title(str(shot))
        ax[0].legend(loc='best').set_draggable(True)
        
    else:
        ax = None

    if full_out:
        return rhop, t_peak, [t_start,t_end], t_prof, ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, Js_prof, Js_unc_prof, Is_prof, Is_unc_prof, Vp_prof, Vp_unc_prof, ax
    else:
        return rhop, [t_start,t_end], ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, ax

def get_edgedb_data(shot, t_range, geqdsk=None, probe='A', plot=False):

    '''Simple interface to Brian's edge SQL database to see if there exist ASP or FSP probe data for a given shot.
    If so, returns data collected nearest to the requested time.
    '''

    # establish connection with edge database
    connection = pymssql.connect(host='alcdb2',database='edge',user='millerma',password='pfcworld')
    cur = connection.cursor()

    # fetch rho, ne, and Te
    cur.execute('SELECT Rho, Density, Te, Time_slice FROM {}SP_PROFILE2 WHERE Shot = {} and Time_slice > {} and Time_slice < {}'.format(probe, shot, t_range[0]*1e3, t_range[1]*1e3))
    _out = cur.fetchall()

    rho = []; ne = []; Te = []; t_slice = []
    for each_field in _out:
        rho.append(each_field[0])
        ne.append(each_field[1])
        Te.append(each_field[2])
        t_slice = each_field[3]
    rho, ne, Te = np.array(rho), np.array(ne), np.array(Te)

    if t_slice != []:
        # fetch offset - manual shift in rho decided on upon inspection
        cur.execute('SELECT Offset_{}SP from MASTER Where Shot = {} and Time_slice = {}'.format(probe, shot, t_slice))
        offset = cur.fetchall()
        offset = offset[0][0]

        # Values of rho mapped to midplane already
        if offset is not None:
            rho += offset
        
            # return offset in roa units
            #offset = offset/(R_sep - R_0)
    
    # get position of separatrix in m
    R_sep = aurora.rad_coord_transform(np.array(1),'rhop','Rmid',geqdsk)    
    R_0 = geqdsk['RMAXIS']

    # get rhop and roa locations of probe measurements, mapped at the midplane
    rhop = aurora.get_rhop_RZ(R_sep+rho, np.zeros_like(rho), geqdsk) # conversion at the midplane
    #roa = (R_sep+rho - R_0)/(R_sep - R_0)
 
    return rhop, ne, Te


def load_ets(shot,time, ax=None, plot=True):
    '''Load edge Thomson.

    Time in seconds. Currently averaging 200 ms on each side of the given time.
    '''
    # plot also edge Thomson data        
    p_Te_ETS = profiletools.Te(shot, include=['ETS'],
                              abscissa='sqrtpsinorm',t_min=time-0.2,t_max=time+0.2)
    p_ne_ETS = profiletools.ne(shot, include=['ETS'],
                              abscissa='sqrtpsinorm',t_min=time-0.2,t_max=time+0.2)
        
    # time average over this time window
    p_ne_ETS.time_average(weighted=True)
    p_Te_ETS.time_average(weighted=True)
        
    # clean up
    p_ne_ETS.remove_points(p_ne_ETS.y==0)
    p_Te_ETS.remove_points(p_Te_ETS.y==0)
    p_ne_ETS.remove_points(p_ne_ETS.err_y/p_ne_ETS.y>1.0)
    p_Te_ETS.remove_points(p_Te_ETS.err_y/p_Te_ETS.y>1.0)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(2, figsize=(10, 12), sharex=True)
            ax[1].set_xlabel(r'$\rho$')
            ax[0].set_ylabel(r'$n_e$ [$m^{-3}$]')
            ax[1].set_ylabel(r'$T_e$ [$eV$]')
            ax[0].set_title(str(shot))

        ax[0].errorbar(p_ne_ETS.X[:,0], p_ne_ETS.y*1e20, p_ne_ETS.err_y*1e20, fmt='.', label='ETS')
        ax[1].errorbar(p_Te_ETS.X[:,0], p_Te_ETS.y*1e3, p_Te_ETS.err_y*1e3, fmt='.')
    
        ax[0].legend(loc='best').set_draggable(True)

    else:
        ax = None
        
    return p_ne_ETS, p_Te_ETS, ax


def calc_Vp(Vf, Vf_unc, Te, Te_unc):

    m_e = 9.11e-31
    m_i = 1.673e-27 + 1.675e-27

    Vp = Vf - Te*0.5*(np.log(2*np.pi*m_e/m_i) - 1)
    Vp_unc = np.sqrt(Vf_unc**2 + (Te_unc*0.5*np.log(2*np.pi*m_e/m_i)-1)**2)

    return Vp, Vp_unc


if __name__=='__main__':

    # millerma
    #shot = 1070710003
    #time = 1.0 #0.001

    # awesome data
    shot = 1070511002
    time = 1.0

    #shot = 1070511010
    #time = 1.0


    probes_shots = [1070511002, 1070511010, 1070511011, 1070511013, 1070511019,\
                    1070511022, 1070511035, 1070518013, 1070518016, 1070518021,\
                    1070518028, 1070627005, 1070627006, 1070627007, 1070627009,\
                    1070627011, 1070627013, 1070627014, 1070627018, 1070627020,\
                    1070627022, 1070627023, 1070627024, 1070627025, 1070627026,\
                    1070627027, 1070627029, 1070627030, 1070627031, 1070627032,\
                    1070710004, 1070710005, 1070710006, 1070710011, 1070725009,\
                    1070725010, 1070725011, 1070725016, 1070725023]

    # check 08 shots
    file_ohmic = 'lyman_ohmic_fy08.txt'
    file_lmodes = 'lyman_lmodes_fy08.txt'
    file_hmodes = 'lyman_hmodes_fy08.txt'

    shots_ohmic_fromtxt = np.genfromtxt(file_ohmic, skip_header=2)
    shots_lmodes_fromtxt = np.genfromtxt(file_lmodes, skip_header=2)
    shots_hmodes_fromtxt = np.genfromtxt(file_hmodes, skip_header=2)

    shots_ohmic = [int(each_shot[1]) for each_shot in shots_ohmic_fromtxt]
    shots_lmodes = [int(each_shot[1]) for each_shot in shots_lmodes_fromtxt]
    shots_hmodes = [int(each_shot[1]) for each_shot in shots_hmodes_fromtxt]
    
    #    shot = shots_ohmic[100]
    #    time = 1.0
    #    out = get_clean_data(shot, time,probe='A', plot=True)
    #    rho, rho_unc, t_range, ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, ax = out
    
    Agood = 0
    for shot in shots_ohmic:
        try:
            out = get_clean_data(shot, time, probe='A', plot=False)
            roa, roa_unc, t_range, ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, ax = out
            Agood+=1
        except:
            pass

    Fgood = 0
    for shot in shots_ohmic[10:20]:
        try:
            out = get_clean_data(shot, time, probe='F', plot=False)
            roa, roa_unc, t_range, ne_prof, ne_unc_prof, Te_prof, Te_unc_prof, ax = out
            Fgood+=1
            print(shot,time)
        except:
            pass
