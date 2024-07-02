def ne(shot, include=['CTS', 'ETS'], TCI_quad_points=None, TCI_flag_threshold=None,
       TCI_thin=None, TCI_ds=None, **kwargs):
    """Returns a profile representing electron density from both the core and edge Thomson scattering systems.
    
    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:
        
            ======= ========================
            CTS     Core Thomson scattering
            ETS     Edge Thomson scattering
            TCI     Two color interferometer
            reflect SOL reflectometer
            ======= ========================
        
        The default is to include all TS data sources, but not TCI or the
        reflectometer.
    **kwargs
        All remaining parameters are passed to the individual loading methods.
    """
    if 'electrons' not in kwargs:
        kwargs['electrons'] = MDSplus.Tree('electrons', shot)
    if 'efit_tree' not in kwargs:
        kwargs['efit_tree'] = eqtools.CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == 'CTS':
            p_list.append(neCTS(shot, **kwargs))
        elif system == 'ETS':
            p_list.append(neETS(shot, **kwargs))
        elif system == 'TCI':
            p_list.append(
                neTCI(
                    shot,
                    quad_points=TCI_quad_points,
                    flag_threshold=TCI_flag_threshold,
                    thin=TCI_thin,
                    ds=TCI_ds,
                    **kwargs
                )
            )
        elif system == 'reflect':
            p_list.append(neReflect(shot, **kwargs))
        else:
            raise ValueError("Unknown profile '%s'." % (system,))

    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)

    return p


def neETS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False, remove_zeros=True, Z_shift=0.0):
    """Returns a profile representing electron density from the edge Thomson scattering system.
    
    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
    remove_zeros: bool, optional
        If True, will remove points that are identically zero. Default is True
        (remove zero points). This was added because clearly bad points aren't
        always flagged with a sentinel value of errorbar.
    Z_shift: float, optional
        The shift to apply to the vertical coordinate, sometimes needed to
        correct EFIT mapping. Default is 0.0.
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='$10^{20}$ m$^{-3}$',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label=r'$n_e$, ETS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    N_ne_ETS = electrons.getNode(r'yag_edgets.results:ne')

    t_ne_ETS = N_ne_ETS.dim_of().data()
    ne_ETS = N_ne_ETS.data() / 1e20
    dev_ne_ETS = electrons.getNode(r'yag_edgets.results:ne:error').data() / 1e20

    Z_ETS = electrons.getNode(r'yag_edgets.data:fiber_z').data() + Z_shift
    R_ETS = (electrons.getNode(r'yag.results.param:R').data() *
             scipy.ones_like(Z_ETS))
    channels = list(range(0, len(Z_ETS)))

    t_grid, Z_grid = scipy.meshgrid(t_ne_ETS, Z_ETS)
    t_grid, R_grid = scipy.meshgrid(t_ne_ETS, R_ETS)
    t_grid, channel_grid = scipy.meshgrid(t_ne_ETS, channels)

    ne = ne_ETS.flatten()
    err_ne = dev_ne_ETS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())

    X = scipy.hstack((t.T, R.T, Z.T))

    p.shot = shot
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'

    p.add_data(X, ne, err_y=err_ne, channels={1: channels, 2: channels})
    # Remove flagged points:
    try:
        pm = electrons.getNode(r'yag_edgets.data:pointmask').data().flatten()
    except:
        pm = scipy.ones_like(p.y)
    p.remove_points(
        (pm == 0) |
        scipy.isnan(p.err_y) |
        scipy.isinf(p.err_y) |
        (p.err_y == 0.0) |
        (p.err_y == 1.0) |
        (p.err_y == 2.0) |
        ((p.y == 0.0) & remove_zeros) |
        scipy.isnan(p.y) |
        scipy.isinf(p.y)
    )
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p

