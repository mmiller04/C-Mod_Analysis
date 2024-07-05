###############################

### stuff from profiletools ###

###############################

try:
    import MDSplus
    _has_MDS = True
except Exception as _e_MDS:
    if isinstance(_e_MDS, ImportError):
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work.",
                      ModuleWarning)
    else:
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work. Exception raised "
                      "was of type %s, message was '%s'."
                      % (_e_MDS.__class__, _e_MDS.message),
                      ModuleWarning)
    _has_MDS = False

import scipy
import matplotlib.pyplot as plt

_X_label_mapping = {'psinorm': r'$\psi_n$',
                    'phinorm': r'$\phi_n$',
                    'volnorm': r'$V_n$',
                    'Rmid': r'$R_{mid}$',
                    'r/a': '$r/a$',
                    'sqrtpsinorm': r'$\sqrt{\psi_n}$',
                    'sqrtphinorm': r'$\sqrt{\phi_n}$',
                    'sqrtvolnorm': r'$\sqrt{V_n}$',
                    'sqrtr/a': r'$\sqrt{r/a}$'}
_abscissa_mapping = {y:x for x, y in _X_label_mapping.items()}
_X_unit_mapping = {'psinorm': '',
                   'phinorm': '',
                   'volnorm': '',
                   'Rmid': 'm',
                   'r/a': '',
                   'sqrtpsinorm': '',
                   'sqrtphinorm': '',
                   'sqrtvolnorm': '',
                   'sqrtr/a': ''} 

class Profile(object):
    """Object to abstractly represent a profile.
    
    Parameters
    ----------
    X_dim : positive int, optional
        Number of dimensions of the independent variable. Default value is 1.
    X_units : str, list of str or None, optional
        Units for each of the independent variables. If `X_dim`=1, this should
        given as a single string, if `X_dim`>1, this should be given as a list
        of strings of length `X_dim`. Default value is `None`, meaning a list
        of empty strings will be used.
    y_units : str, optional
        Units for the dependent variable. Default is an empty string.
    X_labels : str, list of str or None, optional
        Descriptive label for each of the independent variables. If `X_dim`=1,
        this should be given as a single string, if `X_dim`>1, this should be
        given as a list of strings of length `X_dim`. Default value is `None`,
        meaning a list of empty strings will be used.
    y_label : str, optional
        Descriptive label for the dependent variable. Default is an empty string.
    weightable : bool, optional
        Whether or not it is valid to use weighted estimators on the data, or if
        the error bars are too suspect for this to be valid. Default is True
        (allow use of weighted estimators).
    
    Attributes
    ----------
    y : :py:class:`Array`, (`M`,)
        The `M` dependent variables.
    X : :py:class:`Matrix`, (`M`, `X_dim`)
        The `M` independent variables.
    err_y : :py:class:`Array`, (`M`,)
        The uncertainty in the `M` dependent variables.
    err_X : :py:class:`Matrix`, (`M`, `X_dim`)
        The uncertainties in each dimension of the `M` independent variables.
    channels : :py:class:`Matrix`, (`M`, `X_dim`)
        The logical groups of points into channels along each of the independent variables.
    X_dim : positive int
        The number of dimensions of the independent variable.
    X_units : list of str, (X_dim,)
        The units for each of the independent variables.
    y_units : str
        The units for the dependent variable.
    X_labels : list of str, (X_dim,)
        Descriptive labels for each of the independent variables.
    y_label : str
        Descriptive label for the dependent variable.
    weightable : bool
        Whether or not weighted estimators can be used.
    transformed : list of :py:class:`Channel`
        The transformed quantities associated with the :py:class:`Profile` instance.
    gp : :py:class:`gptools.GaussianProcess` instance
        The Gaussian process with the local and transformed data included.
    """
    def __init__(self, X_dim=1, X_units=None, y_units='', X_labels=None, y_label='',
                 weightable=True):
        self.X_dim = X_dim
        self.weightable = weightable
        if X_units is None:
            X_units = [''] * X_dim
        elif X_dim == 1:
            X_units = [X_units]
        elif len(X_units) != X_dim:
            raise ValueError("The length of X_units must be equal to X_dim!")
        
        if X_labels is None:
            X_labels = [''] * X_dim
        elif X_dim == 1:
            X_labels = [X_labels]
        elif len(X_labels) != X_dim:
            raise ValueError("The length of X_labels must be equal to X_dim!")
        
        self.X_units = X_units
        self.y_units = y_units
        
        self.X_labels = X_labels
        self.y_label = y_label
        
        self.y = scipy.array([], dtype=float)
        self.X = None
        self.err_y = scipy.array([], dtype=float)
        self.err_X = None
        self.channels = None
        
        self.transformed = scipy.array([], dtype=Channel)
        
        self.gp = None


    def add_data(self, X, y, err_X=0, err_y=0, channels=None):
        """Add data to the training data set of the :py:class:`Profile` instance.
        
        Will also update the Profile's Gaussian process instance (if it exists).
        
        Parameters
        ----------
        X : array-like, (`M`, `N`)
            `M` independent variables of dimension `N`.
        y : array-like, (`M`,)
            `M` dependent variables.
        err_X : array-like, (`M`, `N`), or scalar float, or single array-like (`N`,), optional
            Non-negative values only. Error given as standard deviation for
            each of the `N` dimensions in the `M` independent variables. If a
            scalar is given, it is used for all of the values. If a single
            array of length `N` is given, it is used for each point. The
            default is to assign zero error to each point.
        err_y : array-like (`M`,) or scalar float, optional
            Non-negative values only. Error given as standard deviation in the
            `M` dependent variables. If `err_y` is a scalar, the data set is
            taken to be homoscedastic (constant error). Otherwise, the length
            of `err_y` must equal the length of `y`. Default value is 0
            (noiseless observations).
        channels : dict or array-like (`M`, `N`)
            Keys to logically group points into "channels" along each dimension
            of `X`. If not passed, channels are based simply on which points
            have equal values in `X`. If only certain dimensions have groupings
            other than the simple default equality conditions, then you can
            pass a dict with integer keys in the interval [0, `X_dim`-1] whose
            values are the arrays of length `M` indicating the channels.
            Otherwise, you can pass in a full (`M`, `N`) array.
        
        Raises
        ------
        ValueError
            Bad shapes for any of the inputs, negative values for `err_y` or `n`.
        """
        # Verify y has only one non-trivial dimension:
        y = scipy.atleast_1d(scipy.asarray(y, dtype=float))
        if y.ndim != 1:
            raise ValueError(
                "Dependent variables y must have only one dimension! Shape of y "
                "given is %s" % (y.shape,)
            )

        #from IPython import embed
        #embed()

        # FS: some new error with nan's in err_y...
        # Should be fine to set those nan's to 0:
        err_y[scipy.isnan(err_y)] = 0.0
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError(
                    "When using array-like err_y, shape must match shape of y! "
                    "Shape of err_y given is %s, shape of y given is %s."
                    % (err_y.shape, y.shape)
                )
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar independent variable or convert array input into matrix.
        X = scipy.atleast_2d(scipy.asarray(X, dtype=float))
        # Correct single-dimension inputs:
        if self.X_dim == 1 and X.shape[0] == 1:
            X = X.T
        if X.shape != (len(y), self.X_dim):
            raise ValueError(
                "Shape of independent variables must be (len(y), self.X_dim)! "
                "X given has shape %s, shape of y is %s and X_dim=%d."
                % (X.shape, y.shape, self.X_dim)
            )
        
        # Process uncertainty in X:
        try:
            iter(err_X)
        except TypeError:
            err_X = err_X * scipy.ones_like(X, dtype=float)
        else:
            err_X = scipy.asarray(err_X, dtype=float)
            # TODO: Steal this idiom for handling n in gptools!
            if err_X.ndim == 1 and self.X_dim != 1:
                err_X = scipy.tile(err_X, (X.shape[0], 1))
        err_X = scipy.atleast_2d(scipy.asarray(err_X, dtype=float))
        if self.X_dim == 1 and err_X.shape[0] == 1:
            err_X = err_X.T
        if err_X.shape != X.shape:
            raise ValueError(
                "Shape of uncertainties on independent variables must be "
                "(len(y), self.X_dim)! X given has shape %s, shape of y is %s "
                "and X_dim=%d." % (X.shape, y.shape, self.X_dim)
            )
        
        if (err_X < 0).any():
            raise ValueError("All elements of err_X must be non-negative!")
        
        # Process channel flags:
        if channels is None:
            channels = scipy.tile(scipy.arange(0, len(y)), (X.shape[1], 1)).T
            # channels = scipy.copy(X)
        else:
            if isinstance(channels, dict):
                d_channels = channels
                channels = scipy.tile(scipy.arange(0, len(y)), (X.shape[1], 1)).T
                # channels = scipy.copy(X)
                for idx in d_channels:
                    channels[:, idx] = d_channels[idx]
            else:
                channels = scipy.asarray(channels)
                if channels.shape != (len(y), X.shape[1]):
                    raise ValueError("Shape of channels and X must be the same!")
        
        if self.X is None:
            self.X = X
        else:
            self.X = scipy.vstack((self.X, X))
        if self.channels is None:
            self.channels = channels
        else:
            self.channels = scipy.vstack((self.channels, channels))
        if self.err_X is None:
            self.err_X = err_X
        else:
            self.err_X = scipy.vstack((self.err_X, err_X))
        self.y = scipy.append(self.y, y)
        self.err_y = scipy.append(self.err_y, err_y)
        
        if self.gp is not None:
            self.gp.add_data(X, y, err_y=err_y)


    def add_profile(self, other):
        """Absorbs the data from one profile object.
        
        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        if self.X_dim != other.X_dim:
            raise ValueError(
                "When merging profiles, X_dim must be equal between the two "
                "profiles!"
            )
        if self.y_units != other.y_units:
            raise ValueError("When merging profiles, the y_units must agree!")
        if self.X_units != other.X_units:
            raise ValueError("When merging profiles, the X_units must agree!")
        if len(other.y) > 0:
            # Modify the channels of self.channels to avoid clashes:
            if other.channels is not None and self.channels is not None:
                self.channels = (
                    self.channels - self.channels.min(axis=0) +
                    other.channels.max(axis=0) + 1
                )
            self.add_data(other.X, other.y, err_X=other.err_X, err_y=other.err_y,
                          channels=other.channels)
        
        if len(other.transformed) > 0:
            self.transformed = scipy.append(self.transformed, other.transformed)


    def remove_points(self, conditional):
        """Remove points where conditional is True.
        
        Note that this does NOT remove anything from the GP -- you either need
        to call :py:meth:`create_gp` again or act manually on the :py:attr:`gp`
        attribute.
        
        Also note that this does not include any provision for removing points
        that represent linearly-transformed quantities -- you will need to
        operate directly on :py:attr:`transformed` to remove such points.
        
        Parameters
        ----------
        conditional : array-like of bool, (`M`,)
            Array of booleans corresponding to each entry in `y`. Where an
            entry is True, that value will be removed.
        
        Returns
        -------
        X_bad : matrix
            Input values of the bad points.
        y_bad : array
            Bad values.
        err_X_bad : array
            Uncertainties on the abcissa of the bad values.
        err_y_bad : array
            Uncertainties on the bad values.
        """
        idxs = ~conditional
        
        y_bad = self.y[conditional]
        X_bad = self.X[conditional, :]
        err_y_bad = self.err_y[conditional]
        err_X_bad = self.err_X[conditional, :]
        
        self.y = self.y[idxs]
        self.X = self.X[idxs, :]
        self.err_y = self.err_y[idxs]
        self.err_X = self.err_X[idxs, :]
        self.channels = self.channels[idxs, :]
        
        # Cause other methods to fail gracefully if this causes all pointlike
        # data to be removed:
        if len(self.y) == 0:
            self.X = None
            self.err_X = None
        
        return (X_bad, y_bad, err_X_bad, err_y_bad)

    def drop_axis(self, axis):
        """Drops a selected axis from `X`.
        
        Parameters
        ----------
        axis : int
            The index of the axis to drop.
        """
        if self.X_dim == 1:
            raise ValueError("Can't drop axis from a univariate profile!")
        self.X_dim -= 1
        if self.X is not None:
            self.channels = scipy.delete(self.channels, axis, axis=1)
            self.X = scipy.delete(self.X, axis, axis=1)
            self.err_X = scipy.delete(self.err_X, axis, axis=1)
        self.X_labels.pop(axis)
        self.X_units.pop(axis)
        
        for p in self.transformed:
            p.X = scipy.delete(p.X, axis, axis=2)
            p.err_X = scipy.delete(p.err_X, axis, axis=2)

    def plot_data(self, ax=None, label_axes=True, **kwargs):
        """Plot the data stored in this Profile. Only works for X_dim = 1 or 2.
        
        Parameters
        ----------
        ax : axis instance, optional
            Axis to plot the result on. If no axis is passed, one is created.
            If the string 'gca' is passed, the current axis (from plt.gca())
            is used. If X_dim = 2, the axis must be 3d.
        label_axes : bool, optional
            If True, the axes will be labelled with strings constructed from
            the labels and units set when creating the Profile instance.
            Default is True (label axes).
        **kwargs : extra plotting arguments, optional
            Extra arguments that are passed to errorbar/errorbar3d.
        
        Returns
        -------
        The axis instance used.
        """
        if self.X is not None:
            if self.X_dim > 2:
                raise ValueError("Plotting is not supported for X_dim > 2!")
            if ax is None:
                f = plt.figure()
                if self.X_dim == 1:
                    ax = f.add_subplot(1, 1, 1)
                elif self.X_dim == 2:
                    ax = f.add_subplot(111, projection='3d')
            elif ax == 'gca':
                ax = plt.gca()
            
            if 'label' not in kwargs:
                kwargs['label'] = self.y_label
            
            if 'fmt' not in kwargs and 'marker' not in kwargs:
                kwargs['fmt'] = 'o'
            
            if self.X_dim == 1:
                ax.errorbar(self.X.ravel(), self.y,
                            yerr=self.err_y, xerr=self.err_X.flatten(),
                            **kwargs)
                if label_axes:
                    ax.set_xlabel(
                        "%s [%s]" % (self.X_labels[0], self.X_units[0],) if self.X_units[0]
                        else self.X_labels[0]
                    )
                    ax.set_ylabel(
                        "%s [%s]" % (self.y_label, self.y_units,) if self.y_units
                        else self.y_label
                    )
            elif self.X_dim == 2:
                errorbar3d(ax, self.X[:, 0], self.X[:, 1], self.y,
                           xerr=self.err_X[:, 0], yerr=self.err_X[:, 1], zerr=self.err_y,
                           **kwargs)
                if label_axes:
                    ax.set_xlabel(
                        "%s [%s]" % (self.X_labels[0], self.X_units[0],) if self.X_units[0]
                        else self.X_labels[0]
                    )
                    ax.set_ylabel(
                        "%s [%s]" % (self.X_labels[1], self.X_units[1],) if self.X_units[1]
                        else self.X_labels[1]
                    )
                    ax.set_zlabel(
                        "%s [%s]" % (self.y_label, self.y_units,) if self.y_units
                        else self.y_label
                    )
            
            return ax

class BivariatePlasmaProfile(Profile):
    """Class to represent bivariate (y=f(t, psi)) plasma data.
    
    The first column of `X` is always time. If the abscissa is 'RZ', then the
    second column is `R` and the third is `Z`. Otherwise the second column is
    the desired abscissa (psinorm, etc.).
    """

    def remake_efit_tree(self):
        """Remake the EFIT tree.
        
        This is needed since EFIT tree instances aren't pickleable yet, so to
        store a :py:class:`BivariatePlasmaProfile` in a pickle file, you must
        delete the EFIT tree.
        """
        self.efit_tree = CModEFITTree(self.shot)

    def convert_abscissa(self, new_abscissa, drop_nan=True, ddof=1):
        """Convert the internal representation of the abscissa to new coordinates.
        
        The target abcissae are what are supported by `rho2rho` from the
        `eqtools` package. Namely,
        
            ======= ========================
            psinorm Normalized poloidal flux
            phinorm Normalized toroidal flux
            volnorm Normalized volume
            Rmid    Midplane major radius
            r/a     Normalized minor radius
            ======= ========================
        
        Additionally, each valid option may be prepended with 'sqrt' to return
        the square root of the desired normalized unit.
        
        Parameters
        ----------
        new_abscissa : str
            The new abscissa to convert to. Valid options are defined above.
        drop_nan : bool, optional
            Set this to True to drop any elements whose value is NaN following
            the conversion. Default is True (drop NaN elements).
        ddof : int, optional
            Degree of freedom correction to use when time-averaging a conversion.
        """
        if self.abscissa == new_abscissa:
            return
        elif self.X_dim == 1 or (self.X_dim == 2 and self.abscissa == 'RZ'):
            if self.abscissa.startswith('sqrt') and self.abscissa[4:] == new_abscissa:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 0], 2)
                    # Approximate form from uncertainty propagation:
                    err_new_rho = self.err_X[:, 0] * 2 * self.X[:, 0]
                
                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 0] = scipy.power(p.X[:, :, 0], 2)
                    p.err_X[:, :, 0] = p.err_X[:, :, 0] * 2 * p.X[:, :, 0]
            elif new_abscissa.startswith('sqrt') and self.abscissa == new_abscissa[4:]:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 0], 0.5)
                    # Approximate form from uncertainty propagation:
                    err_new_rho = self.err_X[:, 0] / (2 * scipy.sqrt(self.X[:, 0]))
                
                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 0] = scipy.power(p.X[:, :, 0], 0.5)
                    p.err_X[:, :, 0] = p.err_X[:, :, 0]  / (2 * scipy.sqrt(p.X[:, :, 0]))
            else:
                times = self._get_efit_times_to_average()
                
                if self.abscissa == 'RZ':
                    if self.X is not None:
                        new_rhos = self.efit_tree.rz2rho(
                            new_abscissa,
                            self.X[:, 0],
                            self.X[:, 1],
                            times,
                            each_t=True
                        )
                        self.channels = self.channels[:, 0:1]
                    self.X_dim = 1
                    
                    # Handle transformed quantities:
                    for p in self.transformed:
                        new_rhos = self.efit_tree.rz2rho(
                            new_abscissa,
                            p.X[:, :, 0],
                            p.X[:, :, 1],
                            times,
                            each_t=True
                        )
                        p.X = scipy.delete(p.X, 1, axis=2)
                        p.err_X = scipy.delete(p.err_X, 1, axis=2)
                        p.X[:, :, 0] = scipy.atleast_3d(scipy.mean(new_rhos, axis=0))
                        p.err_X[:, :, 0] = scipy.atleast_3d(scipy.std(new_rhos, axis=0, ddof=ddof))
                        p.err_X[scipy.isnan(p.err_X)] = 0
                else:
                    if self.X is not None:
                        new_rhos = self.efit_tree.rho2rho(
                            self.abscissa,
                            new_abscissa,
                            self.X[:, 0],
                            times,
                            each_t=True
                        )
                    
                    # Handle transformed quantities:
                    for p in self.transformed:
                        new_rhos = self.efit_tree.rho2rho(
                            self.abscissa,
                            new_abscissa,
                            p.X[:, :, 0],
                            times,
                            each_t=True
                        )
                        p.X[:, :, 0] = scipy.atleast_3d(scipy.mean(new_rhos, axis=0))
                        p.err_X[:, :, 0] = scipy.atleast_3d(scipy.std(new_rhos, axis=0, ddof=ddof))
                        p.err_X[scipy.isnan(p.err_X)] = 0
                if self.X is not None:
                    new_rho = scipy.mean(new_rhos, axis=0)
                    err_new_rho = scipy.std(new_rhos, axis=0, ddof=ddof)
                    err_new_rho[scipy.isnan(err_new_rho)] = 0
            
            if self.X is not None:
                self.X = scipy.atleast_2d(new_rho).T
                self.err_X = scipy.atleast_2d(err_new_rho).T
            self.X_labels = [_X_label_mapping[new_abscissa]]
            self.X_units = [_X_unit_mapping[new_abscissa]]
        else:
            if self.abscissa.startswith('sqrt') and self.abscissa[4:] == new_abscissa:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 1], 2)
                
                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = scipy.power(p.X[:, :, 1], 2)
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            elif new_abscissa.startswith('sqrt') and self.abscissa == new_abscissa[4:]:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 1], 0.5)
                
                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = scipy.power(p.X[:, :, 1], 0.5)
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            elif self.abscissa == 'RZ':
                # Need to handle this case separately because of the extra column:
                if self.X is not None:
                    new_rho = self.efit_tree.rz2rho(
                        new_abscissa,
                        self.X[:, 1],
                        self.X[:, 2],
                        self.X[:, 0],
                        each_t=False
                    )
                    self.channels = self.channels[:, 0:2]
                self.X_dim = 2
                
                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = self.efit_tree.rz2rho(
                        new_abscissa,
                        p.X[:, :, 1],
                        p.X[:, :, 2],
                        p.X[:, :, 0],
                        each_t=False
                    )
                    p.X = scipy.delete(p.X, 2, axis=2)
                    p.err_X = scipy.delete(p.err_X, 2, axis=2)
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            else:
                if self.X is not None:
                    new_rho = self.efit_tree.rho2rho(
                        self.abscissa,
                        new_abscissa,
                        self.X[:, 1],
                        self.X[:, 0],
                        each_t=False
                    )
                
                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = self.efit_tree.rho2rho(
                        self.abscissa,
                        new_abscissa,
                        p.X[:, :, 1],
                        p.X[:, :, 0],
                        each_t=False
                    )
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            
            if self.X is not None:
                err_new_rho = scipy.zeros_like(self.X[:, 0])
            
                self.X = scipy.hstack((
                    scipy.atleast_2d(self.X[:, 0]).T,
                    scipy.atleast_2d(new_rho).T
                ))
                self.err_X = scipy.hstack((
                    scipy.atleast_2d(self.err_X[:, 0]).T,
                    scipy.atleast_2d(err_new_rho).T
                ))
            
            self.X_labels = [self.X_labels[0], _X_label_mapping[new_abscissa]]
            self.X_units = [self.X_units[0], _X_unit_mapping[new_abscissa]]
        self.abscissa = new_abscissa
        if drop_nan and self.X is not None:
            self.remove_points(scipy.isnan(self.X).any(axis=1))

    def add_profile(self, other):
        """Absorbs the data from another profile object.
        
        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        # Warn about merging profiles from different shots:
        if self.shot != other.shot:
            warnings.warn("Merging data from two different shots: %d and %d"
                          % (self.shot, other.shot,))
        other.convert_abscissa(self.abscissa)
        # Split off the diagnostic description when merging profiles:
        super(BivariatePlasmaProfile, self).add_profile(other)
        if self.y_label != other.y_label:
            self.y_label = self.y_label.split(', ')[0]

    def drop_axis(self, axis):
        """Drops a selected axis from `X`.
        
        Parameters
        ----------
        axis : int
            The index of the axis to drop.
        """
        if self.X_labels[axis] == '$t$':
            if self.X is not None:
                self.t_min = self.X[:, 0].min()
                self.t_max = self.X[:, 0].max()
            if len(self.transformed) > 0:
                t_min_T = min([p.X[:, :, 0].min() for p in self.transformed])
                t_max_T = max([p.X[:, :, 0].max() for p in self.transformed])
                if self.X is None:
                    self.t_min = t_min_T
                    self.t_max = t_max_T
                else:
                    self.t_min = min(self.t_min, t_min_T)
                    self.t_max = max(self.t_max, t_max_T)
        super(BivariatePlasmaProfile, self).drop_axis(axis)


class Channel(object):
    """Class to store data from a single channel.
    
    This is particularly useful for storing linearly transformed data, but
    should work for general data just as well.
    
    Parameters
    ----------
    X : array, (`M`, `N`, `D`)
        Abscissa values to use.
    y : array, (`M`,)
        Data values.
    err_X : array, same shape as `X`
        Uncertainty in `X`.
    err_y : array, (`M`,)
        Uncertainty in data.
    T : array, (`M`, `N`), optional
        Linear transform to get from latent variables to data in `y`. Default is
        that `y` represents untransformed data.
    y_label : str, optional
        Label for the `y` data. Default is empty string.
    y_units : str, optional
        Units of the `y` data. Default is empty string.
    """
    def __init__(self, X, y, err_X=0, err_y=0, T=None, y_label='', y_units=''):
        self.y_label = y_label
        self.y_units = y_units
        # Verify y has only one non-trivial dimension:
        y = scipy.atleast_1d(scipy.asarray(y, dtype=float))
        if y.ndim != 1:
            raise ValueError(
                "Dependent variables y must have only one dimension! Shape of y "
                "given is %s" % (y.shape,)
            )
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError(
                    "When using array-like err_y, shape must match shape of y! "
                    "Shape of err_y given is %s, shape of y given is %s."
                    % (err_y.shape, y.shape)
                )
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar independent variable or convert array input into matrix.
        X = scipy.atleast_3d(scipy.asarray(X, dtype=float))
        if T is None and X.shape[0] != len(y):
            raise ValueError(
                "Shape of independent variables must be (len(y), D)! "
                "X given has shape %s, shape of y is %s."
                % (X.shape, y.shape,)
            )
        
        if T is not None:
            # Promote T if it is a single observation:
            T = scipy.atleast_2d(scipy.asarray(T, dtype=float))
            if T.ndim != 2:
                raise ValueError("T must have exactly 2 dimensions!")
            if T.shape[0] != len(y):
                raise ValueError("Length of first dimension of T must match length of y!")
            if T.shape[1] != X.shape[1]:
                raise ValueError("Second dimension of T must match second dimension of X!")
        else:
            T = scipy.eye(len(y))
        
        # Process uncertainty in X:
        try:
            iter(err_X)
        except TypeError:
            err_X = err_X * scipy.ones_like(X, dtype=float)
        else:
            err_X = scipy.asarray(err_X, dtype=float)
            if err_X.ndim == 1 and X.shape[2] != 1:
                err_X = scipy.tile(err_X, (X.shape[0], 1))
        err_X = scipy.atleast_2d(scipy.asarray(err_X, dtype=float))
        if err_X.shape != X.shape:
            raise ValueError(
                "Shape of uncertainties on independent variables must be "
                "(len(y), self.X_dim)! X given has shape %s, shape of y is %s."
                % (X.shape, y.shape,)
            )
        
        if (err_X < 0).any():
            raise ValueError("All elements of err_X must be non-negative!")
        
        self.X = X
        self.y = y
        self.err_X = err_X
        self.err_y = err_y
        self.T = T
    

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
        kwargs['efit_tree'] = CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == 'CTS':
            p_list.append(neCTS(shot, **kwargs))
        elif system == 'ETS':
            p_list.append(neETS(shot, **kwargs))
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
        p.efit_tree = CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'

    p.add_data(X, ne, err_y=err_ne, channels={1: channels, 2: channels})
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p



def neCTS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False, remove_zeros=True, Z_shift=0.0):
    """Returns a profile representing electron density from the core Thomson scattering system.
    
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
                               y_label=r'$n_e$, CTS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    try:
        N_ne_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:ne_rz')
        if (shot > 1030000000) & (shot < 1040000000):
            N_ne_TS_old = electrons.getNode(r'\electrons::top.yag.results.global.profile:ne_rz_t')
    except:
        N_ne_TS = electrons.getNode(r'\electrons::top.yag.results.global.profile:ne_rz_t')

    t_ne_TS = N_ne_TS.dim_of().data() # only need to get timebase for one of them
    if (shot > 1030000000) & (shot < 1040000000):
        ne_TS = scipy.concatenate((N_ne_TS.data() / 1e20, N_ne_TS_old.data() / 1e20))
    else:
        ne_TS = N_ne_TS.data() / 1e20

    try:
        N_dev_ne_TS = electrons.getNode(r'yag_new.results.profiles:ne_err')
        if (shot > 1030000000) & (shot < 1040000000):
            N_dev_ne_TS_old = electrons.getNode(r'yag.results.global.profile:ne_err_zt')
    except:
        N_dev_ne_TS = electrons.getNode(r'yag.results.global.profile:ne_err_zt')

    try:
        N_Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted')
        if (shot > 1030000000) & (shot < 1040000000):
            N_Z_CTS_old = electrons.getNode(r'yag.results.global.profile.z_sorted')
    except:
        N_Z_CTS = electrons.getNode(r'yag.results.global.profile.z_sorted')

    if (shot > 1030000000) & (shot < 1040000000):
        dev_ne_TS = scipy.concatenate((N_dev_ne_TS.data() / 1e20, N_dev_ne_TS_old.data() / 1e20))
        Z_CTS = scipy.concatenate((N_Z_CTS.data() + Z_shift, N_Z_CTS_old.data() + Z_shift))
    else:
        dev_ne_TS = N_dev_ne_TS.data() / 1e20
        Z_CTS = N_Z_CTS.data() + Z_shift

    R_CTS = (electrons.getNode(r'yag.results.param:r').data() * scipy.ones_like(Z_CTS))
    channels = list(range(0, len(Z_CTS)))
    
    t_grid, Z_grid = scipy.meshgrid(t_ne_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_ne_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_ne_TS, channels)
    
    ne = ne_TS.flatten()
    err_ne = dev_ne_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, ne, err_y=err_ne, channels={1: channels, 2: channels})
    
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)
    
    if remove_edge:
        p.remove_edge_points()
    
    return p

def neTS(shot, **kwargs):
    """Returns a profile representing electron density from both the core and edge Thomson scattering systems.
    """
    return ne(shot, include=['CTS', 'ETS'], **kwargs)


def Te(shot, include=['CTS', 'ETS', 'FRCECE', 'GPC2', 'GPC', 'Mic'], FRCECE_rate='s',
       FRCECE_cutoff=0.15, GPC_cutoff=0.15, remove_ECE_edge=True, **kwargs):
    """Returns a profile representing electron temperature from the Thomson scattering and ECE systems.

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:

            ====== ===============================
            CTS    Core Thomson scattering
            ETS    Edge Thomson scattering
            FRCECE FRC electron cyclotron emission
            GPC    Grating polychromator
            GPC2   Grating polychromator 2
            ====== ===============================

        The default is to include all data sources.
    FRCECE_rate : {'s', 'f'}, optional
        Which timebase to use for FRCECE -- the fast or slow data. Default is
        's' (slow).
    FRCECE_cutoff : float, optional
        The cutoff value for eliminating cut-off points from FRCECE. All points
        with values less than this will be discarded. Default is 0.15.
    GPC_cutoff : float, optional
        The cutoff value for eliminating cut-off points from GPC. All points
        with values less than this will be discarded. Default is 0.15.
    remove_ECE_edge : bool, optional
        If True, the points outside of the LCFS for the ECE diagnostics will be
        removed. Note that this overrides remove_edge, if present, in kwargs.
        Furthermore, this may lead to abscissa being converted to psinorm if an
        incompatible option was used.
    **kwargs
        All remaining parameters are passed to the individual loading methods.
    """
    if 'electrons' not in kwargs:
        kwargs['electrons'] = MDSplus.Tree('electrons', shot)
    if 'efit_tree' not in kwargs:
        kwargs['efit_tree'] = CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == 'CTS':
            p_list.append(TeCTS(shot, **kwargs))
        elif system == 'ETS':
            p_list.append(TeETS(shot, **kwargs))
        else:
            raise ValueError("Unknown profile '%s'." % (system,))
    
    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)
    
    return p


def TeETS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False, remove_zeros=False, Z_shift=0.0):
    """Returns a profile representing electron temperature from the edge Thomson scattering system.

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
        If True, will remove points that are identically zero. Default is False
        (keep zero points). This was added because clearly bad points aren't
        always flagged with a sentinel value of errorbar.
    Z_shift: float, optional
        The shift to apply to the vertical coordinate, sometimes needed to
        correct EFIT mapping. Default is 0.0.
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='keV',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label=r'$T_e$, ETS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    N_Te_TS = electrons.getNode(r'yag_edgets.results:te')

    t_Te_TS = N_Te_TS.dim_of().data()
    Te_TS = N_Te_TS.data() / 1e3
    dev_Te_TS = electrons.getNode(r'yag_edgets.results:te:error').data() / 1e3
    
    Z_CTS = electrons.getNode(r'yag_edgets.data:fiber_z').data() + Z_shift
    R_CTS = (electrons.getNode(r'yag.results.param:r').data() *
             scipy.ones_like(Z_CTS))
    channels = list(range(0, len(Z_CTS)))
    
    t_grid, Z_grid = scipy.meshgrid(t_Te_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_Te_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_Te_TS, channels)
    
    Te = Te_TS.flatten()
    err_Te = dev_Te_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, Te, err_y=err_Te, channels={1: channels, 2: channels})
    
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)
    
    if remove_edge:
        p.remove_edge_points()
    
    return p

def TeCTS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False, remove_zeros=True, Z_shift=0.0):
    """Returns a profile representing electron temperature from the core Thomson scattering system.
    
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
                               y_units='keV',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label=r'$T_e$, CTS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    try:
        N_Te_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:Te_rz')
        if (shot > 1030000000) & (shot < 1040000000):
            N_Te_TS_old = electrons.getNode(r'\electrons::top.yag.results.global.profile:Te_rz_t')
    except:
        N_Te_TS = electrons.getNode(r'\electrons::top.yag.results.global.profile:Te_rz_t')

    t_Te_TS = N_Te_TS.dim_of().data()
    if (shot > 1030000000) & (shot < 1040000000):
        Te_TS = scipy.concatenate((N_Te_TS.data(), N_Te_TS_old.data()))
    else:
        Te_TS = N_Te_TS.data()

    try:
        N_dev_Te_TS = electrons.getNode(r'yag_new.results.profiles:Te_err')
        if (shot > 1030000000) & (shot < 1040000000):
            N_dev_Te_TS_old = electrons.getNode(r'yag.results.global.profile:Te_err_zt')
    except:
        N_dev_Te_TS = electrons.getNode(r'yag.results.global.profile:Te_err_zt')

    try:
        N_Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted')
        if (shot > 1030000000) & (shot < 1040000000):
            N_Z_CTS_old = electrons.getNode(r'yag.results.global.profile.z_sorted')
    except:
        N_Z_CTS = electrons.getNode(r'yag.results.global.profile.z_sorted')

    if (shot > 1030000000) & (shot < 1040000000):
        dev_Te_TS = scipy.concatenate((N_dev_Te_TS.data(), N_dev_Te_TS_old.data()))
        Z_CTS = scipy.concatenate((N_Z_CTS.data() + Z_shift, N_Z_CTS_old.data() + Z_shift))
    else:
        dev_Te_TS = N_dev_Te_TS.data()
        Z_CTS = N_Z_CTS.data() + Z_shift

    R_CTS = (electrons.getNode(r'yag.results.param:r').data() *
             scipy.ones_like(Z_CTS))
    channels = list(range(0, len(Z_CTS)))
 
    t_grid, Z_grid = scipy.meshgrid(t_Te_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_Te_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_Te_TS, channels)
    
    Te = Te_TS.flatten()
    err_Te = dev_Te_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, Te, err_y=err_Te, channels={1: channels, 2: channels})
    
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)
    
    if remove_edge:
        p.remove_edge_points()
    
    return p

def TeTS(shot, **kwargs):
    """Returns a profile representing electron temperature data from the Thomson scattering system.
    
    Includes both core and edge system.
    """
    return Te(shot, include=['CTS', 'ETS'], **kwargs)



##########################

### stuff from eqtools ###

##########################

import warnings
import re
from past.utils import old_div

"""The following is a dictionary to implement length unit conversions. The first
key is the unit are converting FROM, the second the unit you are converting TO.
Supports: m, cm, mm, in, ft, yd, smoot, cubit, hand
"""
_length_conversion = {'m': {'m': 1.0,
                            'cm': 100.0,
                            'mm': 1000.0,
                            'in': 39.37,
                            'ft': old_div(39.37, 12.0),
                            'yd': old_div(39.37, (12.0 * 3.0)),
                            'smoot': old_div(39.37, 67.0),
                            'cubit': old_div(39.37, 18.0),
                            'hand': old_div(39.37, 4.0)},
                      'cm': {'m': 0.01,
                             'cm': 1.0,
                             'mm': 10.0,
                             'in': old_div(39.37, 100.0),
                             'ft': old_div(39.37, (100.0 * 12.0)),
                             'yd': old_div(39.37, (100.0 * 12.0 * 3.0)),
                             'smoot': old_div(39.37, (100.0 * 67.0)),
                             'cubit': old_div(39.37, (100.0 * 18.0)),
                             'hand': old_div(39.37, (100.0 * 4.0))},
                      'mm': {'m': 0.001,
                             'cm': 0.1,
                             'mm': 1.0,
                             'in': old_div(39.37, 1000.0),
                             'ft': old_div(39.37, (1000.0 * 12.0)),
                             'yd': old_div(39.37, (1000.0 * 12.0 * 3.0)),
                             'smoot': old_div(39.37, (1000.0 * 67.0)),
                             'cubit': old_div(39.37, (1000.0 * 18.0)),
                             'hand': old_div(39.37, (1000.0 * 4.0))},
                      'in': {'m': old_div(1.0, 39.37),
                             'cm': old_div(100.0, 39.37),
                             'mm': old_div(1000.0, 39.37),
                             'in': 1.0,
                             'ft': old_div(1.0, 12.0),
                             'yd': old_div(1.0, (12.0 * 3.0)),
                             'smoot': old_div(1.0, 67.0),
                             'cubit': old_div(1.0, 18.0),
                             'hand': old_div(1.0, 4.0)},
                      'ft': {'m': old_div(12.0, 39.37),
                             'cm': 12.0 * 100.0 / 39.37,
                             'mm': 12.0 * 1000.0 / 39.37,
                             'in': 12.0,
                             'ft': 1.0,
                             'yd': old_div(1.0, 3.0),
                             'smoot': old_div(12.0, 67.0),
                             'cubit': old_div(12.0, 18.0),
                             'hand': old_div(12.0, 4.0)},
                      'yd': {'m': 3.0 * 12.0 / 39.37,
                             'cm': 3.0 * 12.0 * 100.0 / 39.37,
                             'mm': 3.0 * 12.0 * 1000.0 / 39.37,
                             'in': 3.0 * 12.0,
                             'ft': 3.0,
                             'yd': 1.0,
                             'smoot': 3.0 * 12.0 / 67.0,
                             'cubit': 3.0 * 12.0 / 18.0,
                             'hand': 3.0 * 12.0 / 4.0},
                      'smoot': {'m': old_div(67.0, 39.37),
                                'cm': 67.0 * 100.0 / 39.37,
                                'mm': 67.0 * 1000.0 / 39.37,
                                'in': 67.0,
                                'ft': old_div(67.0, 12.0),
                                'yd': old_div(67.0, (12.0 * 3.0)),
                                'smoot': 1.0,
                                'cubit': old_div(67.0, 18.0),
                                'hand': old_div(67.0, 4.0)},
                      'cubit': {'m': old_div(18.0, 39.37),
                                'cm': 18.0 * 100.0 / 39.37,
                                'mm': 18.0 * 1000.0 / 39.37,
                                'in': 18.0,
                                'ft': old_div(18.0, 12.0),
                                'yd': old_div(18.0, (12.0 * 3.0)),
                                'smoot': old_div(18.0, 67.0),
                                'cubit': 1.0,
                                'hand': old_div(18.0, 4.0)},
                      'hand': {'m': old_div(4.0, 39.37),
                               'cm': 4.0 * 100.0 / 39.37,
                               'mm': 4.0 * 1000.0 / 39.37,
                               'in': 4.0,
                               'ft': old_div(4.0, 12.0),
                               'yd': old_div(4.0, (12.0 * 3.0)),
                               'smoot': old_div(4.0, 67.0),
                               'cubit': old_div(4.0, 18.0),
                               'hand': 1.0}}

class Equilibrium(object):
    """Abstract class of data handling object for magnetic reconstruction outputs.
    
    Defines the mapping routines and method fingerprints necessary. Each
    variable or set of variables is recovered with a corresponding getter method.
    Essential data for mapping are pulled on initialization (psirz grid, for
    example) to frontload overhead. Additional data are pulled at the first
    request and stored for subsequent usage.
    
    .. note:: This abstract class should not be used directly. Device- and code-
        specific subclasses are set up to account for inter-device/-code
        differences in data storage.
    
    Keyword Args:
        length_unit (String): Sets the base unit used for any quantity whose
            dimensions are length to any power. Valid options are:
            
                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================
            
            Default is 'm' (all units taken and returned in meters).
        tspline (Boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor interpolation.
            Tricubic spline interpolation requires at least four complete
            equilibria at different times. It is also assumed that they are
            functionally correlated, and that parameters do not vary out of
            their boundaries (derivative = 0 boundary condition). Default is
            False (use nearest-neighbor interpolation).
        monotonic (Boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
        verbose (Boolean): Allows or blocks console readout during operation.
            Defaults to True, displaying useful information for the user. Set to
            False for quiet usage or to avoid console clutter for multiple
            instances.
    
    Raises:
        ValueError: If `length_unit` is not a valid unit specifier.
        ValueError: If `tspline` is True but module trispline did not load
            successfully.
    """
    def __init__(self, length_unit='m', tspline=False, monotonic=True, verbose=True):
        if length_unit != 'default' and not (length_unit in _length_conversion):
            raise ValueError("Unit '%s' not a valid unit specifier!" % length_unit)
        else:
            self._length_unit = length_unit
        
        self._tricubic = bool(tspline)
        self._monotonic = bool(monotonic)
        self._verbose = bool(verbose)
            
        # These are indexes of splines, and become higher dimensional splines
        # with the setting of the tspline keyword.
        self._psiOfRZSpline = {}
        self._phiNormSpline = {}
        self._volNormSpline = {}
        self._RmidSpline = {}
        self._magRSpline = {}
        self._magZSpline = {}
        self._RmidOutSpline = {}
        self._psiOfPsi0Spline = {}
        self._psiOfLCFSSpline = {}
        self._RmidToPsiNormSpline = {}
        self._phiNormToPsiNormSpline = {}
        self._volNormToPsiNormSpline = {}
        self._AOutSpline = {}
        self._qSpline = {}
        self._FSpline = {}
        self._FToPsinormSpline = {}
        self._FFPrimeSpline = {}
        self._pSpline = {}
        self._pPrimeSpline = {}
        self._vSpline = {}
        self._BtVacSpline = {}
  
    ####################
    # Mapping routines #
    ####################
    
    def rho2rho(self, origin, destination, *args, **kwargs):
        r"""Convert from one coordinate to another.
        
        Args:
            origin (String): Indicates which coordinates the data are given in.
                Valid options are:
                
                    ======= ========================
                    RZ      R,Z coordinates
                    psinorm Normalized poloidal flux
                    phinorm Normalized toroidal flux
                    volnorm Normalized volume
                    Rmid    Midplane major radius
                    r/a     Normalized minor radius
                    ======= ========================
                
                Additionally, each valid option may be prepended with 'sqrt'
                to specify the square root of the desired unit.
            destination (String): Indicates which coordinates to convert to.
                Valid options are:
                
                    ======= =================================
                    psinorm Normalized poloidal flux
                    phinorm Normalized toroidal flux
                    volnorm Normalized volume
                    Rmid    Midplane major radius
                    r/a     Normalized minor radius
                    q       Safety factor
                    F       Flux function :math:`F=RB_{\phi}`
                    FFPrime Flux function :math:`FF'`
                    p       Pressure
                    pprime  Pressure gradient
                    v       Flux surface volume
                    ======= =================================
                
                Additionally, each valid option may be prepended with 'sqrt'
                to specify the square root of the desired unit.
            rho (Array-like or scalar float): Values of the starting coordinate
                to map to the new coordinate. Will be two arguments `R`, `Z` if
                `origin` is 'RZ'.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `rho`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `rho` (or the meshgrid of `R`
                and `Z` if `make_grid` is True).
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of `rho`. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `rho` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `rho` or be
                a scalar. Default is True (evaluate ALL `rho` at EACH element in
                `t`).
            make_grid (Boolean): Only applicable if `origin` is 'RZ'. Set to
                True to pass `R` and `Z` through :py:func:`scipy.meshgrid`
                before evaluating. If this is set to True, `R` and `Z` must each
                only have a single dimension, but can have different lengths.
                Default is False (do not form meshgrid).
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of Rmid when `destination` is Rmid. Default is False
                (return major radius, Rmid).            
            length_unit (String or 1): Length unit that quantities are
                given/returned in, as applicable. If a string is given, it must
                be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            
        Returns:
            `rho` or (`rho`, `time_idxs`)
        
            * **rho** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `rho`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Raises:
            ValueError: If `origin` is not one of the supported values.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value at r/a=0.6, t=0.26s::
            
                psi_val = Eq_instance.rho2rho('r/a', 'psinorm', 0.6, 0.26)
            
            Find psinorm values at r/a points 0.6 and 0.8 at the
            single time t=0.26s::
            
                psi_arr = Eq_instance.rho2rho('r/a', 'psinorm', [0.6, 0.8], 0.26)
            
            Find psinorm values at r/a of 0.6 at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.rho2rho('r/a', 'psinorm', 0.6, [0.2, 0.3])
            
            Find psinorm values at (r/a, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                psi_arr = Eq_instance.rho2rho('r/a', 'psinorm', [0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        
        if origin.startswith('sqrt'):
            args = list(args)
            args[0] = scipy.asarray(args[0])**2
            origin = origin[4:]
        
        if destination.startswith('sqrt'):
            kwargs['sqrt'] = True
            destination = destination[4:]
        
        if origin == 'RZ':
            return self.rz2rho(destination, *args, **kwargs)
        elif origin == 'Rmid':
            return self.rmid2rho(destination, *args, **kwargs)
        elif origin == 'r/a':
            return self.roa2rho(destination, *args, **kwargs)
        elif origin == 'psinorm':
            return self.psinorm2rho(destination, *args, **kwargs)
        elif origin == 'phinorm':
            return self.phinorm2rho(destination, *args, **kwargs)
        elif origin == 'volnorm':
            return self.volnorm2rho(destination, *args, **kwargs)
        else:
            raise ValueError("rho2rho: Unsupported origin coordinate method '%s'!" % origin)
    
    def rz2psi(self, R, Z, t, return_t=False, make_grid=False, each_t=True, length_unit=1):
        r"""Converts the passed R, Z, t arrays to psi (unnormalized poloidal flux) values.
        
        What is usually returned by EFIT is the stream function,
        :math:`\psi=\psi_p/(2\pi)` which has units of Wb/rad.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to poloidal flux. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to poloidal flux. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `R` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `Z` must
                have exactly one dimension.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If the `each_t` keyword is True, then `t` must be
                scalar or have exactly one dimension. If the `each_t` keyword is
                False, `t` must have the same shape as `R` and `Z` (or their
                meshgrid if `make_grid` is True).
        
        Keyword Args:
            each_t (Boolean): When True, the elements in `R`, `Z` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R` and
                `Z` or be a scalar. Default is True (evaluate ALL `R`, `Z` at
                EACH element in `t`).
            make_grid (Boolean): Set to True to pass `R` and `Z` through
                :py:func:`scipy.meshgrid` before evaluating. If this is set to
                True, `R` and `Z` must each only have a single dimension, but
                can have different lengths. Default is False (do not form
                meshgrid).
            length_unit (String or 1): Length unit that `R`, `Z` are given in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            
        Returns:
            `psi` or (`psi`, `time_idxs`)
            
            * **psi** (`Array or scalar float`) - The unnormalized poloidal
              flux. If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `psi` has this shape as well,
              unless the `make_grid` keyword was True, in which case `psi` has
              shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `psi`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psi value at R=0.6m, Z=0.0m, t=0.26s::
            
                psi_val = Eq_instance.rz2psi(0.6, 0, 0.26)
            
            Find psi values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully
            specified, even if the values are all the same::
            
                psi_arr = Eq_instance.rz2psi([0.6, 0.8], [0, 0], 0.26)
            
            Find psi values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.rz2psi(0.6, 0, [0.2, 0.3])
            
            Find psi values at (R, Z, t) points (0.6m, 0m, 0.2s) and
            (0.5m, 0.2m, 0.3s)::
            
                psi_arr = Eq_instance.rz2psi([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find psi values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                psi_mat = Eq_instance.rz2psi(R, Z, 0.2, make_grid=True)
        """
        
        # Check inputs and process into flat arrays with units of meters:
        R, Z, t, time_idxs, unique_idxs, single_time, single_val, original_shape = self._processRZt(
            R,
            Z,
            t,
            make_grid=make_grid,
            each_t=each_t,
            length_unit=length_unit,
            compute_unique=True
        )
        
        if self._tricubic:
            out_vals = scipy.reshape(
                self._getFluxTriSpline().ev(t, Z, R),
                original_shape
            )
        else:
            if single_time:
                out_vals = self._getFluxBiSpline(time_idxs[0]).ev(Z, R)
                if single_val:
                    out_vals = out_vals[0]
                else:
                    out_vals = scipy.reshape(out_vals, original_shape)
            elif each_t:
                out_vals = scipy.zeros(
                    scipy.concatenate(([len(time_idxs),], original_shape))
                )
                for idx, t_idx in enumerate(time_idxs):
                    out_vals[idx] = self._getFluxBiSpline(t_idx).ev(Z, R).reshape(original_shape)
            else:
                out_vals = scipy.zeros_like(t, dtype=float)
                for t_idx in unique_idxs:
                    t_mask = (time_idxs == t_idx)
                    out_vals[t_mask] = self._getFluxBiSpline(t_idx).ev(Z[t_mask], R[t_mask])
                out_vals = scipy.reshape(out_vals, original_shape)
        
        # Correct for current sign:
        out_vals = -1.0 * out_vals * self.getCurrentSign()
        
        if return_t:
            if self._tricubic:
                return out_vals, (t, single_time, single_val, original_shape)
            else:
                return out_vals, (time_idxs, unique_idxs, single_time, single_val, original_shape)
        else:
            return out_vals
    
    def rz2psinorm(self, R, Z, t, return_t=False, sqrt=False, make_grid=False,
                   each_t=True, length_unit=1):
        r"""Calculates the normalized poloidal flux at the given (R, Z, t).
        
        Uses the definition:
        
        .. math::
        
            \texttt{psi\_norm} = \frac{\psi - \psi(0)}{\psi(a) - \psi(0)}
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to psinorm. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to psinorm. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `R` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `Z` must
                have exactly one dimension.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If the `each_t` keyword is True, then `t` must be
                scalar or have exactly one dimension. If the `each_t` keyword is
                False, `t` must have the same shape as `R` and `Z` (or their
                meshgrid if `make_grid` is True).
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of psinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `R`, `Z` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R` and
                `Z` or be a scalar. Default is True (evaluate ALL `R`, `Z` at
                EACH element in `t`).
            make_grid (Boolean): Set to True to pass `R` and `Z` through
                :py:func:`scipy.meshgrid` before evaluating. If this is set to
                True, `R` and `Z` must each only have a single dimension, but
                can have different lengths. Default is False (do not form
                meshgrid).
            length_unit (String or 1): Length unit that `R`, `Z` are given in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `psinorm` or (`psinorm`, `time_idxs`)
            
            * **psinorm** (`Array or scalar float`) - The normalized poloidal
              flux. If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `psinorm` has this shape as well,
              unless the `make_grid` keyword was True, in which case `psinorm`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `psinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
       
        Examples:
            All assume that `Eq_instance` is a valid instance of the
            appropriate extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value at R=0.6m, Z=0.0m, t=0.26s::
            
                psi_val = Eq_instance.rz2psinorm(0.6, 0, 0.26)
            
            Find psinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                psi_arr = Eq_instance.rz2psinorm([0.6, 0.8], [0, 0], 0.26)
            
            Find psinorm values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.rz2psinorm(0.6, 0, [0.2, 0.3])
            
            Find psinorm values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                psi_arr = Eq_instance.rz2psinorm([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find psinorm values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                psi_mat = Eq_instance.rz2psinorm(R, Z, 0.2, make_grid=True)
        """
        psi, blob = self.rz2psi(
            R,
            Z,
            t,
            return_t=True,
            make_grid=make_grid,
            each_t=each_t,
            length_unit=length_unit
        )
        
        if self._tricubic:
            psi_boundary = self._getLCFSPsiSpline()(blob[0]).reshape(blob[-1])
            psi_0 = self._getPsi0Spline()(blob[0]).reshape(blob[-1])
        else:
            psi_boundary = self.getFluxLCFS()[blob[0]]
            psi_0 = self.getFluxAxis()[blob[0]]
            
            # If there is more than one time point, we need to expand these
            # arrays to be broadcastable:
            if not blob[-3]:
                if each_t:
                    for k in range(0, len(blob[-1])):
                        psi_boundary = scipy.expand_dims(psi_boundary, -1)
                        psi_0 = scipy.expand_dims(psi_0, -1)
                else:
                    psi_boundary = psi_boundary.reshape(blob[-1])
                    psi_0 = psi_0.reshape(blob[-1])
        
        psi_norm = old_div((psi - psi_0), (psi_boundary - psi_0))
        
        if sqrt:
            if psi_norm.ndim == 0:
                if psi_norm < 0.0:
                    psi_norm = 0.0
            else:
                scipy.place(psi_norm, psi_norm < 0, 0)
            out = scipy.sqrt(psi_norm)
        else:
            out = psi_norm
        
        # Unwrap single values to ensure least surprise:
        if blob[-2] and blob[-3] and not self._tricubic:
            out = out[0]
        
        if return_t:
            return out, blob
        else:
            return out
   
    def rz2rho(self, method, *args, **kwargs):
        r"""Convert the passed (R, Z, t) coordinates into one of several coordinates.
        
        Args:
            method (String): Indicates which coordinates to convert to. Valid
                options are:
                
                    ======= =================================
                    psinorm Normalized poloidal flux
                    phinorm Normalized toroidal flux
                    volnorm Normalized volume
                    Rmid    Midplane major radius
                    r/a     Normalized minor radius
                    q       Safety factor
                    F       Flux function :math:`F=RB_{\phi}`
                    FFPrime Flux function :math:`FF'`
                    p       Pressure
                    pprime  Pressure gradient
                    v       Flux surface volume
                    ======= =================================
                
                Additionally, each valid option may be prepended with 'sqrt'
                to specify the square root of the desired unit.
            R (Array-like or scalar float): Values of the radial coordinate to
                map to `rho`. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to `rho`. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `R` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `Z` must
                have exactly one dimension.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If the `each_t` keyword is True, then `t` must be
                scalar or have exactly one dimension. If the `each_t` keyword is
                False, `t` must have the same shape as `R` and `Z` (or their
                meshgrid if `make_grid` is True).
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of `rho`. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `R`, `Z` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R` and
                `Z` or be a scalar. Default is True (evaluate ALL `R`, `Z` at
                EACH element in `t`).
            make_grid (Boolean): Set to True to pass `R` and `Z` through
                :py:func:`scipy.meshgrid` before evaluating. If this is set to
                True, `R` and `Z` must each only have a single dimension, but
                can have different lengths. Default is False (do not form
                meshgrid).
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of Rmid when `destination` is Rmid. Default is False
                (return major radius, Rmid).            
            length_unit (String or 1): Length unit that `R`, `Z` are given in,
                AND that `Rmid` is returned in. If a string is given, it must
                be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            
        Returns:
            `rho` or (`rho`, `time_idxs`)
            
            * **rho** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `rho`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Raises:
            ValueError: If `method` is not one of the supported values.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value at R=0.6m, Z=0.0m, t=0.26s::
            
                psi_val = Eq_instance.rz2rho('psinorm', 0.6, 0, 0.26)
            
            Find psinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                psi_arr = Eq_instance.rz2rho('psinorm', [0.6, 0.8], [0, 0], 0.26)
            
            Find psinorm values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.rz2rho('psinorm', 0.6, 0, [0.2, 0.3])
            
            Find psinorm values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                psi_arr = Eq_instance.rz2rho('psinorm', [0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find psinorm values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                psi_mat = Eq_instance.rz2rho('psinorm', R, Z, 0.2, make_grid=True)
        """
        
        if method.startswith('sqrt'):
            kwargs['sqrt'] = True
            method = method[4:]
        
        if method == 'psinorm':
            return self.rz2psinorm(*args, **kwargs)
        elif method == 'phinorm':
            return self.rz2phinorm(*args, **kwargs)
        elif method == 'volnorm':
            return self.rz2volnorm(*args, **kwargs)
        elif method == 'Rmid':
            return self.rz2rmid(*args, **kwargs)
        elif method == 'r/a':
            kwargs['rho'] = True
            return self.rz2rmid(*args, **kwargs)
        elif method == 'q':
            return self.rz2q(*args, **kwargs)
        elif method == 'F':
            return self.rz2F(*args, **kwargs)
        elif method == 'FFPrime':
            return self.rz2FFPrime(*args, **kwargs)
        elif method == 'p':
            return self.rz2p(*args, **kwargs)
        elif method == 'pprime':
            return self.rz2pprime(*arg, **kwargs)
        elif method == 'v':
            return self.rz2v(*args, **kwargs)
        else:
            raise ValueError("rz2rho: Unsupported normalized coordinate method '%s'!" % method)
    
    def rmid2psinorm(self, R_mid, t, **kwargs):
        """Calculates the normalized poloidal flux corresponding to the passed R_mid (mapped outboard midplane major radius) values.
        
        Args:
            R_mid (Array-like or scalar float): Values of the outboard midplane
                major radius to map to psinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R_mid`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `R_mid`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of psinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `R_mid` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R_mid`
                or be a scalar. Default is True (evaluate ALL `R_mid` at EACH
                element in `t`).
            length_unit (String or 1): Length unit that `R_mid` is given in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `psinorm` or (`psinorm`, `time_idxs`)
            
            * **psinorm** (`Array or scalar float`) - Normalized poloidal flux.
              If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `psinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value for Rmid=0.7m, t=0.26s::
            
                psinorm_val = Eq_instance.rmid2psinorm(0.7, 0.26)
            
            Find psinorm values at R_mid values of 0.5m and 0.7m at the single time
            t=0.26s::
            
                psinorm_arr = Eq_instance.rmid2psinorm([0.5, 0.7], 0.26)
            
            Find psinorm values at R_mid=0.5m at times t=[0.2s, 0.3s]::
            
                psinorm_arr = Eq_instance.rmid2psinorm(0.5, [0.2, 0.3])
            
            Find psinorm values at (R_mid, t) points (0.6m, 0.2s) and (0.5m, 0.3s)::
            
                psinorm_arr = Eq_instance.rmid2psinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._psinorm2Quan(self._getRmidToPsiNormSpline, R_mid, t, check_space=True, **kwargs)
    
    def rmid2rho(self, method, R_mid, t, **kwargs):
        r"""Convert the passed (R_mid, t) coordinates into one of several coordinates.
        
        Args:
            method (String): Indicates which coordinates to convert to. Valid
                options are:
                
                    ======= =================================
                    psinorm Normalized poloidal flux
                    phinorm Normalized toroidal flux
                    volnorm Normalized volume
                    r/a     Normalized minor radius
                    F       Flux function :math:`F=RB_{\phi}`
                    FFPrime Flux function :math:`FF'`
                    p       Pressure
                    pprime  Pressure gradient
                    v       Flux surface volume
                    ======= =================================
                
                Additionally, each valid option may be prepended with 'sqrt'
                to specify the square root of the desired unit.
            R_mid (Array-like or scalar float): Values of the outboard midplane
                major radius to map to rho.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R_mid`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `R_mid`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of rho. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `R_mid` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R_mid`
                or be a scalar. Default is True (evaluate ALL `R_mid` at EACH
                element in `t`).
            length_unit (String or 1): Length unit that `R_mid` is given in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            
        Returns:
            `rho` or (`rho`, `time_idxs`)
        
            * **rho** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `rho`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value at R_mid=0.6m, t=0.26s::
            
                psi_val = Eq_instance.rmid2rho('psinorm', 0.6, 0.26)
            
            Find psinorm values at R_mid points 0.6m and 0.8m at the
            single time t=0.26s.::
            
                psi_arr = Eq_instance.rmid2rho('psinorm', [0.6, 0.8], 0.26)
            
            Find psinorm values at R_mid of 0.6m at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.rmid2rho('psinorm', 0.6, [0.2, 0.3])
            
            Find psinorm values at (R_mid, t) points (0.6m, 0.2s) and (0.5m, 0.3s)::
            
                psi_arr = Eq_instance.rmid2rho('psinorm', [0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        if method.startswith('sqrt'):
            kwargs['sqrt'] = True
            method = method[4:]
        
        if method == 'psinorm':
            return self.rmid2psinorm(R_mid, t, **kwargs)
        elif method == 'r/a':
            return self.rmid2roa(R_mid, t, **kwargs)
        elif method == 'phinorm':
            return self.rmid2phinorm(R_mid, t, **kwargs)
        elif method == 'volnorm':
            return self.rmid2volnorm(R_mid, t, **kwargs)
        elif method == 'q':
            return self.rmid2q(R_mid, t, **kwargs)
        elif method == 'F':
            return self.rmid2F(R_mid, t, **kwargs)
        elif method == 'FFPrime':
            return self.rmid2FFPrime(R_mid, t, **kwargs)
        elif method == 'p':
            return self.rmid2p(R_mid, t, **kwargs)
        elif method == 'pprime':
            return self.rmid2pprime(R_mid, t, **kwargs)
        elif method == 'v':
            return self.rmid2v(R_mid, t, **kwargs)
        else:
            # Default back to the old kuldge that wastes time in rz2psi:
            # TODO: This doesn't handle length units properly!
            Z_mid = self.getMagZSpline()(t)
            
            if kwargs.get('each_t', True):
                # Need to override the default in _processRZt, since we are doing
                # the shaping here:
                kwargs['each_t'] = False
                try:
                    iter(t)
                except TypeError:
                    # For a single t, there will only be a single value of Z_mid and
                    # we only need to make it have the same shape as R_mid. Note
                    # that ones_like appears to be clever enough to handle the case
                    # of a scalar R_mid.
                    Z_mid = Z_mid * scipy.ones_like(R_mid, dtype=float)
                else:
                    # For multiple t, we need to repeat R_mid for every t, then
                    # repeat the corresponding Z_mid that many times for each such
                    # entry.
                    t = scipy.asarray(t)
                    if t.ndim != 1:
                        raise ValueError("rmid2rho: When using the each_t keyword, "
                                         "t must have only one dimension.")
                    R_mid = scipy.tile(
                        R_mid,
                        scipy.concatenate(([len(t),], scipy.ones_like(scipy.shape(R_mid), dtype=float)))
                    )
                    # TODO: Is there a clever way to do this without a loop?
                    Z_mid_temp = scipy.ones_like(R_mid, dtype=float)
                    t_temp = scipy.ones_like(R_mid, dtype=float)
                    for k in range(0, len(Z_mid)):
                        Z_mid_temp[k] *= Z_mid[k]
                        t_temp[k] *= t[k]
                    Z_mid = Z_mid_temp
                    t = t_temp
                    
            return self.rz2rho(method, R_mid, Z_mid, t, **kwargs)
    
    def psinorm2rmid(self, psi_norm, t, **kwargs):
        """Calculates the outboard R_mid location corresponding to the passed psinorm (normalized poloidal flux) values.
        
        Args:
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to Rmid.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `psi_norm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `psi_norm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of Rmid. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `psi_norm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `psi_norm` or be
                a scalar. Default is True (evaluate ALL `psi_norm` at EACH element in
                `t`).
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of Rmid. Default is False (return major radius, Rmid).            
            length_unit (String or 1): Length unit that `Rmid` is returned in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `Rmid` or (`Rmid`, `time_idxs`)
            
            * **Rmid** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `Rmid`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single R_mid value for psinorm=0.7, t=0.26s::
            
                R_mid_val = Eq_instance.psinorm2rmid(0.7, 0.26)
            
            Find R_mid values at psi_norm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                R_mid_arr = Eq_instance.psinorm2rmid([0.5, 0.7], 0.26)
            
            Find R_mid values at psi_norm=0.5 at times t=[0.2s, 0.3s]::
            
                R_mid_arr = Eq_instance.psinorm2rmid(0.5, [0.2, 0.3])
            
            Find R_mid values at (psinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                R_mid_arr = Eq_instance.psinorm2rmid([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        # Convert units from meters to desired target but keep units consistent
        # with rho keyword:
        if kwargs.get('rho', False):
            unit_factor = 1
        else:
            unit_factor = self._getLengthConversionFactor('m', kwargs.get('length_unit', 1))
        
        return unit_factor * self._psinorm2Quan(
            self._getRmidSpline,
            psi_norm,
            t,
            **kwargs
        )
    
    def psinorm2rho(self, method, *args, **kwargs):
        r"""Convert the passed (psinorm, t) coordinates into one of several coordinates.
        
        Args:
            method (String): Indicates which coordinates to convert to.
                Valid options are:
                
                    ======= =================================
                    phinorm Normalized toroidal flux
                    volnorm Normalized volume
                    Rmid    Midplane major radius
                    r/a     Normalized minor radius
                    q       Safety factor
                    F       Flux function :math:`F=RB_{\phi}`
                    FFPrime Flux function :math:`FF'`
                    p       Pressure
                    pprime  Pressure gradient
                    v       Flux surface volume
                    ======= =================================
                
                Additionally, each valid option may be prepended with 'sqrt'
                to specify the square root of the desired unit.
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to rho.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `psi_norm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `psi_norm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of rho. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `psi_norm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `psi_norm` or be
                a scalar. Default is True (evaluate ALL `psi_norm` at EACH element in
                `t`).
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of Rmid. Default is False (return major radius, Rmid).            
            length_unit (String or 1): Length unit that `Rmid` is returned in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `rho` or (`rho`, `time_idxs`)
            
            * **rho** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `rho`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Raises:
            ValueError: If `method` is not one of the supported values.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single phinorm value at psinorm=0.6, t=0.26s::
            
                phi_val = Eq_instance.psinorm2rho('phinorm', 0.6, 0.26)
            
            Find phinorm values at phinorm of 0.6 and 0.8 at the
            single time t=0.26s::
            
                phi_arr = Eq_instance.psinorm2rho('phinorm', [0.6, 0.8], 0.26)
            
            Find phinorm values at psinorm of 0.6 at times t=[0.2s, 0.3s]::
            
                phi_arr = Eq_instance.psinorm2rho('phinorm', 0.6, [0.2, 0.3])
            
            Find phinorm values at (psinorm, t) points (0.6, 0.2s) and (0.5m, 0.3s)::
            
                phi_arr = Eq_instance.psinorm2rho('phinorm', [0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        
        if method.startswith('sqrt'):
            kwargs['sqrt'] = True
            method = method[4:]
        
        if method == 'phinorm':
            return self.psinorm2phinorm(*args, **kwargs)
        elif method == 'volnorm':
            return self.psinorm2volnorm(*args, **kwargs)
        elif method == 'Rmid':
            return self.psinorm2rmid(*args, **kwargs)
        elif method == 'r/a':
            kwargs['rho'] = True
            return self.psinorm2rmid(*args, **kwargs)
        elif method == 'q':
            return self.psinorm2q(*args, **kwargs)
        elif method == 'F':
            return self.psinorm2F(*args, **kwargs)
        elif method == 'FFPrime':
            return self.psinorm2FFPrime(*args, **kwargs)
        elif method == 'p':
            return self.psinorm2p(*args, **kwargs)
        elif method == 'pprime':
            return self.psinorm2pprime(*args, **kwargs)
        elif method == 'v':
            return self.psinorm2v(*args, **kwargs)
        else:
            raise ValueError("psinorm2rho: Unsupported normalized coordinate method '%s'!" % method)
    
    def _getLengthConversionFactor(self, start, end, default=None):
        """Gets the conversion factor to convert from units start to units end.
        
        Uses a regex to parse units of the form:
        'm'
        'm^2'
        'm2'
        Leading and trailing spaces are NOT allowed.
        
        Valid unit specifiers are:
            'm'         meters
            'cm'        centimeters
            'mm'        millimeters
            'in'        inches
            'ft'        feet
            'yd'        yards
            'smoot'     smoots
            'cubit'     cubits
            'hand'      hands
        
        Args:
            start (String, int or None):
                Starting unit for the conversion.
                - If None, uses the unit specified when the instance was created.
                - If start is an int, the starting unit is taken to be the unit
                    specified when the instance was created raised to that power.
                - If start is 'default', either explicitly or because of
                    reverting to the instance-level unit, then the value passed
                    in the kwarg default is used. In this case, default must be
                    a complete unit string (i.e., not None, not an int and not
                    'default').
                - Otherwise, start must be a valid unit specifier as given above.
            end (String, int or None):
                Target (ending) unit for the conversion.
                - If None, uses the unit specified when the instance was created.
                - If end is an int, the target unit is taken to be the unit
                    specified when the instance was created raised to that power.
                - If end is 'default', either explicitly or because of
                    reverting to the instance-level unit, then the value passed
                    in the kwarg default is used. In this case, default must be
                    a complete unit string (i.e., not None, not an int and not
                    'default').
                - Otherwise, end must be a valid unit specifier as given above.
                    In this case, if end does not specify an exponent, it uses
                    whatever the exponent on start is. This allows a user to
                    ask for an area in units of m^2 by specifying
                    length_unit='m', for instance. An error will still be
                    raised if the user puts in a completely inconsistent
                    specification such as length_unit='m^3' or length_unit='m^1'.
        
        Keyword Args:
            default (String, int or None):
                The default unit to use in cases
                where start or end is 'default'. If default is None, an int, or 
                'default', then the value given for start is used. (A circular
                definition is prevented for cases in which start is default by
                checking for this case during the handling of the case
                start=='default'.)
        
        Returns:
            Conversion factor: Scalar float. The conversion factor to get from
                the start unit to the end unit.
        
        Raises:
            ValueError: If start is 'default' and default is None, an int, or
                'default'.
            ValueError: If the (processed) exponents of start and end or start
                and default are incompatible.
            ValueError: If the processed units for start and end are not valid.
        """
        # Input handling:
        # Starting unit:
        if start is None:
            # If start is None, it means to use the instance's default unit (implied to the power of 1):
            start = self._length_unit
        elif isinstance(start, (int, int)):
            # If start is an integer type, this is used as the power applied to the instance's default unit:
            if self._length_unit != 'default':
                start = self._length_unit + '^' + str(start)
            else:
                # If the instance's default unit is 'default', this is handled next:
                start = self._length_unit
        if start == 'default':
            # If start is 'default', the thing passed to default is used, but only if it is a complete unit specification:
            if default is None or isinstance(default, (int, int)) or default == 'default':
                raise ValueError("You must specify a complete unit (i.e., "
                                 "non-None, non-integer and not 'default') "
                                 "when using 'default' for the starting unit.")
            else:
                start = default
        
        # Default unit:
        if default is None or isinstance(default, (int, int)) or default == 'default':
            # If start is 'default', these cases have already been caught above.
            default = start
        
        # Target (ending) unit:
        if end is None:
            # If end is None, it means to use the instance's default unit (implied to the power of 1):
            end = self._length_unit
        elif isinstance(end, (int, int)):
            # If end is an integer type, this is used as the power applied to the instance's default unit:
            if self._length_unit != 'default':
                end = self._length_unit + '^' + str(end)
            else:
                # If the instance's default unit is 'default', this is handled next:
                end = self._length_unit
        if end == 'default':
            # If end is 'default', the thing passed to default is used, which
            # defaults to start, which itself is not allowed to be 'default':
            end = default
        
        unit_regex = r'^([A-Za-z]+)\^?([0-9]*)$'
        
        # Need to explicitly cast because MDSplus returns its own classes and
        # re.split doesn't seem to handle the polymorphism properly:
        start = str(start)
        end = str(end)
        default = str(default)
        
        dum1, start_u, start_pow, dum2 = re.split(unit_regex, start)
        dum1, end_u, end_pow, dum2 = re.split(unit_regex, end)
        dum1, default_u, default_pow, dum2 = re.split(unit_regex, default)
        
        start_pow = 1.0 if start_pow == '' else float(start_pow)
        if end_pow == '':
            end_pow = start_pow
        else:
            end_pow = float(end_pow)
        default_pow = 1.0 if default_pow == '' else float(default_pow)
        
        if start_pow != end_pow or start_pow != default_pow:
            raise ValueError("Incompatible exponents between '%s', '%s' and '%s'!" % (start, end, default))
        try:
            return (_length_conversion[start_u][end_u])**start_pow
        except KeyError:
            raise ValueError("Unit '%s' is not a recognized length unit!" % end)

    def _psinorm2Quan(self, spline_func, psi_norm, t, each_t=True, return_t=False,
                      sqrt=False, rho=False, k=3, blob=None,
                      check_space=False, convert_only=True, length_unit=1,
                      convert_roa=False):
        """Convert psinorm to a given quantity.
        
        Utility function for computing a variety of quantities given psi_norm
        and the relevant time indices.
        
        Args:
            spline_func (callable): Function which returns a 1d spline for the 
                quantity you want to convert into as a function of `psi_norm`
                given a time index.
            psi_norm (Array or scalar float): `psi_norm` values to evaluate at.
            time_idxs (Array or scalar float): Time indices for each of the
                `psi_norm` values. Shape must match that of `psi_norm`.
            t: Array or scalar float. Representative time array that `psi_norm`
                and `time_idxs` was formed from (used to determine output shape).
        
        Keyword Args:
            each_t (Boolean): When True, the elements in `psi_norm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `psi_norm` or be
                a scalar. Default is True (evaluate ALL `psi_norm` at EACH element in
                `t`).
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            sqrt (Boolean): Set to True to return the square root of `rho`. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of Rmid. Default is False (return major radius, Rmid).            
                Note that this will have unexpected results if `spline_func`
                returns anything other than R_mid.
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            time_idxs (Array with same shape as `psi_norm` or None):
                The time indices to use (as computed by :py:meth:`_processRZt`).
                Default is None (compute time indices in method).
            convert_roa (Boolean): When True, it is assumed that `psi_norm` is
                actually given as r/a and should be converted to Rmid before
                being passed to the spline for conversion. Default is False.
        
        Returns:
            (`rho`, `time_idxs`)
            
            * **rho** (`Array or scalar float`) - The converted quantity. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `rho`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        """
        if blob is None:
            # When called in this manner, this is just like what was done with
            # rz2psi.
            (
                psi_norm,
                dum,
                t,
                time_idxs,
                unique_idxs,
                single_time,
                single_val,
                original_shape
            ) = self._processRZt(
                psi_norm,
                psi_norm,
                t,
                make_grid=False,
                check_space=check_space,
                each_t=each_t,
                length_unit=length_unit,
                convert_only=convert_only,
                compute_unique=True
            )
            
            if self._tricubic:
                if convert_roa:
                    psi_norm = self._roa2rmid(psi_norm, t)
                quan_norm = spline_func(t).ev(t, psi_norm)
                if rho:
                    quan_norm = self._rmid2roa(quan_norm, t)
                quan_norm = quan_norm.reshape(original_shape)
            else:
                if single_time:
                    if convert_roa:
                        psi_norm = self._roa2rmid(psi_norm, time_idxs[0])
                    quan_norm = spline_func(time_idxs[0], k=k)(psi_norm)
                    if rho:
                        quan_norm = self._rmid2roa(quan_norm, time_idxs[0])
                    if single_val:
                        quan_norm = quan_norm[0]
                    else:
                        quan_norm = scipy.reshape(quan_norm, original_shape)
                elif each_t:
                    quan_norm = scipy.zeros(
                        scipy.concatenate(([len(time_idxs),], original_shape))
                    )
                    for idx, t_idx in enumerate(time_idxs):
                        if convert_roa:
                            psi_tmp = self._roa2rmid(psi_norm, t_idx)
                        else:
                            psi_tmp = psi_norm
                        tmp = spline_func(t_idx, k=k)(psi_tmp)
                        if rho:
                            tmp = self._rmid2roa(tmp, t_idx)
                        quan_norm[idx] = tmp.reshape(original_shape)
                else:
                    if convert_roa:
                        psi_norm = self._roa2rmid(psi_norm, time_idxs)
                    quan_norm = scipy.zeros_like(t, dtype=float)
                    for t_idx in unique_idxs:
                        t_mask = (time_idxs == t_idx)
                        tmp = spline_func(t_idx, k=k)(psi_norm[t_mask])
                        if rho:
                            tmp = self._rmid2roa(tmp, t_idx)
                        quan_norm[t_mask] = tmp
                    quan_norm = quan_norm.reshape(original_shape)
            if sqrt:
                if quan_norm.ndim == 0:
                    if quan_norm < 0.0:
                        quan_norm = 0.0
                else:
                    scipy.place(quan_norm, quan_norm < 0, 0.0)
                quan_norm = scipy.sqrt(quan_norm)
            
            if return_t:
                if self._tricubic:
                    return quan_norm, (t, single_time, single_val, original_shape)
                else:
                    return quan_norm, (time_idxs, unique_idxs, single_time, single_val, original_shape)
            else:
                return quan_norm
        else:
            # When called in this manner, psi_norm has already been expanded
            # through a pass through rz2psinorm, so we need to be more clever.
            if self._tricubic:
                t_proc, single_time, single_val, original_shape = blob
            else:
                time_idxs, unique_idxs, single_time, single_val, original_shape = blob
            # Override original_shape with shape of psi_norm:
            # psi_norm_shape = psi_norm.shape
            psi_norm_flat = psi_norm.reshape(-1)
            if self._tricubic:
                tt = t_proc.reshape(-1)
                if convert_roa:
                    psi_norm_flat = self._roa2rmid(psi_norm_flat, tt)
                quan_norm = spline_func(t).ev(t_proc, psi_norm_flat)
                if rho:
                    quan_norm = self._rmid2roa(quan_norm, tt)
                quan_norm = quan_norm.reshape(original_shape)
            else:
                if convert_roa:
                    psi_norm_flat = self._roa2rmid(psi_norm_flat, time_idxs)
                    if each_t:
                        psi_norm = psi_norm_flat.reshape(-1)
                if single_time:
                    quan_norm = spline_func(time_idxs[0], k=k)(psi_norm_flat)
                    if rho:
                        quan_norm = self._rmid2roa(quan_norm, time_idxs[0])
                    if single_val:
                        quan_norm = quan_norm[0]
                    else:
                        quan_norm = scipy.reshape(quan_norm, original_shape)
                elif each_t:
                    quan_norm = scipy.zeros(
                        scipy.concatenate(([len(time_idxs),], original_shape))
                    )
                    for idx, t_idx in enumerate(time_idxs):
                        tmp = spline_func(t_idx, k=k)(psi_norm[idx].reshape(-1))
                        if rho:
                            tmp = self._rmid2roa(tmp, t_idx)
                        quan_norm[idx] = tmp.reshape(original_shape)
                else:
                    quan_norm = scipy.zeros_like(time_idxs, dtype=float)
                    for t_idx in unique_idxs:
                        t_mask = (time_idxs == t_idx)
                        tmp = spline_func(t_idx, k=k)(psi_norm_flat[t_mask])
                        if rho:
                            tmp = self._rmid2roa(tmp, t_idx)
                        quan_norm[t_mask] = tmp
                    quan_norm = quan_norm.reshape(original_shape)
            
            if sqrt:
                if quan_norm.ndim == 0:
                    if quan_norm < 0:
                        quan_norm = 0.0
                else:
                    scipy.place(quan_norm, quan_norm < 0, 0.0)
                quan_norm = scipy.sqrt(quan_norm)
            
            if return_t:
                return quan_norm, blob
            else:
                return quan_norm

    def _getRmidSpline(self, idx, k=3):
        """Returns the spline object corresponding to the passed time index idx,
        generating it if it does not already exist.
        
        There are two approaches that come to mind:
            -- In Steve Wolfe's implementation of efit_rz2mid and efit_psi2rmid,
                he uses the EFIT output Rmid as a function of normalized flux
                (i.e., what is returned by self.getRmidPsi()) in the core, then
                expands the grid beyond this manually.
            -- A simpler approach would be to just compute the psi_norm(R_mid)
                grid directly from the radial grid.
        
        The latter approach is selected for simplicity.
        
        The units of R_mid are always meters, and are converted by the wrapper
        functions to whatever the user wants.
        
        Args:
            idx (Scalar int):
                The time index to retrieve the flux spline for.
                This is ASSUMED to be a valid index for the first dimension of
                self.getFluxGrid(), otherwise an IndexError will be raised.
        
        Keyword Args:
            k (positive int)
                Polynomial degree of spline to use. Default is 3.
        
        Returns:
            :py:class:`UnivariateInterpolator` or
                :py:class:`tripline.RectBivariateSpline` depending on whether or
                not the instance was created with the `tspline` keyword.
        """
        if not self._tricubic:
            try:
                return self._RmidSpline[idx][k]
            except KeyError:
                # New approach: create a fairly dense radial grid from the
                # global flux grid to avoid 1d interpolation problems in the
                # core. The bivariate spline seems to be a little more robust
                # in this respect.
                resample_factor = 3
                R_grid = scipy.linspace(
                    self.getMagR(length_unit='m')[idx],
                    self.getRGrid(length_unit='m')[-1],
                    resample_factor * len(self.getRGrid(length_unit='m'))
                )
                
                psi_norm_on_grid = self.rz2psinorm(
                    R_grid,
                    self.getMagZ(length_unit='m')[idx] * scipy.ones(R_grid.shape),
                    self.getTimeBase()[idx]
                )
                # Correct for the slight issues at the magnetic axis:
                psi_norm_on_grid[0] = 0.0
                # Find if it ever goes non-monotonic: psinorm is assumed to be
                # strictly INCREASING from the magnetic axis out.
                decr_idx, = scipy.where((psi_norm_on_grid[1:] - psi_norm_on_grid[:-1]) < 0)
                if len(decr_idx) > 0:
                    psi_norm_on_grid = psi_norm_on_grid[:decr_idx[0] + 1]
                    R_grid = R_grid[:decr_idx[0] + 1]
                
                spline = UnivariateInterpolator(
                    psi_norm_on_grid, R_grid, k=k
                )
                try:
                    self._RmidSpline[idx][k] = spline
                except KeyError:
                    self._RmidSpline[idx] = {k: spline}
                return self._RmidSpline[idx][k]
        else:
            if self._RmidSpline:
                return self._RmidSpline
            else:
                resample_factor = 3 * len(self.getRGrid(length_unit='m'))
                
                # generate timebase and R_grid through a meshgrid
                t, R_grid = scipy.meshgrid(
                    self.getTimeBase(),
                    scipy.zeros((resample_factor,))
                )
                Z_grid = scipy.dot(
                    scipy.ones((resample_factor, 1)),
                    scipy.atleast_2d(self.getMagZ(length_unit='m'))
                )
                
                for idx in scipy.arange(self.getTimeBase().size):
                    R_grid[:, idx] = scipy.linspace(
                        self.getMagR(length_unit='m')[idx],
                        self.getRGrid(length_unit='m')[-1],
                        resample_factor
                    )
                
                psi_norm_on_grid = self.rz2psinorm(
                    R_grid,
                    Z_grid,
                    t,
                    each_t=False
                )
                # Correct for the slight issues at the magnetic axis:
                psi_norm_on_grid[0, :] = 0.0
                
                self._RmidSpline = BivariateInterpolator(
                    t.ravel(),
                    psi_norm_on_grid.ravel(),
                    R_grid.ravel()
                )
                
                return self._RmidSpline

    def _processRZt(self, R, Z, t, make_grid=False, each_t=True, check_space=True, length_unit=1, convert_only=False, compute_unique=False):
        """Input checker/processor.
        
        Takes R, Z and t. Appropriately packages them into scipy arrays. Checks
        the validity of the R, Z ranges. If there is a single time value but
        multiple R, Z values, creates matching time vector. If there is a single
        R, Z value but multiple t values, creates matching R and Z vectors.
        Finds list of nearest-neighbor time indices.
        
        Args:
            R (Array-like or scalar float):
                Values of the radial coordinate. If `R` and `Z` are both scalar
                values, they are used as the coordinate pair for all of the
                values in `t`. Must have the same shape as `Z` unless the
                `make_grid` keyword is True. If `make_grid` is True, `R` must
                have only one dimension (or be a scalar).
            Z (Array-like or scalar float):
                Values of the vertical coordinate. If `R` and `Z` are both
                scalar values, they are used as the coordinate pair for all of
                the values in `t`. Must have the same shape as `R` unless the
                `make_grid` keyword is True. If `make_grid` is True, `Z` must
                have only one dimension.
            t (Array-like or single value):
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If `t` is array-like and `make_grid` is False, `t`
                must have the same dimensions as `R` and `Z`. If `t` is
                array-like and `make_grid` is True, `t` must have shape
                (len(Z), len(R)).
        
        Keyword Args:
            make_grid (Boolean):
                Set to True to pass `R` and `Z` through :py:func:`meshgrid`
                before evaluating. If this is set to True, `R` and `Z` must each
                only have a single dimension, but can have different lengths.
                Default is False (do not form meshgrid).
            each_t (Boolean):
                When True, the elements in `R` and `Z` (or the meshgrid thereof
                if `make_grid` is True) are evaluated at each value in `t`. If
                True, `t` must have only one dimension (or be a scalar). If
                False, `t` must match the shape of `R` and `Z` (or their
                meshgrid if `make_grid` is True) or be a scalar. Default is True
                (evaluate ALL `R`, `Z` at each element in `t`).
            check_space (Boolean):
                If True, `R` and `Z` are converted to meters and checked against
                the extents of the spatial grid.
            length_unit (String or 1):
                Length unit that `R` and `Z` are being given in. If a string is
                given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (R and Z given in meters). Note that this factor is
                ONLY applied to the inputs in this function -- if Quan needs to
                be corrected, it must be done in the calling function.
        
        Returns:
            Tuple of:
            
            * **R** - Flattened `R` array with out-of-range values replaced with NaN.
            * **Z** - Flattened `Z` array with out-of-range values replaced with NaN.
            * **t** - Flattened `t` array with out-of-range values replaced with NaN.
            * **time_idxs** - Flattened array of nearest-neighbor time indices.
              None if :py:attr:`self._tricubic`.
            * **unique_idxs** - 1d array of the unique values in time_idxs, can
              be used to save time elsewhere. None if :py:attr:`self._tricubic`.
            * **single_time** - Boolean indicating whether a single time value
              is used. If True, then certain simplifying steps can be made and
              the output should be unwrapped before returning to ensure the
              least surprise.
            * **original_shape** - Original shape tuple, used to return the
              arrays to their starting form. If `single_time` or `each_t` is
              True, this is the shape of the (expanded) `R`, `Z` arrays. It is
              assumed that time will be added as the leading dimension.
        """
        
        # Get everything into sensical datatypes. Must force it to be float to
        # keep scipy.interpolate happy.
        R = scipy.asarray(R, dtype=float)
        Z = scipy.asarray(Z, dtype=float)
        t = scipy.asarray(t, dtype=float)
        single_time = (t.ndim == 0)
        single_val = (R.ndim == 0) and (Z.ndim == 0)
        
        # Check the shape of t:
        if each_t and t.ndim > 1:
            raise ValueError(
                "_processRZt: When using the each_t keyword, t can have at most "
                "one dimension!"
            )
        
        # Form the meshgrid and check the input dimensions as needed:
        if make_grid:
            if R.ndim != 1 or Z.ndim != 1:
                raise ValueError(
                    "_processRZt: When using the make_grid keyword, the number "
                    "of dimensions of R and Z must both be one!"
                )
            R, Z = scipy.meshgrid(R, Z)
        else:
            if R.shape != Z.shape:
                raise ValueError(
                    "_processRZt: Shape of R and Z arrays must match! Use "
                    "make_grid=True to form a meshgrid from 1d R, Z arrays."
                )
        
        if not single_time and not each_t and t.shape != R.shape:
            raise ValueError(
                "_processRZt: Shape of t does not match shape of R and Z!"
            )
        
        # Check that the R, Z points lie within the grid:
        if check_space:
            # Convert units to meters:
            unit_factor = self._getLengthConversionFactor(
                length_unit,
                'm',
                default='m'
            )
            R = unit_factor * R
            Z = unit_factor * Z
            
            if not convert_only:
                good_points, num_good = self._checkRZ(R, Z)
                
                if num_good < 1:
                    raise ValueError('_processRZt: No valid points!')
                
                # Handle bug in older scipy:
                if R.ndim == 0:
                    if not good_points:
                        R = scipy.nan
                else:
                    scipy.place(R, ~good_points, scipy.nan)
                if Z.ndim == 0:
                    if not good_points:
                        Z = scipy.nan
                else:
                    scipy.place(Z, ~good_points, scipy.nan)
        
        if self._tricubic:
            # When using tricubic spline interpolation, the arrays must be
            # replicated when using the each_t keyword.
            if single_time:
                t = t * scipy.ones_like(R, dtype=float)
            elif each_t:
                R = scipy.tile(R, [len(t),] + [1,] * R.ndim)
                Z = scipy.tile(Z, [len(t),] + [1,] * Z.ndim)
                t = t[scipy.indices(R.shape)[0]]
            time_idxs = None
            unique_idxs = None
            t = scipy.reshape(t, -1)
        else:
            t = scipy.reshape(t, -1)
            timebase = self.getTimeBase()
            # Get nearest-neighbor points:
            time_idxs = self._getNearestIdx(t, timebase)
            # Check errors and warn if needed:
            t_errs = scipy.absolute(t - timebase[time_idxs])

            # FS: comment this out to avoid known warning
            # Assume a constant sampling rate to save time:
            #if len(time_idxs) > 1 and (t_errs > (old_div((timebase[1] - timebase[0]), 3.0))).any():
            #    warnings.warn(
            #        "Some time points are off by more than 1/3 the EFIT point "
            #        "spacing. Using nearest-neighbor interpolation between time "
            #        "points. You may want to run EFIT on the timebase you need. "
            #        "Max error: %.3fs" % (max(t_errs),),
            #        RuntimeWarning
            #    )
            if compute_unique and not single_time and not each_t:
                unique_idxs = scipy.unique(time_idxs)
            else:
                unique_idxs = None
        
        original_shape = R.shape
        R = scipy.reshape(R, -1)
        Z = scipy.reshape(Z, -1)
        
        return R, Z, t, time_idxs, unique_idxs, single_time, single_val, original_shape

    def _getNearestIdx(self, v, a):
        """Returns the array of indices of the nearest value in a corresponding to each value in v.
        
        If the monotonic keyword in the instance is True, then this is done using
        scipy.digitize under the assumption that a is monotonic. Otherwise,
        this is done in a general manner by looking for the minimum distance
        between the points in v and a.
        
        Args:
            v (Array):
                Input values to match to nearest neighbors in a.
            a (Array):
                Given values to match against.
        
        Returns:
            Indices in a of the nearest values to each value in v. Has the same
                shape as v.
        """
        # Gracefully handle single-value versus array inputs, returning in the
        # corresponding type.
        if not self._monotonic:
            try:
                return scipy.array([(scipy.absolute(a - val)).argmin() for val in v])
            except TypeError:
                return (scipy.absolute(a - v)).argmin()
        else:
            try:
                return scipy.digitize(v, old_div((a[1:] + a[:-1]), 2.0))
            except ValueError:
                return scipy.digitize(scipy.atleast_1d(v), old_div((a[1:] + a[:-1]), 2.0)).reshape(())

    def _checkRZ(self, R, Z):
        """Checks whether or not the passed arrays of (R, Z) are within the bounds of the reconstruction data.
        
        Returns the mask array of booleans indicating the goodness of each point
        at the corresponding index. Raises warnings if there are no good_points
        and if there are some values out of bounds.
        
        Assumes R and Z are in meters and that the R and Z arrays returned by
        this instance's getRGrid() and getZGrid() are monotonically increasing.
        
        Args:
            R (Array):
                Radial coordinate to check. Must have the same size as Z.
            Z (Array)
                Vertical coordinate to check. Must have the same size as R.
        
        Returns:
            good_points: Boolean array. True where points are within the bounds
                defined by self.getRGrid and self.getZGrid.
            num_good: The number of good points.
        """
        good_points = ((R <= self.getRGrid(length_unit='m')[-1]) &
                       (R >= self.getRGrid(length_unit='m')[0]) &
                       (Z <= self.getZGrid(length_unit='m')[-1]) &
                       (Z >= self.getZGrid(length_unit='m')[0]))
        # Gracefully handle single-value versus array inputs, returning in the
        # corresponding type.
        num_good = scipy.sum(good_points)
        test = scipy.array(R)
        if len(test.shape) > 0:
            num_pts = test.size
        else:
            num_good = good_points
            num_pts = 1
        if num_good == 0:
            warnings.warn("Warning: _checkRZ: No valid (R, Z) points!",
                          RuntimeWarning)
        elif num_good != num_pts:
            warnings.warn("Warning: _checkRZ: Some (R, Z) values out of bounds. "
                          "(%(bad)d bad out of %(tot)d)"
                          % {'bad': num_pts - num_good, 'tot': num_pts},
                          RuntimeWarning)
        
        return (good_points, num_good)

    def _getFluxBiSpline(self, idx):
        """Gets the spline corresponding to the given time index, generating as needed.
        
        This returns a bivariate spline for when the instance is created with
        keyword tspline=False.
        
        Args:
            idx (Scalar int):
                The time index to retrieve the flux spline for.
                This is ASSUMED to be a valid index for the first dimension of
                self.getFluxGrid(), otherwise an IndexError will be raised.
        
        Returns:
            An instance of scipy.interpolate.RectBivariateSpline corresponding
                to the given time index idx.
        """
        try:
            return self._psiOfRZSpline[idx]
        except KeyError:
            # Note the order of the arguments -- psiRZ is stored with t along
            # the first dimension, Z along the second and R along the third.
            # This leads to intuitive behavior when contour plotting, but
            # mandates the syntax here.
            self._psiOfRZSpline[idx] = scipy.interpolate.RectBivariateSpline(
                self.getZGrid(length_unit='m'),
                self.getRGrid(length_unit='m'),
                self.getFluxGrid()[idx, :, :],
                s=0
            )
            return self._psiOfRZSpline[idx]

    def rz2BZ(self, R, Z, t, return_t=False, make_grid=False, each_t=True, length_unit=1):
        r"""Calculates the vertical component of the magnetic field at the given (R, Z, t) coordinates.
        
        Uses
        
        .. math::
            
            B_Z = \frac{1}{R}\frac{\partial \psi}{\partial R}
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to vertical field. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to vertical field. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `R` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `Z` must
                have exactly one dimension.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If the `each_t` keyword is True, then `t` must be
                scalar or have exactly one dimension. If the `each_t` keyword is
                False, `t` must have the same shape as `R` and `Z` (or their
                meshgrid if `make_grid` is True).
        
        Keyword Args:
            each_t (Boolean): When True, the elements in `R`, `Z` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R` and
                `Z` or be a scalar. Default is True (evaluate ALL `R`, `Z` at
                EACH element in `t`).
            make_grid (Boolean): Set to True to pass `R` and `Z` through
                :py:func:`scipy.meshgrid` before evaluating. If this is set to
                True, `R` and `Z` must each only have a single dimension, but
                can have different lengths. Default is False (do not form
                meshgrid).
            length_unit (String or 1): Length unit that `R`, `Z` are given in.
                If a string is given, it must be a valid unit specifier:
                    
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            return_t (Boolean): Set to True to return a tuple of (`BZ`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `BZ` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `BZ`).
        
        Returns:
            `BZ` or (`BZ`, `time_idxs`)
            
            * **BZ** (`Array or scalar float`) - The vertical component of the
              magnetic field. If all of the input arguments are scalar, then a
              scalar is returned. Otherwise, a scipy Array is returned. If `R`
              and `Z` both have the same shape then `BZ` has this shape as well,
              unless the `make_grid` keyword was True, in which case `BZ` has
              shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `BZ`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single BZ value at R=0.6m, Z=0.0m, t=0.26s::
                
                BZ_val = Eq_instance.rz2BZ(0.6, 0, 0.26)
            
            Find BZ values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully
            specified, even if the values are all the same::
                
                BZ_arr = Eq_instance.rz2BZ([0.6, 0.8], [0, 0], 0.26)
            
            Find BZ values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
                
                BZ_arr = Eq_instance.rz2BZ(0.6, 0, [0.2, 0.3])
            
            Find BZ values at (R, Z, t) points (0.6m, 0m, 0.2s) and
            (0.5m, 0.2m, 0.3s)::
                
                BZ_arr = Eq_instance.rz2BZ([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find BZ values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
                
                BZ_mat = Eq_instance.rz2BZ(R, Z, 0.2, make_grid=True)
        """
        
        # Check inputs and process into flat arrays with units of meters:
        R, Z, t, time_idxs, unique_idxs, single_time, single_val, original_shape = self._processRZt(
            R,
            Z,
            t,
            make_grid=make_grid,
            each_t=each_t,
            length_unit=length_unit,
            compute_unique=True
        )
        
        if self._tricubic:
            # TODO: This almost certainly isn't implemented!
            out_vals = scipy.reshape(
                1.0 / R * self._getFluxTriSpline().ev(t, Z, R, dx=1, dy=0, dz=0),
                original_shape
            )
        else:
            if single_time:
                out_vals = 1.0 / R * self._getFluxBiSpline(time_idxs[0]).ev(Z, R, dx=0, dy=1)
                if single_val:
                    out_vals = out_vals[0]
                else:
                    out_vals = scipy.reshape(out_vals, original_shape)
            elif each_t:
                out_vals = scipy.zeros(
                    scipy.concatenate(([len(time_idxs),], original_shape))
                )
                for idx, t_idx in enumerate(time_idxs):
                    out_vals[idx] = scipy.reshape(
                        1.0 / R * self._getFluxBiSpline(t_idx).ev(Z, R, dx=0, dy=1),
                        original_shape
                    )
            else:
                out_vals = scipy.zeros_like(t, dtype=float)
                for t_idx in unique_idxs:
                    t_mask = (time_idxs == t_idx)
                    out_vals[t_mask] = 1.0 / R[t_mask] * self._getFluxBiSpline(t_idx).ev(Z[t_mask], R[t_mask], dx=0, dy=1)
                out_vals = scipy.reshape(out_vals, original_shape)
        
        # Correct for current sign:
        out_vals = -1.0 * out_vals * self.getCurrentSign()
        
        if return_t:
            if self._tricubic:
                return out_vals, (t, single_time, single_val, original_shape)
            else:
                return out_vals, (time_idxs, unique_idxs, single_time, single_val, original_shape)
        else:
            return out_vals

    def rz2BT(self, R, Z, t, **kwargs):
        r"""Calculates the toroidal component of the magnetic field at the given (R, Z, t).
        
        Uses :math:`B_\phi = F / R`.
        
        By default, EFIT only computes this inside the LCFS. To approximate the
        field outside of the LCFS, :math:`B_\phi \approx B_{t, vac} R_0 / R` is
        used, where :math:`B_{t, vac}` is obtained with :py:meth:`getBtVac` and
        :math:`R_0` is the major radius of the magnetic axis obtained from
        :py:meth:`getMagR`.
        
        The coordinate system used is right-handed, such that "forward" field on
        Alcator C-Mod (clockwise when seen from above) has negative BT.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to BT. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to BT. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `R` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `Z` must
                have exactly one dimension.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If the `each_t` keyword is True, then `t` must be
                scalar or have exactly one dimension. If the `each_t` keyword is
                False, `t` must have the same shape as `R` and `Z` (or their
                meshgrid if `make_grid` is True).
        
        Keyword Args:
            each_t (Boolean): When True, the elements in `R`, `Z` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R` and
                `Z` or be a scalar. Default is True (evaluate ALL `R`, `Z` at
                EACH element in `t`).
            make_grid (Boolean): Set to True to pass `R` and `Z` through
                :py:func:`scipy.meshgrid` before evaluating. If this is set to
                True, `R` and `Z` must each only have a single dimension, but
                can have different lengths. Default is False (do not form
                meshgrid).
            length_unit (String or 1): Length unit that `R`, `Z` are given in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            return_t (Boolean): Set to True to return a tuple of (`BT`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `BT` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `BT`).
        
        Returns:
            `BT` or (`BT`, `time_idxs`)
            
            * **BT** (`Array or scalar float`) - The toroidal magnetic field.
              If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `BT` has this shape as well,
              unless the `make_grid` keyword was True, in which case `BT`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `BT`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
       
        Examples:
            All assume that `Eq_instance` is a valid instance of the
            appropriate extension of the :py:class:`Equilibrium` abstract class.
            
            Find single BT value at R=0.6m, Z=0.0m, t=0.26s::
            
                BT_val = Eq_instance.rz2BT(0.6, 0, 0.26)
            
            Find BT values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                BT_arr = Eq_instance.rz2BT([0.6, 0.8], [0, 0], 0.26)
            
            Find BT values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                BT_arr = Eq_instance.rz2BT(0.6, 0, [0.2, 0.3])
            
            Find BT values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                BT_arr = Eq_instance.rz2BT([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find BT values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                BT_mat = Eq_instance.rz2BT(R, Z, 0.2, make_grid=True)
        """
        return_t = kwargs.get('return_t', False)
        unit_factor = self._getLengthConversionFactor('m', kwargs.get('length_unit', 1))
        out = self.rz2F(R, Z, t, **kwargs)
        if return_t:
            F, blob = out
        else:
            F = out
        
        B_T = old_div(F, R)
        
        # This will have NaN anywhere outside of the LCFS. Only handle if we
        # we need to.
        if scipy.isnan(B_T).any():
            warnings.warn(
                "Flux function F not provided outside of LCFS, assuming field "
                "goes like 1/R there to compute BT! This may be inaccurate!",
                RuntimeWarning
            )
            # This unfortunately requires a second call to _processRZt:
            R, Z, t, time_idxs, unique_idxs, single_time, single_val, original_shape = self._processRZt(
                R, Z, t,
                make_grid=kwargs.get('make_grid', False),
                each_t=kwargs.get('each_t', True),
                length_unit=kwargs.get('length_unit', 1),
                compute_unique=True
            )
            if self._tricubic:
                B_T = B_T.ravel()
                mask = scipy.isnan(B_T)
                B_T[mask] = self.getBtVacSpline()(t) * self.getMagRSpline()(t) / R[mask]
                B_T = scipy.reshape(B_T, original_shape)
            else:
                if single_time:
                    B_T = B_T.ravel()
                    mask = scipy.isnan(B_T)
                    B_T[mask] = self.getBtVac()[time_idxs] * self.getMagR()[time_idxs] / R[mask]
                    if single_val:
                        B_T = B_T[0]
                    else:
                        B_T = scipy.reshape(B_T, original_shape)
                elif kwargs.get('each_t', True):
                    for idx, t_idx in enumerate(time_idxs):
                        tmp_out = B_T[idx].ravel()
                        mask = scipy.isnan(tmp_out)
                        tmp_out[mask] = self.getBtVac()[t_idx] * self.getMagR()[t_idx] / R[mask]
                        B_T[idx] = scipy.reshape(tmp_out, original_shape)
                else:
                    B_T = B_T.ravel()
                    for t_idx in unique_idxs:
                        t_mask = (time_idxs == t_idx)
                        tmp_out = B_T[t_mask]
                        mask = scipy.isnan(tmp_out)
                        tmp_out[mask] = self.getBtVac()[t_idx] * self.getMagR()[t_idx] / R[t_mask][mask]
                        B_T[t_mask] = tmp_out
                    B_T = scipy.reshape(B_T, original_shape)
        
        if return_t:
            return unit_factor * B_T, blob
        else:
            return unit_factor * B_T

    def rz2F(self, R, Z, t, **kwargs):
        r"""Calculates the flux function :math:`F=RB_{\phi}` at the given (R, Z, t).
        
        By default, EFIT only computes this inside the LCFS.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to F. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to F. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `R` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `Z` must
                have exactly one dimension.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R`, `Z`. If the `each_t` keyword is True, then `t` must be
                scalar or have exactly one dimension. If the `each_t` keyword is
                False, `t` must have the same shape as `R` and `Z` (or their
                meshgrid if `make_grid` is True).
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of F. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `R`, `Z` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `R` and
                `Z` or be a scalar. Default is True (evaluate ALL `R`, `Z` at
                EACH element in `t`).
            make_grid (Boolean): Set to True to pass `R` and `Z` through
                :py:func:`scipy.meshgrid` before evaluating. If this is set to
                True, `R` and `Z` must each only have a single dimension, but
                can have different lengths. Default is False (do not form
                meshgrid).
            length_unit (String or 1): Length unit that `R`, `Z` are given in.
                If a string is given, it must be a valid unit specifier:
                
                    ===========  ===========
                    'm'          meters
                    'cm'         centimeters
                    'mm'         millimeters
                    'in'         inches
                    'ft'         feet
                    'yd'         yards
                    'smoot'      smoots
                    'cubit'      cubits
                    'hand'       hands
                    'default'    meters
                    ===========  ===========
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (use meters).
            return_t (Boolean): Set to True to return a tuple of (`F`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `F` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `F`).
        
        Returns:
            `F` or (`F`, `time_idxs`)
            
            * **F** (`Array or scalar float`) - The flux function :math:`F=RB_{\phi}`.
              If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `F` has this shape as well,
              unless the `make_grid` keyword was True, in which case `F`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `F`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
       
        Examples:
            All assume that `Eq_instance` is a valid instance of the
            appropriate extension of the :py:class:`Equilibrium` abstract class.
            
            Find single F value at R=0.6m, Z=0.0m, t=0.26s::
            
                F_val = Eq_instance.rz2F(0.6, 0, 0.26)
            
            Find F values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                F_arr = Eq_instance.rz2F([0.6, 0.8], [0, 0], 0.26)
            
            Find F values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                F_arr = Eq_instance.rz2F(0.6, 0, [0.2, 0.3])
            
            Find F values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                F_arr = Eq_instance.rz2F([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find F values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                F_mat = Eq_instance.rz2F(R, Z, 0.2, make_grid=True)
        """
        return self._RZ2Quan(self._getFSpline, R, Z, t, **kwargs)

    def _RZ2Quan(self, spline_func, R, Z, t, **kwargs):
        """Convert RZ to a given quantity.
        
        Utility function for converting R, Z coordinates to a variety of things
        that are interpolated from something measured on a uniform normalized
        flux grid, in particular phi_norm, vol_norm and R_mid.
        
        If tspline is False for this Equilibrium instance, uses
        scipy.interpolate.RectBivariateSpline to interpolate in terms of R and
        Z. Finds the nearest time slices to those given: nearest-neighbor
        interpolation in time. Otherwise, uses the tricubic package to perform
        a trivariate interpolation in space and time.
        
        Args:
            spline_func (callable): Function which returns a 1d spline for the
                quantity you want to convert into as a function of psi_norm
                given a time index.
            R (Array-like or scalar float): Values of the radial coordinate to
                map to Quan. If R and Z are both scalar values, they are used
                as the coordinate pair for all of the values in t. Must have
                the same shape as Z unless the make_grid keyword is set. If the
                make_grid keyword is True, R must have shape (len_R,).
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to Quan. If R and Z are both scalar values, they are used
                as the coordinate pair for all of the values in t. Must have
                the same shape as R unless the make_grid keyword is set. If the
                make_grid keyword is True, Z must have shape (len_Z,).
            t (Array-like or single value): If t is a single value, it is used
                for all of the elements of R, Z. If t is array-like and the
                make_grid keyword is False, t must have the same dimensions as
                R and Z. If t is array-like and the make_grid keyword is True,
                t must have shape (len(Z), len(R)).
        
        Keyword Args:
            each_t (Boolean):
                When True, the elements in `R` and `Z` (or the meshgrid thereof
                if `make_grid` is True) are evaluated at each value in `t`. If
                True, `t` must have only one dimension (or be a scalar). If
                False, `t` must match the shape of `R` and `Z` (or their
                meshgrid if `make_grid` is True) or be a scalar. Default is True
                (evaluate ALL `R`, `Z` at each element in `t`).
            return_t (Boolean):
                Set to True to return a tuple of (Quan,
                time_idxs), where time_idxs is the array of time indices
                actually used in evaluating R_mid with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return Quan).
            sqrt (Boolean):
                Set to True to return the square root of Quan. Only
                the square root of positive values is taken. Negative values
                are replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False (return Quan
                itself).
            make_grid (Boolean):
                Set to True to pass R and Z through meshgrid
                before evaluating. If this is set to True, R and Z must each
                only have a single dimension, but can have different lengths.
                When using this option, it is highly recommended to only pass
                a scalar value for t (such that each point in the flux grid is
                evaluated at this same value t). Otherwise, t must have the
                same shape as the resulting meshgrid, and each element in the
                returned psi array will be at the corresponding time in the t
                array. Default is False (do not form meshgrid).
            rho (Boolean):
                Set to True to return r/a (normalized minor radius)
                instead of R_mid. Default is False (return major radius, R_mid).
                Note that this will have unexpected results if spline_func
                returns anything other than R_mid.
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            length_unit (String or 1):
                Length unit that R and Z are being given
                in. If a string is given, it must be a valid unit specifier:
                
                    =========== ===========
                    'm'         meters
                    'cm'        centimeters
                    'mm'        millimeters
                    'in'        inches
                    'ft'        feet
                    'yd'        yards
                    'smoot'     smoots
                    'cubit'     cubits
                    'hand'      hands
                    'default'   meters
                    =========== ===========
                    
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (R and Z given in meters). Note that this factor is
                ONLY applied to the inputs in this function -- if Quan needs to
                be corrected, it must be done in the calling function.
        
        Returns:
            Quan: Array or scalar float. If all of the input arguments are
                scalar, then a scalar is returned. Otherwise, a scipy Array
                instance is returned. If R and Z both have the same shape then
                Quand has this shape as well. If the make_grid keyword was True
                then R_mid has shape (len(Z), len(R)).
            time_idxs: Array with same shape as R_mid. The indices (in
                self.getTimeBase()) that were used for nearest-neighbor
                interpolation. Only returned if return_t is True.
        """
        return_t = kwargs.get('return_t', False)
        kwargs['return_t'] = True
        
        # Not used by rz2psinorm:
        k = kwargs.pop('k', 3)
        rho = kwargs.pop('rho', False)
        
        # Make sure we don't convert to sqrtpsinorm first!
        sqrt = kwargs.pop('sqrt', False)
        
        psi_norm, blob = self.rz2psinorm(R, Z, t, **kwargs)
        
        kwargs['sqrt'] = sqrt
        kwargs['return_t'] = return_t
        
        # Not used by _psinorm2Quan
        kwargs.pop('length_unit', 1)
        kwargs.pop('make_grid', False)
        
        kwargs['rho'] = rho
        return self._psinorm2Quan(
            spline_func,
            psi_norm,
            t,
            blob=blob,
            k=k,
            **kwargs
        )

    def _getFSpline(self, idx, k=3):
        """Get spline to convert psinorm to F.
        
        Returns the spline object corresponding to the passed time index idx,
        generating it if it does not already exist.
        
        Args:
            idx (Scalar int):
                The time index to retrieve the flux spline for.
                This is ASSUMED to be a valid index for the first dimension of
                self.getFluxGrid(), otherwise an IndexError will be raised.
        
        Keyword Args:
            k (positive int)
                Polynomial degree of spline to use. Default is 3.
        
        Returns:
            :py:class:`trispline.UnivariateInterpolator` or
                :py:class:`tripline.RectBivariateSpline` depending on whether or
                not the instance was created with the `tspline` keyword.
        """
        if not self._tricubic:
            try:
                return self._FSpline[idx][k]
            except KeyError:
                F = self.getF()[idx]
                spline = UnivariateInterpolator(
                    scipy.linspace(0.0, 1.0, len(F)),
                    F,
                    k=k
                )
                try:
                    self._FSpline[idx][k] = spline
                except KeyError:
                    self._FSpline[idx] = {k: spline}
                return self._FSpline[idx][k]
        else:
            if self._FSpline:
                return self._FSpline
            else:
                F = self.getF()
                self._FSpline = RectBivariateSpline(
                    self.getTimeBase(),
                    scipy.linspace(0.0, 1.0, F.shape[1]),
                    F,
                    bounds_error=False,
                    s=0
                )
                return self._FSpline

    def _getRmidToPsiNormSpline(self, idx, k=3):
        """Get the spline which converts Rmid to psinorm.
        
        Returns the spline object corresponding to the passed time index idx,
        generating it if it does not already exist.
        
        There are two approaches that come to mind:
            -- In Steve Wolfe's implementation of efit_rz2mid and efit_psi2rmid,
                he uses the EFIT output Rmid as a function of normalized flux
                (i.e., what is returned by self.getRmidPsi()) in the core, then
                expands the grid beyond this manually.
            -- A simpler approach would be to just compute the psi_norm(R_mid)
                grid directly from the radial grid.
        
        The latter approach is selected for simplicity.
        
        The units of R_mid are always meters, and are converted by the wrapper
        functions to whatever the user wants.
        
        Args:
            idx (Scalar int):
                The time index to retrieve the flux spline for.
                This is ASSUMED to be a valid index for the first dimension of
                self.getFluxGrid(), otherwise an IndexError will be raised.
        
        Keyword Args:
            k (positive int)
                Polynomial degree of spline to use. Default is 3.
        
        Returns:
            :py:class:`trispline.UnivariateInterpolator` or
                :py:class:`tripline.RectBivariateSpline` depending on whether or
                not the instance was created with the `tspline` keyword.
        """
        if not self._tricubic:
            try:
                return self._RmidToPsiNormSpline[idx][k]
            except KeyError:
                # New approach: create a fairly dense radial grid from the global
                # flux grid to avoid 1d interpolation problems in the core. The
                # bivariate spline seems to be a little more robust in this respect.
                resample_factor = 3
                R_grid = scipy.linspace(
                    # self.getMagR(length_unit='m')[idx],
                    self.getRGrid(length_unit='m')[0],
                    self.getRGrid(length_unit='m')[-1],
                    resample_factor * len(self.getRGrid(length_unit='m'))
                )
                
                psi_norm_on_grid = self.rz2psinorm(
                    R_grid,
                    self.getMagZ(length_unit='m')[idx] * scipy.ones(R_grid.shape),
                    self.getTimeBase()[idx]
                )
                
                spline = UnivariateInterpolator(
                    R_grid, psi_norm_on_grid, k=k
                )
                try:
                    self._RmidToPsiNormSpline[idx][k] = spline
                except KeyError:
                    self._RmidToPsiNormSpline[idx] = {k: spline}
                return self._RmidToPsiNormSpline[idx][k]
        else:
            if self._RmidToPsiNormSpline:
                return self._RmidToPsiNormSpline
            else:
                resample_factor = 3 * len(self.getRGrid(length_unit='m'))
                
                #generate timebase and R_grid through a meshgrid
                t, R_grid = scipy.meshgrid(
                    self.getTimeBase(),
                    scipy.zeros((resample_factor,))
                )
                Z_grid = scipy.dot(
                    scipy.ones((resample_factor, 1)),
                    scipy.atleast_2d(self.getMagZ(length_unit='m'))
                )
                
                for idx in scipy.arange(self.getTimeBase().size):
                    # TODO: This can be done much more efficiently!
                    R_grid[:, idx] = scipy.linspace(
                        self.getRGrid(length_unit='m')[0],
                        self.getRGrid(length_unit='m')[-1],
                        resample_factor
                    )
                
                psi_norm_on_grid = self.rz2psinorm(R_grid, Z_grid, t, each_t=False)
                    
                self._RmidToPsiNormSpline = BivariateInterpolator(
                    t.flatten(),
                    R_grid.flatten(),
                    psi_norm_on_grid.flatten()
                )
                
                return self._RmidToPsiNormSpline


class EFITTree(Equilibrium):
    """Inherits :py:class:`Equilibrium <eqtools.core.Equilibrium>` class. 
    EFIT-specific data handling class for machines using standard EFIT tag 
    names/tree structure with MDSplus.  Constructor and/or data loading may 
    need overriding in a machine-specific implementation.  Pulls EFIT data 
    from selected MDS tree and shot, stores as object attributes.  Each EFIT 
    variable or set of variables is recovered with a corresponding getter 
    method.  Essential data for EFIT mapping are pulled on initialization 
    (e.g. psirz grid).  Additional data are pulled at the first request and 
    stored for subsequent usage.
    
    Intializes :py:class:`EFITTree` object. Pulls data from MDS tree for 
    storage in instance attributes. Core attributes are populated from the MDS 
    tree on initialization. Additional attributes are initialized as None,
    filled on the first request to the object.

    Args:
        shot (integer): Shot number
        tree (string): MDSplus tree to open to fetch EFIT data.
        root (string): Root path for EFIT data in MDSplus tree.
    
    Keyword Args:
        length_unit (string): Sets the base unit used for any
            quantity whose dimensions are length to any power.
            Valid options are:

                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================

            Default is 'm' (all units taken and returned in meters).
        tspline (boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic (boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be 
            monotonically increasing. Default is False (use slower,
            safer method).
    """
    def __init__(self, shot, tree, root, length_unit='m', gfile = 'g_eqdsk', 
                 afile='a_eqdsk', tspline=False, monotonic=True):
        if not _has_MDS:
            print("MDSplus module did not load properly. Exception is below:")
            print(_e_MDS.__class__)
            print(_e_MDS.message)
            print(
                "Most functionality will not be available! (But pickled data "
                "will still be accessible.)"
            )

        super(EFITTree, self).__init__(length_unit=length_unit, tspline=tspline, 
                                       monotonic=monotonic)
        
        self._shot = shot
        self._tree = tree
        self._root = root
        self._gfile = gfile
        self._afile = afile

        self._MDSTree = MDSplus.Tree(self._tree, self._shot)
        
        self._defaultUnits = {}
        
        #initialize None for non-essential data

        #grad-shafranov related parameters
        self._fpol = None
        self._fluxPres = None                                                #pressure on flux surface (psi,t)
        self._ffprim = None
        self._pprime = None                                                  #pressure derivative on flux surface (t,psi)

        #fields
        self._btaxp = None                                                   #Bt on-axis, with plasma (t)
        self._btaxv = None                                                   #Bt on-axis, vacuum (t)
        self._bpolav = None                                                  #avg poloidal field (t)
        self._BCentr = None                                                  #Bt at RCentr, vacuum (for gfiles) (t)

        #plasma current
        self._IpCalc = None                                                  #EFIT-calculated plasma current (t)
        self._IpMeas = None                                                  #measured plasma current (t)
        self._Jp = None                                                      #grid of current density (r,z,t)
        self._currentSign = None                                             #sign of current for entire shot (calculated in moderately kludgey manner)

        #safety factor parameters
        self._q0 = None                                                      #q on-axis (t)
        self._q95 = None                                                     #q at 95% flux (t)
        self._qLCFS = None                                                   #q at LCFS (t)
        self._rq1 = None                                                     #outboard-midplane minor radius of q=1 surface (t)
        self._rq2 = None                                                     #outboard-midplane minor radius of q=2 surface (t)
        self._rq3 = None                                                     #outboard-midplane minor radius of q=3 surface (t)

        #shaping parameters
        self._kappa = None                                                   #LCFS elongation (t)
        self._dupper = None                                                  #LCFS upper triangularity (t)
        self._dlower = None                                                  #LCFS lower triangularity (t)

        #(dimensional) geometry parameters
        self._rmag = None                                                    #major radius, magnetic axis (t)
        self._zmag = None                                                    #Z magnetic axis (t)
        self._aLCFS = None                                                   #outboard-midplane minor radius (t)
        self._RmidLCFS = None                                                #outboard-midplane major radius (t)
        self._areaLCFS = None                                                #LCFS surface area (t)
        self._RLCFS = None                                                   #R-positions of LCFS (t,n)
        self._ZLCFS = None                                                   #Z-positions of LCFS (t,n)
        self._RCentr = None                                                  #Radius for BCentr calculation (for gfiles) (t)
        
        #machine geometry parameters
        self._Rlimiter = None                                                #R-positions of vacuum-vessel wall (t)
        self._Zlimiter = None                                                #Z-positions of vacuum-vessel wall (t)

        #calc. normalized-pressure values
        self._betat = None                                                   #EFIT-calc toroidal beta (t)
        self._betap = None                                                   #EFIT-calc avg. poloidal beta (t)
        self._Li = None                                                      #EFIT-calc internal inductance (t)

        #diamagnetic measurements
        self._diamag = None                                                  #diamagnetic flux (t)
        self._betatd = None                                                  #diamagnetic toroidal beta (t)
        self._betapd = None                                                  #diamagnetic poloidal beta (t)
        self._WDiamag = None                                                 #diamagnetic stored energy (t)
        self._tauDiamag = None                                               #diamagnetic energy confinement time (t)

        #energy calculations
        self._WMHD = None                                                    #EFIT-calc stored energy (t)
        self._tauMHD = None                                                  #EFIT-calc energy confinement time (t)
        self._Pinj = None                                                    #EFIT-calc injected power (t)
        self._Wbdot = None                                                   #EFIT d/dt magnetic stored energy (t)
        self._Wpdot = None                                                   #EFIT d/dt plasma stored energy (t)

        #load essential mapping data
        # Set the variables to None first so the loading calls will work right:
        self._time = None                                                    #EFIT timebase
        self._psiRZ = None                                                   #EFIT flux grid (r,z,t)
        self._rGrid = None                                                   #EFIT R-axis (t)
        self._zGrid = None                                                   #EFIT Z-axis (t)
        self._psiLCFS = None                                                 #flux at LCFS (t)
        self._psiAxis = None                                                 #flux at magnetic axis (t)
        self._fluxVol = None                                                 #volume within flux surface (t,psi)
        self._volLCFS = None                                                 #volume within LCFS (t)
        self._qpsi = None                                                    #q profile (psi,t)
        self._RmidPsi = None                                                 #max major radius of flux surface (t,psi)
        
        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        self.getTimeBase()
        self.getFluxGrid() # loads _psiRZ, _rGrid and _zGrid at once.
        self.getFluxLCFS()
        self.getFluxAxis()
        self.getVolLCFS()
        self.getQProfile()
        self.getRmidPsi()

    def getTimeBase(self):
        """returns EFIT time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._time is None:
            try:
                timeNode = self._MDSTree.getNode(self._root+self._afile+':time')
                self._time = timeNode.data()
                self._defaultUnits['_time'] = str(timeNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._time.copy()

    def getFluxGrid(self):
        """returns EFIT flux grid.
        
        Note that this method preserves whatever sign convention is used in the
        tree. For C-Mod, this means that the result should be multiplied by
        -1 * :py:meth:`getCurrentSign()` in most cases.
        
        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        #import pdb
        #pdb.set_trace()

        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.getNode(self._root+self._gfile+':psirz')
                self._psiRZ = psinode.data()
                self._rGrid = psinode.dim_of(0).data()
                self._zGrid = psinode.dim_of(1).data()
                self._defaultUnits['_psiRZ'] = str(psinode.units)
                self._defaultUnits['_rGrid'] = str(psinode.dim_of(0).units)
                self._defaultUnits['_zGrid'] = str(psinode.dim_of(1).units)
            except:
                raise ValueError('data retrieval failed.')
        return self._psiRZ.copy()

    def getFluxAxis(self):
        """returns psi on magnetic axis.

        Returns:
            psiAxis (Array): [nt] array of psi on magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiAxis is None:
            try:
                psiAxisNode = self._MDSTree.getNode(self._root+self._afile+':simagx')
                self._psiAxis = psiAxisNode.data()
                self._defaultUnits['_psiAxis'] = str(psiAxisNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._psiAxis.copy()

    def getFluxLCFS(self):
        """returns psi at separatrix.

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiLCFS is None:
            try:
                psiLCFSNode = self._MDSTree.getNode(self._root+self._afile+':sibdry')
                self._psiLCFS = psiLCFSNode.data()
                self._defaultUnits['_psiLCFS'] = str(psiLCFSNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._psiLCFS.copy()

    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS.

        Keyword Args:
            length_unit (String or 3): unit for LCFS volume.  Defaults to 3, 
                denoting default volumetric unit (typically m^3).

        Returns:
            volLCFS (Array): [nt] array of volume within LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree.getNode(self._root+self._afile+':vout')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = str(volLCFSNode.units)
            except:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()
  
    def getQProfile(self):
        """returns profile of safety factor q.

        Returns:
            qpsi (Array): [nt,npsi] array of q on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qpsi is None:
            try:
                qpsiNode = self._MDSTree.getNode(self._root+self._gfile+':qpsi')
                self._qpsi = qpsiNode.data()
                self._defaultUnits['_qpsi'] = str(qpsiNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    def getRmidPsi(self, length_unit=1):
        """returns maximum major radius of each flux surface.

        Keyword Args:
            length_unit (String or 1): unit of Rmid.  Defaults to 1, indicating 
                the default parameter unit (typically m).

        Returns:
            Rmid (Array): [nt,npsi] array of maximum (outboard) major radius of 
            flux surface psi.

        Raises:
            Value Error: if module cannot retrieve data from MDS tree.
        """
        if self._RmidPsi is None:
            try:
                RmidPsiNode = self._MDSTree.getNode(self._root+'fitout:rpres')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = str(RmidPsiNode.units)
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidPsi'], length_unit)
        return unit_factor * self._RmidPsi.copy()
    
    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius.

        Returns:
            magR (Array): [nt] array of major radius of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rmag is None:
            try:
                rmagNode = self._MDSTree.getNode(self._root+self._afile+':rmagx')
                self._rmag = rmagNode.data()
                self._defaultUnits['_rmag'] = str(rmagNode.units)
            except:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()

    def getRGrid(self, length_unit=1):
        """returns EFIT R-axis.

        Returns:
            rGrid (Array): [nr] array of R-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'],
                                                      length_unit)
        return unit_factor * self._rGrid.copy()

    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zmag is None:
            try:
                zmagNode = self._MDSTree.getNode(self._root+self._afile+':zmagx')
                self._zmag = zmagNode.data()
                self._defaultUnits['_zmag'] = str(zmagNode.units)
            except:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zmag'], length_unit)
        return unit_factor * self._zmag.copy()

    def getZGrid(self, length_unit=1):
        """returns EFIT Z-axis.

        Returns:
            zGrid (Array): [nz] array of Z-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zGrid'],
                                                      length_unit)
        return unit_factor * self._zGrid.copy()

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's 
        IDL implementation efit_rz2psi.pro.

        Returns:
            currentSign (Integer): 1 for positive-direction current, -1 for negative.
        """
        if self._currentSign is None:
            self._currentSign = 1 if scipy.mean(self.getIpMeas()) > 1e5 else -1
        return self._currentSign

    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpMeas is None:
            try:
                IpMeasNode = self._MDSTree.getNode(self._root+self._afile+':pasmat')
                self._IpMeas = IpMeasNode.data()
                self._defaultUnits['_IpMeas'] = str(IpMeasNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._IpMeas.copy()

    def getF(self):
        """returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov 
        solutions.
        
        Note that this method preserves whatever sign convention is used in the
        tree. For C-Mod, this means that the result should be multiplied by
        -1 * :py:meth:`getCurrentSign()` in most cases.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fpol is None:
            try:
                fNode = self._MDSTree.getNode(self._root+self._gfile+':fpol')
                self._fpol = fNode.data()
                self._defaultUnits['_fpol'] = str(fNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._fpol.copy()

    def getBtVac(self):
        """Returns vacuum toroidal field on-axis.

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._btaxv is None:
            try:
                btaxvNode = self._MDSTree.getNode(self._root+self._afile+':btaxv')
                self._btaxv = btaxvNode.data()
                self._defaultUnits['_btaxv'] = str(btaxvNode.units)
            except:
                raise ValueError('data retrieval failed.')
        return self._btaxv.copy()


class CModEFITTree(EFITTree):
    """Inherits :py:class:`eqtools.EFIT.EFITTree` class. Machine-specific data
    handling class for Alcator C-Mod. Pulls EFIT data from selected MDS tree
    and shot, stores as object attributes. Each EFIT variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for EFIT mapping are pulled on initialization (e.g. psirz grid). Additional
    data are pulled at the first request and stored for subsequent usage.
    
    Intializes C-Mod version of EFITTree object.  Pulls data from MDS tree for 
    storage in instance attributes.  Core attributes are populated from the MDS 
    tree on initialization.  Additional attributes are initialized as None, 
    filled on the first request to the object.

    Args:
        shot (integer): C-Mod shot index.
    
    Keyword Args:
        tree (string): Optional input for EFIT tree, defaults to 'ANALYSIS'
            (i.e., EFIT data are under \\analysis::top.efit.results).
            For any string TREE (such as 'EFIT20') other than 'ANALYSIS',
            data are taken from \\TREE::top.results.
        length_unit (string): Sets the base unit used for any quantity whose
            dimensions are length to any power. Valid options are:
                
                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================
                
            Default is 'm' (all units taken and returned in meters).
        gfile (string): Optional input for EFIT geqdsk location name, 
            defaults to 'g_eqdsk' (i.e., EFIT data are under
            \\tree::top.results.G_EQDSK)
        afile (string): Optional input for EFIT aeqdsk location name,
            defaults to 'a_eqdsk' (i.e., EFIT data are under 
            \\tree::top.results.A_EQDSK)
        tspline (Boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic (Boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
    """
    def __init__(self, shot, tree='ANALYSIS', length_unit='m', gfile='g_eqdsk', 
                 afile='a_eqdsk', tspline=False, monotonic=True):
        if tree.upper() == 'ANALYSIS':
            root = '\\analysis::top.efit.results.'
        else:
            root = '\\'+tree+'::top.results.'

        super(CModEFITTree, self).__init__(shot, tree, root, 
              length_unit=length_unit, gfile=gfile, afile=afile, 
              tspline=tspline, monotonic=monotonic)
        
        self.getFluxVol() #getFluxVol is called due to wide use on C-Mod
    
    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface.

        Keyword Args:
            length_unit (String or 3): unit for plasma volume.  Defaults to 3, 
                indicating default volumetric unit (typically m^3).

        Returns:
            fluxVol (Array): [nt,npsi] array of volume within flux surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fluxVol is None:
            try:
                fluxVolNode = self._MDSTree.getNode(self._root+'fitout:volp')
                self._fluxVol = fluxVolNode.data()
                # Units aren't properly stored in the tree for this one!
                if fluxVolNode.units != ' ':
                    self._defaultUnits['_fluxVol'] = str(fluxVolNode.units)
                else:
                    self._defaultUnits['_fluxVol'] = 'm^3'
            except:
                raise ValueError('data retrieval failed.')
        # Default units are m^3, but aren't stored in the tree!
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_fluxVol'], length_unit)
        return unit_factor * self._fluxVol.copy()


#######################################

### stuff from trispline in eqtools ###

#######################################

import scipy 
import scipy.interpolate
try:
    from . import _tricub
except:
    # Won't be able to use actual trispline, but still can use other routines.
    pass


class BivariateInterpolator(object):
    """This class provides a wrapper for `scipy.interpolate.CloughTocher2DInterpolator`.
    
    This is necessary because `scipy.interpolate.SmoothBivariateSpline` cannot
    be made to interpolate, and gives inaccurate answers near the boundaries.
    """
    def __init__(self, x, y, z):
        self._ct_interp = scipy.interpolate.CloughTocher2DInterpolator(
            scipy.hstack((scipy.atleast_2d(x).T, scipy.atleast_2d(y).T)),
            z
        )

class UnivariateInterpolator(scipy.interpolate.InterpolatedUnivariateSpline):
    """Interpolated spline class which overcomes the shortcomings of interp1d
    (inaccurate near edges) and InterpolatedUnivariateSpline (can't set NaN
    where it extrapolates).
    """
    def __init__(self, *args, **kwargs):
        self.min_val = kwargs.pop('minval', None)
        self.max_val = kwargs.pop('maxval', None)
        if kwargs.pop('enforce_y', True):
            if self.min_val is None:
                self.min_val = min(args[1])
            if self.max_val is None:
                self.max_val = max(args[1])
        super(UnivariateInterpolator, self).__init__(*args, **kwargs)
