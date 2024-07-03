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

import eqtools
import scipy

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



class BivariatePlasmaProfile(Profile):
    """Class to represent bivariate (y=f(t, psi)) plasma data.
    
    The first column of `X` is always time. If the abscissa is 'RZ', then the
    second column is `R` and the third is `Z`. Otherwise the second column is
    the desired abscissa (psinorm, etc.).
    """

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

    def remove_edge_points(self, allow_conversion=True):
            """Removes points that are outside the LCFS.
            
            Must be called when the abscissa is a normalized coordinate. Assumes
            that the last column of `self.X` is space: so it will do the wrong
            thing if you have already taken an average along space.
            
            Parameters
            ----------
            allow_conversion : bool, optional
                If True and self.abscissa is 'RZ', then the profile will be
                converted to psinorm and the points will be dropped. Default is True
                (allow conversion).
            """
            if self.X is not None:
                if self.abscissa == 'RZ':
                    if allow_conversion:
                        warnings.warn(
                            "Removal of edge points not supported with abscissa RZ. Will "
                            "convert to psinorm."
                        )
                        self.convert_abscissa('psinorm')
                    else:
                        raise ValueError(
                            "Removal of edge points not supported with abscissa RZ!"
                        )
                if 'r/a' in self.abscissa or 'norm' in self.abscissa:
                    x_out = 1.0
                elif self.abscissa == 'Rmid':
                    if self.X_dim == 1:
                        t_EFIT = self._get_efit_times_to_average()
                        x_out = scipy.mean(self.efit_tree.getRmidOutSpline()(t_EFIT))
                    else:
                        assert self.X_dim == 2
                        x_out = self.efit_tree.getRmidOutSpline()(scipy.asarray(self.X[:, 0]).ravel())
                else:
                    raise ValueError(
                        "Removal of edge points not supported with abscissa %s!" % (self.abscissa,)
                    )
                self.remove_points((self.X[:, -1] >= x_out) | scipy.isnan(self.X[:, -1]))



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
        kwargs['efit_tree'] = eqtools.CModEFITTree(shot)
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
    
    N_ne_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:ne_rz')
    
    t_ne_TS = N_ne_TS.dim_of().data()
    ne_TS = N_ne_TS.data() / 1e20
    dev_ne_TS = electrons.getNode(r'yag_new.results.profiles:ne_err').data() / 1e20
    
    Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data() + Z_shift
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
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, ne, err_y=err_ne, channels={1: channels, 2: channels})
    
    # Remove flagged points:
    p.remove_points(
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
        kwargs['efit_tree'] = eqtools.CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == 'CTS':
            p_list.append(TeCTS(shot, **kwargs))
        elif system == 'ETS':
            p_list.append(TeETS(shot, **kwargs))
        elif system == 'FRCECE':
            p_list.append(TeFRCECE(shot, rate=FRCECE_rate, cutoff=FRCECE_cutoff,
                                   **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
        elif system == 'GPC2':
            p_list.append(TeGPC2(shot, **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
        elif system == 'GPC':
            p_list.append(TeGPC(shot, cutoff=GPC_cutoff, **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
        elif system == 'Mic':
            p_list.append(TeMic(shot, cutoff=GPC_cutoff, **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
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
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, Te, err_y=err_Te, channels={1: channels, 2: channels})
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
        (p.err_y == 0.5) |
        ((p.y == 0.0) & remove_zeros) |
        ((p.y == 0.0) & (p.err_y == 0.029999999329447746)) | # This seems to be an old way of flagging. Could be risky...
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
    
    N_Te_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:Te_rz')
    
    t_Te_TS = N_Te_TS.dim_of().data()
    Te_TS = N_Te_TS.data()
    dev_Te_TS = electrons.getNode(r'yag_new.results.profiles:Te_err').data()
    
    Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data() + Z_shift
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
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, Te, err_y=err_Te, channels={1: channels, 2: channels})
    # Remove flagged points:
    p.remove_points(
        scipy.isnan(p.err_y) |
        scipy.isinf(p.err_y) |
        (p.err_y == 0.0) |
        (p.err_y == 1.0) |
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

def TeTS(shot, **kwargs):
    """Returns a profile representing electron temperature data from the Thomson scattering system.
    
    Includes both core and edge system.
    """
    return Te(shot, include=['CTS', 'ETS'], **kwargs)



##########################

### stuff from eqtools ###

##########################

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
    
    def rz2phinorm(self, *args, **kwargs):
        r"""Calculates the normalized toroidal flux.
        
        Uses the definitions:
        
        .. math::
        
            \texttt{phi} &= \int q(\psi)\,d\psi\\
            \texttt{phi\_norm} &= \frac{\phi}{\phi(a)}
        
        This is based on the IDL version efit_rz2rho.pro by Steve Wolfe.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to phinorm. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to phinorm. If `R` and `Z` are both scalar values,
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
            sqrt (Boolean): Set to True to return the square root of phinorm. 
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
            k (positive int): The degree of polynomial spline interpolation to
                use in converting psinorm to phinorm.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `phinorm` or (`phinorm`, `time_idxs`)
            
            * **phinorm** (`Array or scalar float`) - The normalized toroidal
              flux. If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `phinorm` has this shape as well,
              unless the `make_grid` keyword was True, in which case `phinorm`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `phinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single phinorm value at R=0.6m, Z=0.0m, t=0.26s::
            
                phi_val = Eq_instance.rz2phinorm(0.6, 0, 0.26)
            
            Find phinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                phi_arr = Eq_instance.rz2phinorm([0.6, 0.8], [0, 0], 0.26)
            
            Find phinorm values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                phi_arr = Eq_instance.rz2phinorm(0.6, 0, [0.2, 0.3])
            
            Find phinorm values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                phi_arr = Eq_instance.rz2phinorm([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find phinorm values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                phi_mat = Eq_instance.rz2phinorm(R, Z, 0.2, make_grid=True)
        """
        return self._RZ2Quan(self._getPhiNormSpline, *args, **kwargs)
    
    def rz2volnorm(self, *args, **kwargs):
        """Calculates the normalized flux surface volume.
        
        Based on the IDL version efit_rz2rho.pro by Steve Wolfe.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to volnorm. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to volnorm. If `R` and `Z` are both scalar values,
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
            sqrt (Boolean): Set to True to return the square root of volnorm. 
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
            k (positive int): The degree of polynomial spline interpolation to
                use in converting psinorm to volnorm.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `volnorm` or (`volnorm`, `time_idxs`)
            
            * **volnorm** (`Array or scalar float`) - The normalized volume.
              If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `volnorm` has this shape as well,
              unless the `make_grid` keyword was True, in which case `volnorm`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `volnorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single volnorm value at R=0.6m, Z=0.0m, t=0.26s::
            
                psi_val = Eq_instance.rz2volnorm(0.6, 0, 0.26)
            
            Find volnorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                vol_arr = Eq_instance.rz2volnorm([0.6, 0.8], [0, 0], 0.26)
            
            Find volnorm values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                vol_arr = Eq_instance.rz2volnorm(0.6, 0, [0.2, 0.3])
            
            Find volnorm values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                vol_arr = Eq_instance.rz2volnorm([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find volnorm values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                vol_mat = Eq_instance.rz2volnorm(R, Z, 0.2, make_grid=True)
        """
        return self._RZ2Quan(self._getVolNormSpline, *args, **kwargs)
    
    def rz2rmid(self, *args, **kwargs):
        """Maps the given points to the outboard midplane major radius, Rmid.
        
        Based on the IDL version efit_rz2rmid.pro by Steve Wolfe.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to Rmid. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to Rmid. If `R` and `Z` are both scalar values,
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
            sqrt (Boolean): Set to True to return the square root of Rmid. 
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
                instead of Rmid. Default is False (return major radius, Rmid).            
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
                use in converting psinorm to Rmid.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `Rmid` or (`Rmid`, `time_idxs`)
            
            * **Rmid** (`Array or scalar float`) - The outboard midplan major
              radius. If all of the input arguments are scalar, then a scalar
              is returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `Rmid` has this shape as well,
              unless the `make_grid` keyword was True, in which case `Rmid`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `Rmid`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single Rmid value at R=0.6m, Z=0.0m, t=0.26s::
            
                R_mid_val = Eq_instance.rz2rmid(0.6, 0, 0.26)
            
            Find R_mid values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the `Z` vector must be fully specified,
            even if the values are all the same::
            
                R_mid_arr = Eq_instance.rz2rmid([0.6, 0.8], [0, 0], 0.26)
            
            Find Rmid values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                R_mid_arr = Eq_instance.rz2rmid(0.6, 0, [0.2, 0.3])
            
            Find Rmid values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                R_mid_arr = Eq_instance.rz2rmid([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find Rmid values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                R_mid_mat = Eq_instance.rz2rmid(R, Z, 0.2, make_grid=True)
        """
        
        # Steve Wolfe's version has an extra (linear) interpolation step for
        # small psi_norm. Should check to see if we need this still with the
        # scipy spline. So far looks fine...
        
        # Convert units from meters to desired target but keep units consistent
        # with rho keyword:
        if kwargs.get('rho', False):
            unit_factor = 1
        else:
            unit_factor = self._getLengthConversionFactor('m', kwargs.get('length_unit', 1))
        
        return unit_factor * self._RZ2Quan(self._getRmidSpline, *args, **kwargs)
    
    def rz2roa(self, *args, **kwargs):
        """Maps the given points to the normalized minor radius, r/a.
        
        Based on the IDL version efit_rz2rmid.pro by Steve Wolfe.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to r/a. If `R` and `Z` are both scalar values,
                they are used as the coordinate pair for all of the values in
                `t`. Must have the same shape as `Z` unless the `make_grid`
                keyword is set. If the `make_grid` keyword is True, `R` must
                have exactly one dimension.
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to r/a. If `R` and `Z` are both scalar values,
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
            sqrt (Boolean): Set to True to return the square root of r/a. 
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
            k (positive int): The degree of polynomial spline interpolation to
                use in converting psinorm to Rmid.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `roa` or (`roa`, `time_idxs`)
            
            * **roa** (`Array or scalar float`) - The normalized minor radius.
              If all of the input arguments are scalar, then a scalar
              is returned. Otherwise, a scipy Array is returned. If `R` and `Z`
              both have the same shape then `roa` has this shape as well,
              unless the `make_grid` keyword was True, in which case `roa`
              has shape (len(`Z`), len(`R`)).
            * **time_idxs** (Array with same shape as `roa`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single r/a value at R=0.6m, Z=0.0m, t=0.26s::
            
                roa_val = Eq_instance.rz2roa(0.6, 0, 0.26)
            
            Find r/a values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the
            single time t=0.26s. Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                roa_arr = Eq_instance.rz2roa([0.6, 0.8], [0, 0], 0.26)
            
            Find r/a values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]::
            
                roa_arr = Eq_instance.rz2roa(0.6, 0, [0.2, 0.3])
            
            Find r/a values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s)::
            
                roa_arr = Eq_instance.rz2roa([0.6, 0.5], [0, 0.2], [0.2, 0.3], each_t=False)
            
            Find r/a values on grid defined by 1D vector of radial positions `R`
            and 1D vector of vertical positions `Z` at time t=0.2s::
            
                roa_mat = Eq_instance.rz2roa(R, Z, 0.2, make_grid=True)
        """
        
        # Steve Wolfe's version has an extra (linear) interpolation step for
        # small psi_norm. Should check to see if we need this still with the
        # scipy spline. So far looks fine...
        kwargs['rho'] = True
        return self._RZ2Quan(self._getRmidSpline, *args, **kwargs)
    
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
    
    def rmid2roa(self, R_mid, t, each_t=True, return_t=False, sqrt=False, blob=None, length_unit=1):
        """Convert the passed (R_mid, t) coordinates into r/a.
        
        Args:
            R_mid (Array-like or scalar float): Values of the outboard midplane
                major radius to map to r/a.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R_mid`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `R_mid`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of r/a. 
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
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `roa` or (`roa`, `time_idxs`)
            
            * **roa** (`Array or scalar float`) - Normalized midplane minor
              radius. If all of the input arguments are scalar, then a scalar
              is returned. Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `roa`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single r/a value at R_mid=0.6m, t=0.26s::
            
                roa_val = Eq_instance.rmid2roa(0.6, 0.26)
            
            Find roa values at R_mid points 0.6m and 0.8m at the
            single time t=0.26s.::
            
                roa_arr = Eq_instance.rmid2roa([0.6, 0.8], 0.26)
            
            Find roa values at R_mid of 0.6m at times t=[0.2s, 0.3s]::
            
                roa_arr = Eq_instance.rmid2roa(0.6, [0.2, 0.3])
            
            Find r/a values at (R_mid, t) points (0.6m, 0.2s) and (0.5m, 0.3s)::
            
                roa_arr = Eq_instance.rmid2roa([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        # TODO: Make this map inboard to outboard!
        
        # It looks like this is never actually called with pre-computed time
        # indices internally, so I am going to not support that functionality
        # for now.
        if blob is not None:
            raise NotImplementedError("Passing of time indices not supported!")
        
        (
            R_mid,
            dum,
            t,
            time_idxs,
            unique_idxs,
            single_time,
            single_val,
            original_shape
        ) = self._processRZt(
            R_mid,
            R_mid,
            t,
            make_grid=False,
            each_t=each_t,
            length_unit=length_unit,
            compute_unique=False,
            convert_only=True
        )
        
        if self._tricubic:
            roa = self._rmid2roa(R_mid, t).reshape(original_shape)
        else:
            if single_time:
                roa = self._rmid2roa(R_mid, time_idxs[0])
                if single_val:
                    roa = roa[0]
                else:
                    roa = roa.reshape(original_shape)
            elif each_t:
                roa = scipy.zeros(
                    scipy.concatenate(([len(time_idxs),], original_shape))
                )
                for idx, t_idx in enumerate(time_idxs):
                    roa[idx] = self._rmid2roa(R_mid, t_idx).reshape(original_shape)
            else:
                roa = self._rmid2roa(R_mid, time_idxs).reshape(original_shape)
        
        if sqrt:
            if roa.ndim == 0:
                if roa < 0:
                    roa = 0.0
            else:
                scipy.place(roa, roa < 0, 0.0)
            roa = scipy.sqrt(roa)
        
        if return_t:
            if self._tricubic:
                return roa, (t, single_time, single_val, original_shape)
            else:
                return roa, (time_idxs, unique_idxs, single_time, single_val, original_shape)
        else:
            return roa
    
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
    
    def rmid2phinorm(self, *args, **kwargs):
        r"""Calculates the normalized toroidal flux.
        
        Uses the definitions:
        
        .. math::
        
            \texttt{phi} &= \int q(\psi)\,d\psi
            
            \texttt{phi\_norm} &= \frac{\phi}{\phi(a)}
            
        This is based on the IDL version efit_rz2rho.pro by Steve Wolfe.
        
        Args:
            R_mid (Array-like or scalar float): Values of the outboard midplane
                major radius to map to phinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R_mid`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `R_mid`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of phinorm. 
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
            `phinorm` or (`phinorm`, `time_idxs`)
            
            * **phinorm** (`Array or scalar float`) - Normalized toroidal flux.
              If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `phinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single phinorm value at R_mid=0.6m, t=0.26s::
            
                phi_val = Eq_instance.rmid2phinorm(0.6, 0.26)
            
            Find phinorm values at R_mid points 0.6m and 0.8m at the single time
            t=0.26s::
            
                phi_arr = Eq_instance.rmid2phinorm([0.6, 0.8], 0.26)
            
            Find phinorm values at R_mid point 0.6m at times t=[0.2s, 0.3s]::
            
                phi_arr = Eq_instance.rmid2phinorm(0.6, [0.2, 0.3])
            
            Find phinorm values at (R, t) points (0.6m, 0.2s) and (0.5m, 0.3s)::
            
                phi_arr = Eq_instance.rmid2phinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._Rmid2Quan(self._getPhiNormSpline, *args, **kwargs)
    
    def rmid2volnorm(self, *args, **kwargs):
        """Calculates the normalized flux surface volume.
        
        Based on the IDL version efit_rz2rho.pro by Steve Wolfe.
        
        Args:
            R_mid (Array-like or scalar float): Values of the outboard midplane
                major radius to map to volnorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `R_mid`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `R_mid`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of volnorm. 
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
            `volnorm` or (`volnorm`, `time_idxs`)
            
            * **volnorm** (`Array or scalar float`) - Normalized volume.
              If all of the input arguments are scalar, then a scalar is
              returned. Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `volnorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single volnorm value at R_mid=0.6m, t=0.26s::
            
                vol_val = Eq_instance.rmid2volnorm(0.6, 0.26)
            
            Find volnorm values at R_mid points 0.6m and 0.8m at the single time
            t=0.26s::
            
                vol_arr = Eq_instance.rmid2volnorm([0.6, 0.8], 0.26)
            
            Find volnorm values at R_mid points 0.6m at times t=[0.2s, 0.3s]::
            
                vol_arr = Eq_instance.rmid2volnorm(0.6, [0.2, 0.3])
            
            Find volnorm values at (R_mid, t) points (0.6m, 0.2s) and (0.5m, 0.3s)::
            
                vol_arr = Eq_instance.rmid2volnorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._Rmid2Quan(self._getVolNormSpline, *args, **kwargs)
    
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
    
    def roa2rmid(self, roa, t, each_t=True, return_t=False, blob=None, length_unit=1):
        """Convert the passed (r/a, t) coordinates into Rmid.
        
        Args:
            roa (Array-like or scalar float): Values of the normalized minor
                radius to map to Rmid.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `roa`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `roa`.
        
        Keyword Args:
            each_t (Boolean): When True, the elements in `roa` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `roa`
                or be a scalar. Default is True (evaluate ALL `roa` at EACH
                element in `t`).
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
            
            Find single R_mid value at r/a=0.6, t=0.26s::
            
                R_mid_val = Eq_instance.roa2rmid(0.6, 0.26)
            
            Find R_mid values at r/a points 0.6 and 0.8 at the
            single time t=0.26s.::
            
                R_mid_arr = Eq_instance.roa2rmid([0.6, 0.8], 0.26)
            
            Find R_mid values at r/a of 0.6 at times t=[0.2s, 0.3s]::
            
                R_mid_arr = Eq_instance.roa2rmid(0.6, [0.2, 0.3])
            
            Find R_mid values at (roa, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                R_mid_arr = Eq_instance.roa2rmid([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        # It looks like this is never actually called with pre-computed time
        # indices internally, so I am going to not support that functionality
        # for now.
        if blob is not None:
            raise NotImplementedError("Passing of time indices not supported!")
        
        (
            roa,
            dum,
            t,
            time_idxs,
            unique_idxs,
            single_time,
            single_val,
            original_shape
        ) = self._processRZt(
            roa,
            roa,
            t,
            make_grid=False,
            each_t=each_t,
            length_unit=length_unit,
            compute_unique=False,
            check_space=False
        )
        
        if self._tricubic:
            R_mid = self._roa2rmid(roa, t).reshape(original_shape)
        else:
            if single_time:
                R_mid = self._roa2rmid(roa, time_idxs[0])
                if single_val:
                    R_mid = R_mid[0]
                else:
                    R_mid = R_mid.reshape(original_shape)
            elif each_t:
                R_mid = scipy.zeros(
                    scipy.concatenate(([len(time_idxs),], original_shape))
                )
                for idx, t_idx in enumerate(time_idxs):
                    R_mid[idx] = self._roa2rmid(roa, t_idx).reshape(original_shape)
            else:
                R_mid = self._roa2rmid(roa, time_idxs).reshape(original_shape)
        
        R_mid *= self._getLengthConversionFactor('m', length_unit)
        
        if return_t:
            if self._tricubic:
                return R_mid, (t, single_time, single_val, original_shape)
            else:
                return R_mid, (time_idxs, unique_idxs, single_time, single_val, original_shape)
        else:
            return R_mid
    
    def roa2psinorm(self, *args, **kwargs):
        """Convert the passed (r/a, t) coordinates into psinorm.
        
        Args:
            roa (Array-like or scalar float): Values of the normalized minor
                radius to map to psinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `roa`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `roa`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of psinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `roa` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `roa`
                or be a scalar. Default is True (evaluate ALL `roa` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
                
        Returns:
            `psinorm` or (`psinorm`, `time_idxs`)
            
            * **psinorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `psinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value at r/a=0.6, t=0.26s::
            
                psinorm_val = Eq_instance.roa2psinorm(0.6, 0.26)
            
            Find psinorm values at r/a points 0.6 and 0.8 at the
            single time t=0.26s.::
            
                psinorm_arr = Eq_instance.roa2psinorm([0.6, 0.8], 0.26)
            
            Find psinorm values at r/a of 0.6 at times t=[0.2s, 0.3s]::
            
                psinorm_arr = Eq_instance.roa2psinorm(0.6, [0.2, 0.3])
            
            Find psinorm values at (roa, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                psinorm_arr = Eq_instance.roa2psinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self.roa2rho('psinorm', *args, **kwargs)
    
    def roa2phinorm(self, *args, **kwargs):
        """Convert the passed (r/a, t) coordinates into phinorm.
        
        Args:
            roa (Array-like or scalar float): Values of the normalized minor
                radius to map to phinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `roa`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `roa`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of phinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `roa` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `roa`
                or be a scalar. Default is True (evaluate ALL `roa` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
                
        Returns:
            `phinorm` or (`phinorm`, `time_idxs`)
            
            * **phinorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `phinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single phinorm value at r/a=0.6, t=0.26s::
            
                phinorm_val = Eq_instance.roa2phinorm(0.6, 0.26)
            
            Find phinorm values at r/a points 0.6 and 0.8 at the
            single time t=0.26s.::
            
                phinorm_arr = Eq_instance.roa2phinorm([0.6, 0.8], 0.26)
            
            Find phinorm values at r/a of 0.6 at times t=[0.2s, 0.3s]::
            
                phinorm_arr = Eq_instance.roa2phinorm(0.6, [0.2, 0.3])
            
            Find phinorm values at (roa, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                phinorm_arr = Eq_instance.roa2phinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self.roa2rho('phinorm', *args, **kwargs)
    
    def roa2volnorm(self, *args, **kwargs):
        """Convert the passed (r/a, t) coordinates into volnorm.
        
        Args:
            roa (Array-like or scalar float): Values of the normalized minor
                radius to map to volnorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `roa`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `roa`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of volnorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `roa` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `roa`
                or be a scalar. Default is True (evaluate ALL `roa` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
                
        Returns:
            `volnorm` or (`volnorm`, `time_idxs`)
            
            * **volnorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `volnorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single volnorm value at r/a=0.6, t=0.26s::
            
                volnorm_val = Eq_instance.roa2volnorm(0.6, 0.26)
            
            Find volnorm values at r/a points 0.6 and 0.8 at the
            single time t=0.26s.::
            
                volnorm_arr = Eq_instance.roa2volnorm([0.6, 0.8], 0.26)
            
            Find volnorm values at r/a of 0.6 at times t=[0.2s, 0.3s]::
            
                volnorm_arr = Eq_instance.roa2volnorm(0.6, [0.2, 0.3])
            
            Find volnorm values at (roa, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                volnorm_arr = Eq_instance.roa2volnorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self.roa2rho('volnorm', *args, **kwargs)
    
    def roa2rho(self, method, *args, **kwargs):
        r"""Convert the passed (r/a, t) coordinates into one of several coordinates.
        
        Args:
            method (String): Indicates which coordinates to convert to.
                Valid options are:
                
                    ======= =================================
                    psinorm Normalized poloidal flux
                    phinorm Normalized toroidal flux
                    volnorm Normalized volume
                    Rmid    Midplane major radius
                    q       Safety factor
                    F       Flux function :math:`F=RB_{\phi}`
                    FFPrime Flux function :math:`FF'`
                    p       Pressure
                    pprime  Pressure gradient
                    v       Flux surface volume
                    ======= =================================
                
                Additionally, each valid option may be prepended with 'sqrt'
                to specify the square root of the desired unit.
            roa (Array-like or scalar float): Values of the normalized minor
                radius to map to rho.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `roa`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `roa`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of rho. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `roa` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `roa`
                or be a scalar. Default is True (evaluate ALL `roa` at EACH
                element in `t`).
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
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value at r/a=0.6, t=0.26s::
            
                psi_val = Eq_instance.roa2rho('psinorm', 0.6, 0.26)
            
            Find psinorm values at r/a points 0.6 and 0.8 at the
            single time t=0.26s::
            
                psi_arr = Eq_instance.roa2rho('psinorm', [0.6, 0.8], 0.26)
            
            Find psinorm values at r/a of 0.6 at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.roa2rho('psinorm', 0.6, [0.2, 0.3])
            
            Find psinorm values at (r/a, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                psi_arr = Eq_instance.roa2rho('psinorm', [0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        if method == 'Rmid':
            return self.roa2rmid(*args, **kwargs)
        else:
            kwargs['convert_roa'] = True
            return self.rmid2rho(method, *args, **kwargs)
    
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
    
    def psinorm2roa(self, psi_norm, t, **kwargs):
        """Calculates the normalized minor radius location corresponding to the passed psi_norm (normalized poloidal flux) values.
        
        Args:
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to r/a.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `psi_norm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `psi_norm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of r/a. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `psi_norm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `psi_norm` or be
                a scalar. Default is True (evaluate ALL `psi_norm` at EACH element in
                `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            
        Returns:
            `roa` or (`roa`, `time_idxs`)
        
            * **roa** (`Array or scalar float`) - Normalized midplane minor
              radius. If all of the input arguments are scalar, then a scalar
              is returned. Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `roa`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single r/a value for psinorm=0.7, t=0.26s::
            
                roa_val = Eq_instance.psinorm2roa(0.7, 0.26)
            
            Find r/a values at psi_norm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                roa_arr = Eq_instance.psinorm2roa([0.5, 0.7], 0.26)
            
            Find r/a values at psi_norm=0.5 at times t=[0.2s, 0.3s]::
            
                roa_arr = Eq_instance.psinorm2roa(0.5, [0.2, 0.3])
            
            Find r/a values at (psinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                roa_arr = Eq_instance.psinorm2roa([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        kwargs['rho'] = True
        return self._psinorm2Quan(self._getRmidSpline, psi_norm, t, **kwargs)
    
    def psinorm2volnorm(self, psi_norm, t, **kwargs):
        """Calculates the normalized volume corresponding to the passed psi_norm (normalized poloidal flux) values.
        
        Args:
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to volnorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `psi_norm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `psi_norm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of volnorm. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `psi_norm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `psi_norm` or be
                a scalar. Default is True (evaluate ALL `psi_norm` at EACH element in
                `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
            
        Returns:
            `volnorm` or (`volnorm`, `time_idxs`)
        
            * **volnorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `volnorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single volnorm value for psinorm=0.7, t=0.26s::
            
                volnorm_val = Eq_instance.psinorm2volnorm(0.7, 0.26)
            
            Find volnorm values at psi_norm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                volnorm_arr = Eq_instance.psinorm2volnorm([0.5, 0.7], 0.26)
            
            Find volnorm values at psi_norm=0.5 at times t=[0.2s, 0.3s]::
            
                volnorm_arr = Eq_instance.psinorm2volnorm(0.5, [0.2, 0.3])
            
            Find volnorm values at (psinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                volnorm_arr = Eq_instance.psinorm2volnorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._psinorm2Quan(self._getVolNormSpline, psi_norm, t, **kwargs)
    
    def psinorm2phinorm(self, psi_norm, t, **kwargs):
        """Calculates the normalized toroidal flux corresponding to the passed psi_norm (normalized poloidal flux) values.
        
        Args:
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to phinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `psi_norm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `psi_norm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of phinorm. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `psi_norm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `psi_norm` or be
                a scalar. Default is True (evaluate ALL `psi_norm` at EACH element in
                `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `phinorm` or (`phinorm`, `time_idxs`)
            
            * **phinorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `phinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single phinorm value for psinorm=0.7, t=0.26s::
            
                phinorm_val = Eq_instance.psinorm2phinorm(0.7, 0.26)
                
            Find phinorm values at psi_norm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                phinorm_arr = Eq_instance.psinorm2phinorm([0.5, 0.7], 0.26)
            
            Find phinorm values at psi_norm=0.5 at times t=[0.2s, 0.3s]::
            
                phinorm_arr = Eq_instance.psinorm2phinorm(0.5, [0.2, 0.3])
            
            Find phinorm values at (psinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                phinorm_arr = Eq_instance.psinorm2phinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._psinorm2Quan(self._getPhiNormSpline, psi_norm, t, **kwargs)
    
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
    
    def phinorm2psinorm(self, phinorm, t, **kwargs):
        """Calculates the normalized poloidal flux corresponding to the passed phinorm (normalized toroidal flux) values.
        
        Args:
            phinorm (Array-like or scalar float): Values of the normalized
                toroidal flux to map to psinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `phinorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `phinorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of psinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `phinorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `phinorm`
                or be a scalar. Default is True (evaluate ALL `phinorm` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `psinorm` or (`psinorm`, `time_idxs`)
            
            * **psinorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `psinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value for phinorm=0.7, t=0.26s::
            
                psinorm_val = Eq_instance.phinorm2psinorm(0.7, 0.26)
            
            Find psinorm values at phinorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                psinorm_arr = Eq_instance.phinorm2psinorm([0.5, 0.7], 0.26)
            
            Find psinorm values at phinorm=0.5 at times t=[0.2s, 0.3s]::
            
                psinorm_arr = Eq_instance.phinorm2psinorm(0.5, [0.2, 0.3])
            
            Find psinorm values at (phinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                psinorm_arr = Eq_instance.phinorm2psinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._psinorm2Quan(self._getPhiNormToPsiNormSpline, phinorm, t, **kwargs)
    
    def phinorm2volnorm(self, *args, **kwargs):
        """Calculates the normalized flux surface volume corresponding to the passed phinorm (normalized toroidal flux) values.
        
        Args:
            phinorm (Array-like or scalar float): Values of the normalized
                toroidal flux to map to volnorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `phinorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `phinorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of volnorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `phinorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `phinorm`
                or be a scalar. Default is True (evaluate ALL `phinorm` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `volnorm` or (`volnorm`, `time_idxs`)
            
            * **volnorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `volnorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single volnorm value for phinorm=0.7, t=0.26s::
            
                volnorm_val = Eq_instance.phinorm2volnorm(0.7, 0.26)
            
            Find volnorm values at phinorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                volnorm_arr = Eq_instance.phinorm2volnorm([0.5, 0.7], 0.26)
            
            Find volnorm values at phinorm=0.5 at times t=[0.2s, 0.3s]::
            
                volnorm_arr = Eq_instance.phinorm2volnorm(0.5, [0.2, 0.3])
            
            Find volnorm values at (phinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                volnorm_arr = Eq_instance.phinorm2volnorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._phinorm2Quan(self._getVolNormSpline, *args, **kwargs)
    
    def phinorm2rmid(self, *args, **kwargs):
        """Calculates the mapped outboard midplane major radius corresponding to the passed phinorm (normalized toroidal flux) values.
        
        Args:
            phinorm (Array-like or scalar float): Values of the normalized
                toroidal flux to map to Rmid.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `phinorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `phinorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of Rmid. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `phinorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `phinorm`
                or be a scalar. Default is True (evaluate ALL `phinorm` at EACH
                element in `t`).
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
            
            Find single Rmid value for phinorm=0.7, t=0.26s::
            
                Rmid_val = Eq_instance.phinorm2rmid(0.7, 0.26)
            
            Find Rmid values at phinorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                Rmid_arr = Eq_instance.phinorm2rmid([0.5, 0.7], 0.26)
            
            Find Rmid values at phinorm=0.5 at times t=[0.2s, 0.3s]::
            
                Rmid_arr = Eq_instance.phinorm2rmid(0.5, [0.2, 0.3])
            
            Find Rmid values at (phinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                Rmid_arr = Eq_instance.phinorm2rmid([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        # Convert units from meters to desired target but keep units consistent
        # with rho keyword:
        if kwargs.get('rho', False):
            unit_factor = 1
        else:
            unit_factor = self._getLengthConversionFactor('m', kwargs.get('length_unit', 1))
        
        return unit_factor * self._phinorm2Quan(
            self._getRmidSpline,
            *args,
            **kwargs
        )
    
    def phinorm2roa(self, phi_norm, t, **kwargs):
        """Calculates the normalized minor radius corresponding to the passed phinorm (normalized toroidal flux) values.
        
        Args:
            phinorm (Array-like or scalar float): Values of the normalized
                toroidal flux to map to r/a.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `phinorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `phinorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of r/a. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `phinorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `phinorm`
                or be a scalar. Default is True (evaluate ALL `phinorm` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `roa` or (`roa`, `time_idxs`)
            
            * **roa** (`Array or scalar float`) - Normalized midplane minor
              radius. If all of the input arguments are scalar, then a scalar
              is returned. Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `roa`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single r/a value for phinorm=0.7, t=0.26s::
            
                roa_val = Eq_instance.phinorm2roa(0.7, 0.26)
            
            Find r/a values at phinorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                roa_arr = Eq_instance.phinorm2roa([0.5, 0.7], 0.26)
            
            Find r/a values at phinorm=0.5 at times t=[0.2s, 0.3s]::
            
                roa_arr = Eq_instance.phinorm2roa(0.5, [0.2, 0.3])
            
            Find r/a values at (phinorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                roa_arr = Eq_instance.phinorm2roa([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        kwargs['rho'] = True
        return self._phinorm2Quan(self._getRmidSpline, phi_norm, t, **kwargs)
    
    def phinorm2rho(self, method, *args, **kwargs):
        r"""Convert the passed (phinorm, t) coordinates into one of several coordinates.
        
        Args:
            method (String): Indicates which coordinates to convert to.
                Valid options are:
                
                    ======= =================================
                    psinorm Normalized poloidal flux
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
            phinorm (Array-like or scalar float): Values of the normalized
                toroidal flux to map to rho.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `phinorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `phinorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of rho. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `phinorm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `phinorm` or be
                a scalar. Default is True (evaluate ALL `phinorm` at EACH element in
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
            
            Find single psinorm value at phinorm=0.6, t=0.26s::
            
                psi_val = Eq_instance.phinorm2rho('psinorm', 0.6, 0.26)
            
            Find psinorm values at phinorm of 0.6 and 0.8 at the
            single time t=0.26s::
            
                psi_arr = Eq_instance.phinorm2rho('psinorm', [0.6, 0.8], 0.26)
            
            Find psinorm values at phinorm of 0.6 at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.phinorm2rho('psinorm', 0.6, [0.2, 0.3])
            
            Find psinorm values at (phinorm, t) points (0.6, 0.2s) and (0.5m, 0.3s)::
            
                psi_arr = Eq_instance.phinorm2rho('psinorm', [0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        
        if method.startswith('sqrt'):
            kwargs['sqrt'] = True
            method = method[4:]
        
        if method == 'psinorm':
            return self.phinorm2psinorm(*args, **kwargs)
        elif method == 'volnorm':
            return self.phinorm2volnorm(*args, **kwargs)
        elif method == 'Rmid':
            return self.phinorm2rmid(*args, **kwargs)
        elif method == 'r/a':
            return self.phinorm2roa(*args, **kwargs)
        elif method == 'q':
            return self.phinorm2q(*args, **kwargs)
        elif method == 'F':
            return self.phinorm2F(*args, **kwargs)
        elif method == 'FFPrime':
            return self.phinorm2FFPrime(*args, **kwargs)
        elif method == 'p':
            return self.phinorm2p(*args, **kwargs)
        elif method == 'pprime':
            return self.phinorm2pprime(*args, **kwargs)
        elif method == 'v':
            return self.phinorm2v(*args, **kwargs)
        else:
            raise ValueError("phinorm2rho: Unsupported normalized coordinate method '%s'!" % method)
    
    def volnorm2psinorm(self, *args, **kwargs):
        """Calculates the normalized poloidal flux corresponding to the passed volnorm (normalized flux surface volume) values.
        
        Args:
            volnorm (Array-like or scalar float): Values of the normalized
                flux surface volume to map to psinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `volnorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `volnorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of psinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `volnorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `volnorm`
                or be a scalar. Default is True (evaluate ALL `volnorm` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `psinorm` or (`psinorm`, `time_idxs`)
            
            * **psinorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `psinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single psinorm value for volnorm=0.7, t=0.26s::
            
                psinorm_val = Eq_instance.volnorm2psinorm(0.7, 0.26)
            
            Find psinorm values at volnorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                psinorm_arr = Eq_instance.volnorm2psinorm([0.5, 0.7], 0.26)
            
            Find psinorm values at volnorm=0.5 at times t=[0.2s, 0.3s]::
            
                psinorm_arr = Eq_instance.volnorm2psinorm(0.5, [0.2, 0.3])
            
            Find psinorm values at (volnorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                psinorm_arr = Eq_instance.volnorm2psinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._psinorm2Quan(self._getVolNormToPsiNormSpline, *args, **kwargs)
    
    def volnorm2phinorm(self, *args, **kwargs):
        """Calculates the normalized toroidal flux corresponding to the passed volnorm (normalized flux surface volume) values.
        
        Args:
            volnorm (Array-like or scalar float): Values of the normalized
                flux surface volume to map to phinorm.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `volnorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `volnorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of phinorm. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `volnorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `volnorm`
                or be a scalar. Default is True (evaluate ALL `volnorm` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `phinorm` or (`phinorm`, `time_idxs`)
            
            * **phinorm** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `phinorm`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single phinorm value for volnorm=0.7, t=0.26s::
            
                phinorm_val = Eq_instance.volnorm2phinorm(0.7, 0.26)
            
            Find phinorm values at volnorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                phinorm_arr = Eq_instance.volnorm2phinorm([0.5, 0.7], 0.26)
            
            Find phinorm values at volnorm=0.5 at times t=[0.2s, 0.3s]::
            
                phinorm_arr = Eq_instance.volnorm2phinorm(0.5, [0.2, 0.3])
            
            Find phinorm values at (volnorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                phinorm_arr = Eq_instance.volnorm2phinorm([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        return self._volnorm2Quan(self._getPhiNormSpline, *args, **kwargs)
    
    def volnorm2rmid(self, *args, **kwargs):
        """Calculates the mapped outboard midplane major radius corresponding to the passed volnorm (normalized flux surface volume) values.
        
        Args:
            volnorm (Array-like or scalar float): Values of the normalized
                flux surface volume to map to Rmid.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `volnorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `volnorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of Rmid. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `volnorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `volnorm`
                or be a scalar. Default is True (evaluate ALL `volnorm` at EACH
                element in `t`).
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
            
            Find single Rmid value for volnorm=0.7, t=0.26s::
            
                Rmid_val = Eq_instance.volnorm2rmid(0.7, 0.26)
            
            Find Rmid values at volnorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                Rmid_arr = Eq_instance.volnorm2rmid([0.5, 0.7], 0.26)
            
            Find Rmid values at volnorm=0.5 at times t=[0.2s, 0.3s]::
            
                Rmid_arr = Eq_instance.volnorm2rmid(0.5, [0.2, 0.3])
            
            Find Rmid values at (volnorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                Rmid_arr = Eq_instance.volnorm2rmid([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        # Convert units from meters to desired target but keep units consistent
        # with rho keyword:
        if kwargs.get('rho', False):
            unit_factor = 1
        else:
            unit_factor = self._getLengthConversionFactor('m', kwargs.get('length_unit', 1))
        
        return unit_factor * self._volnorm2Quan(self._getRmidSpline, *args, **kwargs)
    
    def volnorm2roa(self, *args, **kwargs):
        """Calculates the normalized minor radius corresponding to the passed volnorm (normalized flux surface volume) values.
        
        Args:
            volnorm (Array-like or scalar float): Values of the normalized
                flux surface volume to map to r/a.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `volnorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `volnorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of r/a. 
                Only the square root of positive values is taken. Negative 
                values are replaced with zeros, consistent with Steve Wolfe's
                IDL implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `volnorm` are evaluated 
                at each value in `t`. If True, `t` must have only one dimension
                (or be a scalar). If False, `t` must match the shape of `volnorm`
                or be a scalar. Default is True (evaluate ALL `volnorm` at EACH
                element in `t`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            return_t (Boolean): Set to True to return a tuple of (`rho`,
                `time_idxs`), where `time_idxs` is the array of time indices
                actually used in evaluating `rho` with nearest-neighbor
                interpolation. (This is mostly present as an internal helper.)
                Default is False (only return `rho`).
        
        Returns:
            `roa` or (`roa`, `time_idxs`)
            
            * **roa** (`Array or scalar float`) - The converted coordinates. If
              all of the input arguments are scalar, then a scalar is returned.
              Otherwise, a scipy Array is returned.
            * **time_idxs** (Array with same shape as `roa`) - The indices 
              (in :py:meth:`self.getTimeBase`) that were used for
              nearest-neighbor interpolation. Only returned if `return_t` is
              True.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class.
            
            Find single r/a value for volnorm=0.7, t=0.26s::
            
                roa_val = Eq_instance.volnorm2roa(0.7, 0.26)
            
            Find r/a values at volnorm values of 0.5 and 0.7 at the single time
            t=0.26s::
            
                roa_arr = Eq_instance.volnorm2roa([0.5, 0.7], 0.26)
            
            Find r/a values at volnorm=0.5 at times t=[0.2s, 0.3s]::
            
                roa_arr = Eq_instance.volnorm2roa(0.5, [0.2, 0.3])
            
            Find r/a values at (volnorm, t) points (0.6, 0.2s) and (0.5, 0.3s)::
            
                roa_arr = Eq_instance.volnorm2roa([0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        kwargs['rho'] = True
        return self._volnorm2Quan(self._getRmidSpline, *args, **kwargs)
    
    def volnorm2rho(self, method, *args, **kwargs):
        r"""Convert the passed (volnorm, t) coordinates into one of several coordinates.
        
        Args:
            method (String): Indicates which coordinates to convert to.
                Valid options are:
                
                    ======= =================================
                    psinorm Normalized poloidal flux
                    phinorm Normalized toroidal flux
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
            volnorm (Array-like or scalar float): Values of the normalized
                flux surface volume to map to rho.
            t (Array-like or scalar float): Times to perform the conversion at.
                If `t` is a single value, it is used for all of the elements of
                `volnorm`. If the `each_t` keyword is True, then `t` must be scalar
                or have exactly one dimension. If the `each_t` keyword is False,
                `t` must have the same shape as `volnorm`.
        
        Keyword Args:
            sqrt (Boolean): Set to True to return the square root of rho. Only
                the square root of positive values is taken. Negative values are
                replaced with zeros, consistent with Steve Wolfe's IDL
                implementation efit_rz2rho.pro. Default is False.
            each_t (Boolean): When True, the elements in `volnorm` are evaluated at
                each value in `t`. If True, `t` must have only one dimension (or
                be a scalar). If False, `t` must match the shape of `volnorm` or be
                a scalar. Default is True (evaluate ALL `volnorm` at EACH element in
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
            
            Find single psinorm value at volnorm=0.6, t=0.26s::
            
                psi_val = Eq_instance.volnorm2rho('psinorm', 0.6, 0.26)
            
            Find psinorm values at volnorm of 0.6 and 0.8 at the
            single time t=0.26s::
            
                psi_arr = Eq_instance.volnorm2rho('psinorm', [0.6, 0.8], 0.26)
            
            Find psinorm values at volnorm of 0.6 at times t=[0.2s, 0.3s]::
            
                psi_arr = Eq_instance.volnorm2rho('psinorm', 0.6, [0.2, 0.3])
            
            Find psinorm values at (volnorm, t) points (0.6, 0.2s) and (0.5m, 0.3s)::
            
                psi_arr = Eq_instance.volnorm2rho('psinorm', [0.6, 0.5], [0.2, 0.3], each_t=False)
        """
        
        if method.startswith('sqrt'):
            kwargs['sqrt'] = True
            method = method[4:]
        
        if method == 'psinorm':
            return self.volnorm2psinorm(*args, **kwargs)
        elif method == 'phinorm':
            return self.volnorm2phinorm(*args, **kwargs)
        elif method == 'Rmid':
            return self.volnorm2rmid(*args, **kwargs)
        elif method == 'r/a':
            return self.volnorm2roa(*args, **kwargs)
        elif method == 'q':
            return self.volnorm2q(*args, **kwargs)
        elif method == 'F':
            return self.volnorm2F(*args, **kwargs)
        elif method == 'FFPrime':
            return self.volnorm2FFPrime(*args, **kwargs)
        elif method == 'p':
            return self.volnorm2p(*args, **kwargs)
        elif method == 'pprime':
            return self.volnorm2pprime(*args, **kwargs)
        elif method == 'v':
            return self.volnorm2v(*args, **kwargs)
        else:
            raise ValueError("volnorm2rho: Unsupported normalized coordinate method '%s'!" % method)
    


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




