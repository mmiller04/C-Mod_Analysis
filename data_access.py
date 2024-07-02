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
    
    def keep_slices(self, axis, vals, tol=None, keep_mixed=False):
        """Only keep the indices closest to given `vals`.
        
        Parameters
        ----------
        axis : int
            The column in `X` to check values on.
        vals : float or 1-d array
            The value(s) to keep the points that are nearest to.
        keep_mixed : bool, optional
            Set this flag to keep transformed quantities that depend on multiple
            values of `X[:, :, axis]`. Default is False (drop mixed quantities).
        
        Returns
        -------
        still_good : bool
            Returns True if there are still any points left in the channel,
            False otherwise.
        """
        unique_vals = []
        num_unique = []
        for pt in self.X:
            unique_vals += [scipy.unique(pt[:, axis])]
            num_unique += [len(unique_vals[-1])]
        if max(num_unique) > 1:
            if keep_mixed:
                return True
            else:
                return False
        else:
            # TODO: Make sure raveling doesn't have unexpected consequences...
            unique_vals = scipy.asarray(unique_vals).ravel()
            
            keep_idxs = get_nearest_idx(vals, unique_vals)
            if tol is not None:
                keep_idxs = keep_idxs[
                    scipy.absolute(unique_vals[keep_idxs] - vals) <= tol
                ]
            keep_idxs = scipy.unique(keep_idxs)
            
            self.X = self.X[keep_idxs, :, :]
            self.y = self.y[keep_idxs]
            self.err_X = self.err_X[keep_idxs, :, :]
            self.err_y = self.err_y[keep_idxs]
            self.T = self.T[keep_idxs, :]
            
            return True
        
    def average_data(self, axis=0, **kwargs):
        """Average the data along the given `axis`.
        
        Parameters
        ----------
        axis : int, optional
            Axis to average along. Default is 0.
        **kwargs : optional keyword arguments
            All additional kwargs are passed to :py:func:`average_points`.
        """
        reduced_X = scipy.delete(self.X, axis, axis=2)
        reduced_err_X = scipy.delete(self.err_X, axis, axis=2)
        self.X, self.y, self.err_X, self.err_y, self.T = average_points(
            reduced_X,
            self.y,
            reduced_err_X,
            self.err_y,
            T=self.T,
            **kwargs
        )
        self.X = scipy.expand_dims(self.X, axis=0)
        self.y = scipy.expand_dims(self.y, axis=0)
        self.err_X = scipy.expand_dims(self.err_X, axis=0)
        self.err_y = scipy.expand_dims(self.err_y, axis=0)
        self.T = scipy.expand_dims(self.T, axis=0)
    
    def remove_points(self, conditional):
        """Remove points satisfying `conditional`.
        
        Parameters
        ----------
        conditional : array, same shape as `self.y`
            Boolean array with True wherever a point should be removed.
        
        Returns
        -------
        bad_X : array
            The removed `X` values.
        bad_err_X : array
            The uncertainty in the removed `X` values.
        bad_y : array
            The removed `y` values.
        bad_err_y : array
            The uncertainty in the removed `y` values.
        bad_T : array
            The transformation matrix of the removed `y` values.
        """
        keep_idxs = ~conditional
        
        bad_X = self.X[conditional, :, :]
        bad_y = self.y[conditional]
        bad_err_X = self.err_X[conditional, :, :]
        bad_err_y = self.err_y[conditional]
        bad_T = self.T[conditional, :]
        
        self.X = self.X[keep_idxs, :, :]
        self.y = self.y[keep_idxs]
        self.err_X = self.err_X[keep_idxs, :, :]
        self.err_y = self.err_y[keep_idxs]
        self.T = self.T[keep_idxs, :]
        
        return (bad_X, bad_err_X, bad_y, bad_err_y, bad_T)


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



