import astropy.units as u
import numpy as np
from astropy.cosmology import z_at_value
from astropy.table import QTable
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.stats.sampling import NumericalInversePolynomial


class Catalog:
    """Define a distribution from a catalog of astronomical objects.
    Assumes that the catalog contains ra, dec, and redshift."""

    def __init__(
        self,
        catalog: str | QTable,
        ra_column: str = "ra",
        dec_column: str = "dec",
        redshift_column: str = "redshift",
    ):
        # Load data
        self.catalog = catalog
        self.table = self.load_catalog()
        # Validate columns
        self._validate_columns((ra_column, dec_column, redshift_column))
        self.ra_column = ra_column
        self.dec_column = dec_column
        self.redshift_column = redshift_column

    def load_catalog(self):
        """Load catalog from file to QTable

        :return: Catalog data as QTable
        :rtype: astropy.table.QTable
        """
        if isinstance(self.catalog, QTable):
            return self.catalog
        elif isinstance(self.catalog, str):
            return QTable.read(self.catalog)
        else:
            raise ValueError("Catalog must be a file path or an astropy.table.QTable instance.")

    def _validate_columns(self, columns):
        """Ensure whether columns are in table

        :param columns: Iterable of column names to validate
        :type columns: Iterable[str]
        :raises ValueError: If any column is not found in the catalog
        """
        missing_columns = [column for column in columns if column not in self.table.colnames]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in catalog.")

    def sum_over_healpix(self, healpix_map, cosmo):
        """Given a Boolean HEALPix map,
        sum the number of objects in the catalog that fall within the specified voxels.

        :param healpix_map: HEALPix object, with upper/lower distance bounds saved as DISTMIN/DISTMAX
        :type healpix_map: _type_
        :param cosmo: Cosmology instance for distance calculations
        :type cosmo: astropy.cosmology.Cosmology
        """
        # Find tiles for each object in the catalog
        ras = self.table[self.ra_column]
        decs = self.table[self.dec_column]
        tile_indices = healpix_map.get_tile_indices(ras * u.deg, decs * u.deg)
        tiles = healpix_map.table[tile_indices]
        # Check if each object is within the distance bounds of its corresponding tile
        redshifts = self.table[self.redshift_column]
        dls = cosmo.comoving_distance(redshifts).to(u.Mpc).value
        in_bounds = (dls >= tiles["DISTMIN"]) & (dls <= tiles["DISTMAX"])
        # Count the number of objects that are in bounds and within the HEALPix map
        count = np.sum(in_bounds)
        return count

    def _calculate_cartesian(self, cosmo):
        """Convert ra, dec, redshift coordinates to Cartesian representation.
        Will remain in units of redshift.
        """
        ras = self.table[self.ra_column]
        decs = self.table[self.dec_column]
        redshifts = self.table[self.redshift_column]
        # Convert to Cartesian coordinates
        _dls = cosmo.comoving_distance(redshifts).to(u.Mpc).value
        x = _dls * np.cos(np.radians(decs)) * np.cos(np.radians(ras))
        y = _dls * np.cos(np.radians(decs)) * np.sin(np.radians(ras))
        z = _dls * np.sin(np.radians(decs))
        return x, y, z

    def dn_d3x(self, ras, decs, dls, cosmo):
        """Compute the number density in 3D space at the given locations.

        :param ras: Right ascension values
        :type ras: array-like
        :param decs: Declination values
        :type decs: array-like
        :param dLs: Luminosity distance values
        :type dLs: array-like
        :param cosmo: Cosmology instance for distance calculations
        :type cosmo: astropy.cosmology.Cosmology
        """
        # Convert to Cartesian coordinates
        x, y, z = self._calculate_cartesian(cosmo)
        # Calculate 3D KDE model
        positions = np.vstack([x, y, z])
        kde = gaussian_kde(positions)
        # Evaluate KDE at the given positions
        eval_positions = np.vstack(
            [
                dls * np.cos(np.radians(decs)) * np.cos(np.radians(ras)),
                dls * np.cos(np.radians(decs)) * np.sin(np.radians(ras)),
                dls * np.sin(np.radians(decs)),
            ]
        )
        return kde(eval_positions)

    def sample_positions(self, n_samples, rng=None):
        """Generate random samples from the catalog distribution.

        :param n_samples: Number of random samples to generate
        :type n_samples: int
        :param rng: Random number generator instance (optional)
        :type rng: np.random.Generator, optional
        :return: Randomly sampled entries from the catalog
        :rtype: astropy.table.QTable
        """
        # Set rng if needed
        if rng is None:
            rng = np.random.default_rng()
        # Sample random indices from the catalog
        indices = rng.choice(len(self.table), size=n_samples, replace=False)
        sample = self.table[indices]
        # Package and return
        ras = sample[self.ra_column]
        decs = sample[self.dec_column]
        redshifts = sample[self.redshift_column]
        return ras, decs, redshifts


class AnalyticIsotropic:
    """Distribution defined by analytic function dp/ddL.
    Assumes isotropy, so only depends on distance."""

    # NOTE: scipy.stats.powerlaw can be initialized with bounds:
    # pl = powerlaw(a=alpha, loc=lower_bound, scale=upper_bound - lower_bound)

    def __init__(self, los_distance_distribution, n_total):
        """Initialize distribution.
        Distribution should be a scipy.stats distribution object, e.g., scipy.stats.powerlaw.
        n_total is the total number of objects in the volume domain, used for normalization.

        :param distance_distribution: Function defining dp/ddL
        :type distance_distribution: callable
        :param n_total: Total number of objects in the distribution
        :type n_total: int
        """
        self.los_distance_distribution = los_distance_distribution
        self.n_total = n_total

    def sum_over_healpix(self, healpix_map, cosmo):
        """Integrate the distribution over a given HEALPix map.

        :param healpix_map: HEALPix map to integrate over
        :type healpix_map: _type_
        :param cosmo: Cosmology object
        :type cosmo: astropy.cosmology.Cosmology
        """

        # Get the distance bounds for each tile in the HEALPix map
        dist_mins = healpix_map.table["DISTMIN"]
        dist_maxs = healpix_map.table["DISTMAX"]
        # Integrate the distribution over the distance bounds for each tile
        integrals = [
            self.los_distance_distribution.cdf(dist_max) - self.los_distance_distribution.cdf(dist_min)
            for dist_min, dist_max in zip(dist_mins, dist_maxs, strict=True)
        ]
        # Get fraction of sky covered by each tile
        tile_fractions = healpix_map.get_tile_areas() / (4 * np.pi)
        # Sum the integrals to get the total number of objects in the HEALPix map
        total_count = np.nansum(np.array(integrals) * np.array(tile_fractions)) * self.n_total
        return total_count

    def dn_d3x(self, ras, decs, dls):
        """Compute the number density in 3D space at the given locations.
        Ras/decs included to maintain the calling signature, but are not used in the isotropic case.

        :param ras: Right ascension values
        :type ras: array-like
        :param decs: Declination values
        :type decs: array-like
        :param dLs: Luminosity distance values
        :type dLs: array-like
        """
        return self.los_distance_distribution.pdf(dls) / (4 * np.pi * dls**2) * self.n_total

    def _sample_distances(self, n_samples, rng=None):
        """Sample distances based on the dp/ddL distribution.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :param rng: Random number generator instance (optional)
        :type rng: np.random.Generator, optional
        :return: Sampled distances
        :rtype: array-like
        """
        # Set rng if needed
        if rng is None:
            rng = np.random.default_rng()
        # Create a numerical inverse polynomial for sampling
        nip = NumericalInversePolynomial(
            self.los_distance_distribution,
        )
        return nip.rvs(size=n_samples, random_state=rng)

    def sample_positions(self, n_samples, rng=None):
        """Generate random samples from the analytic distribution.

        :param n_samples: Number of random samples to generate
        :type n_samples: int
        :param rng: Random number generator instance (optional)
        :type rng: np.random.Generator, optional
        :return: Randomly sampled positions (ras, decs, dls)
        :rtype: tuple of arrays
        """
        # Set rng if needed
        if rng is None:
            rng = np.random.default_rng()
        # Sample distances based on the dp/ddL distribution
        dls = self._sample_distances(n_samples, rng)
        # Sample random directions uniformly on the sphere
        ras = rng.uniform(0, 360, size=n_samples)
        decs = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n_samples)))
        return ras, decs, dls


class Interpolated:
    """Distribution defined via interpolation from provided grid/values."""

    def __init__(self, dl_grid, dn_d3dl_grid):
        """Initialize an interpolated distribution.

        :param dl_grid: Grid of distance values
        :type dl_grid: array-like
        :param dn_d3dl_grid: Corresponding los dn/d3dl values for the distance grid
        :type dn_d3dl_grid: array-like
        """
        self.dl_grid = dl_grid
        self.dn_d3dl_grid = dn_d3dl_grid
        self.dn_d3dl_interpolator = interp1d(
            self.dl_grid, self.dn_d3dl_grid, kind="linear", fill_value="extrapolate"
        )

    def dn_d3x(self, ras, decs, dls):
        """Compute the number density in 3D space at the given locations.
        Ras/decs included to maintain the calling signature, but are not used in the isotropic case.

        :param ras: Right ascension values
        :type ras: array-like
        :param decs: Declination values
        :type decs: array-like
        :param dLs: Luminosity distance values
        :type dLs: array-like
        """
        dn_d3dl = self.dn_d3dl_interpolator(dls)
        return dn_d3dl

    def sum_over_healpix(self, healpix_map, cosmo):
        """Integrate the distribution over a given HEALPix map.

        :param healpix_map: HEALPix map to integrate over
        :type healpix_map: _type_
        :param cosmo: Cosmology object
        :type cosmo: astropy.cosmology.FlatLambdaCDM
        :return: Total number of objects in the HEALPix map
        :rtype: float
        """

        # Make a distance grid that covers the range of distances in the HEALPix map
        dl_mins = healpix_map.table["DISTMIN"]
        dl_maxs = healpix_map.table["DISTMAX"]
        dl_grid = np.linspace(np.nanmin(dl_mins), np.nanmax(dl_maxs), 1000)
        # Add all the distance bounds to the grid to ensure we capture the edges
        dl_grid = np.unique(np.concatenate([dl_grid, dl_mins, dl_maxs]))
        # Calculate integrand
        dn_d3dl_grid = self.dn_d3dl_interpolator(dl_grid)
        dn_domega_dl_grid = dn_d3dl_grid * dl_grid**2
        # Get trapezoidal contribution for each step in dl_grid
        trap_dl_grid = np.diff(dl_grid) * (dn_domega_dl_grid[:-1] + dn_domega_dl_grid[1:]) / 2
        # Outer product by the tile areas to get total contribution for each voxel
        trap_voxel_grid = np.outer(
            trap_dl_grid, healpix_map.get_tile_areas()
        )  # len(dl_grid)-1 x len(healpix_map.table)
        # Mask out the contributions that are outside the distance bounds for each tile
        dl_mins = healpix_map.table["DISTMIN"]
        dl_maxs = healpix_map.table["DISTMAX"]
        dl_grid_mids = (dl_grid[:-1] + dl_grid[1:]) / 2
        mask = np.greater_equal.outer(dl_grid_mids, dl_mins) & np.less_equal.outer(dl_grid_mids, dl_maxs)
        # Sum over the distance grid and tile areas to get total count
        total_count = np.nansum(trap_voxel_grid[mask])

        return total_count

    def sample_positions(self, n_samples, rng=None):
        """Generate random samples from the interpolated distribution.

        :param n_samples: Number of random samples to generate
        :type n_samples: int
        :param rng: Random number generator instance (optional)
        :type rng: np.random.Generator, optional
        :return: Randomly sampled positions (ras, decs, dls)
        :rtype: tuple of arrays
        """
        # Set rng if needed
        if rng is None:
            rng = np.random.default_rng()
        # Scale density by dl**2
        p_dl = self.dn_d3dl_interpolator(self.dl_grid) * self.dl_grid**2
        # Draw samples from the distribution using inverse transform sampling
        cdf = np.cumsum(p_dl)
        cdf /= cdf[-1]
        dls = np.interp(rng.uniform(0, 1, size=n_samples), cdf, self.dl_grid)
        # Sample random directions uniformly on the sphere
        ras = rng.uniform(0, 360, size=n_samples)
        decs = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n_samples)))
        return ras, decs, dls


class UniformComovingDensity(Interpolated):
    """Special case of Interpolated with a constant comoving number density in 3D space."""

    def __init__(self, dn_d3dm, dl_min, dl_max, cosmo, n_dl=1000):
        """Initialize a constant density distribution.

        :param dn_d3dm: Constant comoving number density in Mpc^-3
        :type dn_d3dm: float
        :param dl_min: Minimum distance for the distribution
        :type dl_min: float
        :param dl_max: Maximum distance for the distribution
        :type dl_max: float
        :param cosmo: Cosmology instance for distance calculations
        :type cosmo: astropy.cosmology.Cosmology
        """

        # Save params
        self.dn_d3dm = dn_d3dm
        self.dl_min = dl_min
        self.dl_max = dl_max
        self.cosmo = cosmo
        # Create a uniform distribution over the specified distance range
        if dl_min == 0:
            dl_grid = np.linspace(dl_min, dl_max, n_dl + 1)[1:]
        else:
            dl_grid = np.linspace(dl_min, dl_max, n_dl)
        dm_grid = cosmo.comoving_distance(z_at_value(cosmo.luminosity_distance, dl_grid * u.Mpc)).value
        dn_d3dm_grid = np.full_like(dm_grid, dn_d3dm)
        ddm_per_ddl_grid = np.gradient(dm_grid, dl_grid)
        dn_d3dl_grid = dn_d3dm_grid * ddm_per_ddl_grid**3
        super().__init__(dl_grid, dn_d3dl_grid)
