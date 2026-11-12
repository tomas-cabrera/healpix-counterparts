import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.table import QTable
from healpix_counterparts.distributions import (
    AnalyticIsotropic,
    Catalog,
    UniformComovingDensity,
)
from scipy.stats import powerlaw

RNG = np.random.default_rng(980123)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

### Initialize mock catalog
# Create a mock catalog with gaussian distribution in space
n_samples = 1000
x = RNG.normal(loc=250, scale=100, size=n_samples)
y = RNG.normal(loc=150, scale=100, size=n_samples)
z = RNG.normal(loc=400, scale=100, size=n_samples)
# Convert to spherical coordinates
dls = np.sqrt(x**2 + y**2 + z**2)
decs = np.degrees(np.arcsin(z / dls))
ras = np.degrees(np.arctan2(y, x)) % 360
redshifts = z_at_value(COSMO.comoving_distance, dls * u.Mpc).value
# Initialize mock catalog
mock_catalog = QTable(
    {
        "ra": ras,
        "dec": decs,
        "redshift": redshifts,
    }
)
# Initialize the Catalog class with the mock catalog
test_catalog = Catalog(catalog=mock_catalog)


def test_catalog_sampling():
    """Test that the Catalog class can sample from a catalog of astronomical objects."""
    # Sample random positions from the catalog
    sampled_ras, sampled_decs, sampled_redshifts = test_catalog.sample_positions(
        n_samples,
        rng=RNG,
    )

    # Check that the sampled positions are within the bounds of the original catalog
    assert np.all(sampled_ras >= 0) and np.all(sampled_ras <= 360)
    assert np.all(sampled_decs >= -90) and np.all(sampled_decs <= 90)
    assert np.all(sampled_redshifts >= 0)


def test_catalog_density_estimation():
    """Test that the Catalog class can estimate the number density in 3D space."""
    # Sample random positions from the catalog
    sampled_ras, sampled_decs, sampled_redshifts = test_catalog.sample_positions(
        n_samples,
        rng=RNG,
    )
    sampled_dls = COSMO.comoving_distance(sampled_redshifts).value

    # Estimate the number density at the sampled positions
    densities = test_catalog.dn_d3x(
        sampled_ras,
        sampled_decs,
        sampled_dls,
        cosmo=COSMO,
    )

    # Check that the densities are non-negative
    assert np.all(densities >= 0)


### Initialize mock analytic distribution
# Create a mock analytic distribution with a simple isotropic model
pl = powerlaw(a=2, loc=0, scale=500)
test_ai = AnalyticIsotropic(pl, n_total=1000000)


def test_ai_sampling():
    """Test that the AnalyticIsotropic class can sample from an analytic distribution."""
    # Sample random positions from the analytic distribution
    sampled_ras, sampled_decs, sampled_dls = test_ai.sample_positions(
        n_samples,
        rng=RNG,
    )

    # Check that the sampled positions are within the expected bounds
    assert np.all(sampled_ras >= 0) and np.all(sampled_ras <= 360)
    assert np.all(sampled_decs >= -90) and np.all(sampled_decs <= 90)
    assert np.all(sampled_dls >= 0) and np.all(sampled_dls <= 500)


def test_ai_density_estimation():
    """Test that the AnalyticIsotropic class can estimate the number density in 3D space."""
    # Sample random positions from the analytic distribution
    sampled_ras, sampled_decs, sampled_dls = test_ai.sample_positions(
        n_samples,
        rng=RNG,
    )

    # Estimate the number density at the sampled positions
    densities = test_ai.dn_d3x(
        sampled_ras,
        sampled_decs,
        sampled_dls,
    )

    # Check that the densities are non-negative
    assert np.all(densities >= 0)


### Initialize constant density analytic distribution
# Create a mock analytic distribution with a constant density in 3D space
test_cmd = UniformComovingDensity(dn_d3dm=10**-4.75, dl_min=0, dl_max=500, cosmo=COSMO)


# def test_constant_density_equivalence():
#     """Test that the ConstantDensity class is equivalent to a uniform distribution in 3D space."""
#     # Sample random positions from the constant density distribution
#     sampled_ras, sampled_decs, sampled_dls = test_cmd.sample_positions(
#         n_samples,
#         rng=RNG,
#     )

#     # Estimate the number density at the sampled positions
#     densities = test_cmd.dn_d3x(
#         sampled_ras,
#         sampled_decs,
#         sampled_dls,
#     )

#     # Check that the densities are approximately equal to the specified constant density
#     assert np.allclose(densities, test_cmd.dn_d3mpc, rtol=1e-2)


def test_constant_density_sampling():
    """Test that the UniformComovingDensity class can sample from a constant density distribution."""
    # Sample random positions from the constant density distribution
    sampled_ras, sampled_decs, sampled_dls = test_cmd.sample_positions(
        n_samples,
        rng=RNG,
    )

    # Check that the sampled positions are within the expected bounds
    assert np.all(sampled_ras >= 0) and np.all(sampled_ras <= 360)
    assert np.all(sampled_decs >= -90) and np.all(sampled_decs <= 90)
    assert np.all(sampled_dls >= test_cmd.dl_min) and np.all(sampled_dls <= test_cmd.dl_max)


def test_constant_density_estimation():
    """Test that the UniformComovingDensity class can estimate the number density in 3D space."""
    # Sample random positions from the constant density distribution
    sampled_ras, sampled_decs, sampled_dls = test_cmd.sample_positions(
        n_samples,
        rng=RNG,
    )

    # Estimate the number density at the sampled positions
    densities = test_cmd.dn_d3x(
        sampled_ras,
        sampled_decs,
        sampled_dls,
    )

    # Check that all the densities are non-negative
    assert np.all(densities >= 0)
