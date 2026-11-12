# Test integration of the HEALPix class with the Catalog and AnalyticIsotropic classes
import os.path as pa
from copy import copy

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.table import QTable
from healpix_counterparts import healpix
from healpix_counterparts.distributions import (
    AnalyticIsotropic,
    Catalog,
    UniformComovingDensity,
)
from scipy.stats import powerlaw

RNG = np.random.default_rng(980123)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

# Load skymap
sample_skymap_path = f"{pa.dirname(__file__)}/test_data/Bilby.S230922g.multiorder.fits"
sample_hp = healpix.HEALPix(sample_skymap_path)
sample_hp.compute_ci_volume(0.9, cache=True)

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

### Initialize mock analytic distribution
# Create a mock analytic distribution with a simple isotropic model
pl = powerlaw(a=2, loc=0, scale=500)
test_ai = AnalyticIsotropic(pl, n_total=1000000)
# Create a mock constant density distribution with a constant density in 3D space
test_cmd = UniformComovingDensity(dn_d3dm=10**-4.75, dl_min=0, dl_max=500, cosmo=COSMO)


def test_catalog_sum_over_healpix():
    """Test the integration of the HEALPix class with the Catalog class."""
    # Sum over HEALPix map using the catalog
    count = test_catalog.sum_over_healpix(sample_hp, cosmo=COSMO)
    print(f"Number of objects in the catalog within the HEALPix map: {count}")
    assert count >= 0  # Ensure that the count is non-negative


def test_ai_sum_over_healpix():
    """Test the integration of the HEALPix class with the AnalyticIsotropic class."""
    # Sample random positions from the analytic distribution
    sampled_ras, sampled_decs, sampled_dls = test_ai.sample_positions(n_samples, rng=RNG)
    # Check if sampled positions fall within the HEALPix map
    count = test_ai.sum_over_healpix(sample_hp, cosmo=COSMO)
    print(f"Number of objects within the HEALPix map: {count}")
    assert count >= 0  # Ensure that the count is non-negative


def test_ai_sum_over_healpix_normalization():
    """Test the normalization of the AnalyticIsotropic distribution over the HEALPix map."""

    # Check if sampled positions fall within the HEALPix map
    _hp = copy(sample_hp)
    _hp.table["DISTMIN"] = 0.0
    _hp.table["DISTMAX"] = 1000.0
    count = test_ai.sum_over_healpix(_hp, cosmo=COSMO)
    print(f"Number of objects within the HEALPix map: {count}")
    assert np.isclose(
        count, test_ai.n_total, rtol=0.1
    )  # Check if the count is close to the total number of objects in the distribution


def test_cmd_sum_over_healpix():
    """Test the integration of the HEALPix class with the UniformComovingDensity class."""
    # Check if sampled positions fall within the HEALPix map
    count = test_cmd.sum_over_healpix(sample_hp, cosmo=COSMO)
    print(f"Number of objects within the HEALPix map: {count}")
    assert count >= 0  # Ensure that the count is non-negative
