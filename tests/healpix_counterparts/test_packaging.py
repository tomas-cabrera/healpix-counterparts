import healpix_counterparts


def test_version():
    """Check to see that we can get the package version"""
    assert healpix_counterparts.__version__ is not None
