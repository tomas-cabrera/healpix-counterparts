import os.path as pa

from healpix_counterparts import healpix

sample_skymap_path = f"{pa.dirname(__file__)}/test_data/Bilby.S230922g.multiorder.fits"


def test_compute_ci_volume():
    """Test compute_ci_volume method."""
    hp = healpix.HEALPix(sample_skymap_path)
    print(hp.compute_ci_volume(0.9, cache=True))
    assert True


def test_sample_2d_positions():
    """Test sample_2D_positions method."""
    n_samples = 5
    hp = healpix.HEALPix(sample_skymap_path)
    selected_tiles, ras, decs = hp.sample_positions_2d(n_samples=n_samples)
    print(selected_tiles, ras, decs)
    assert len(selected_tiles) == n_samples
    assert all(ras.deg >= 0) and all(ras.deg < 360)
    assert all(decs.deg >= -90) and all(decs.deg <= 90)


def test_sample_3d_positions():
    """Test sample_3D_positions method."""
    n_samples = 5
    hp = healpix.HEALPix(sample_skymap_path)
    selected_tiles, ras, decs, dls = hp.sample_positions_3d(n_samples=n_samples)
    print(selected_tiles, ras, decs, dls)
    assert len(selected_tiles) == n_samples
    assert all(ras.deg >= 0) and all(ras.deg < 360)
    assert all(decs.deg >= -90) and all(decs.deg <= 90)
    assert all(dls > 0)
