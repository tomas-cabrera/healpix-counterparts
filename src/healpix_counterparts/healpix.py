import astropy_healpix as ah
import healpy as hp
import ligo.skymap.moc as lsm_moc
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time


class HEALPix:
    """Class to house HEALPix skymaps.
    Generalized to work with either flat or multi-order skymaps.
    Adds two main features:
    - Enables the caching of a calculated CI volume in the skymap Table
    - Facilitates probability-weighted sampling of 2- and 3D positions.
    """

    def __init__(self, path_to_file, time_key=None, order="nested"):
        # Save path
        self.path = path_to_file
        # Load skymap
        self.load(time_key=time_key, order=order)

    def _calc_prob_from_probdensity(self):
        self.table["PROB"] = lsm_moc.uniq2pixarea(self.table["UNIQ"]) * self.table["PROBDENSITY"]

    def _calc_probdensity_from_prob(self):
        self.table["PROBDENSITY"] = self.table["PROB"] / lsm_moc.uniq2pixarea(self.table["UNIQ"])

    def _generate_uniqs(self, order="nested"):
        # Verify that length of table is a valid HEALPix number of tiles
        n_tiles = len(self.table)
        if not hp.isnpixok(n_tiles):
            raise ValueError(f"Number of tiles {n_tiles} is not valid for HEALPix.")
        # Get level from n_tiles
        _nside = ah.npix_to_nside(n_tiles)
        _level = ah.nside_to_level(_nside)
        # Generate uniqs
        ipixs = np.arange(n_tiles)
        if order == "ring":
            ipixs = hp.ring2nest(_nside, ipixs)
        elif order != "nested":
            raise ValueError(f"Order '{order}' not recognized. Use 'nested' or 'ring'.")
        self.table["UNIQ"] = ah.level_ipix_to_uniq(_level, np.arange(n_tiles))

    def _get_time_from_header(self, time_key):
        with fits.open(self.path) as hdul:
            if time_key == "MJD-OBS":
                self.time = Time(hdul[1].header[time_key], format="mjd")  # type: ignore (header attribute not recognized)
            else:
                raise NotImplementedError(f"time_key '{time_key}' not implemented.")

    def load(self, order="nested", time_key=None):
        """Load the skymap as an astropy Table.
        Generates UNIQ, PROB, and PROBDENSITY columns if not present.
        Optionally extracts time from header.

        :param order: HEALPix order, either "nested" or "ring", defaults to "nested"
        :type order: str, optional
        :param time_key: Key to extract time from header, defaults to None
        :type time_key: str, optional
        """
        # Get data
        self.table = Table.read(self.path)
        # Generate uniqs
        if "UNIQ" not in self.table.columns:
            self._generate_uniqs(order=order)
        # Generate probs/probdensities
        if "PROB" not in self.table.columns:
            self._calc_prob_from_probdensity()
        elif "PROBDENSITY" not in self.table.columns:
            self._calc_probdensity_from_prob()
        # Get time if key specified
        if time_key is not None:
            self._get_time_from_header(time_key)

    def compute_ci_volume(self, ci, cache=False):
        """Calculate the volume for the given confidence interval(s).
        If cache=True, also computes and caches the min/max distance for each tile in the CI in the table,
        with the column names "CIDLMIN" and "CIDLMAX"
        (set to np.nan if tile is not in the CI).
        Accepts multiple CIs as a list/array (only caches the first one if cache=True).

        Mostly follows the ligo.skymap implementation, but adds caching functionality.

        :param ci: Confidence interval(s) to compute volume for, between 0 and 1
        :type ci: float or list or np.ndarray
        :param cache: Whether to cache the computed volume in the skymap table, defaults to False
        :type cache: bool, optional
        :raises KeyError: If distance columns are not found in the table
        :return: Volume(s) corresponding to the confidence interval(s)
        :rtype: float or np.ndarray
        """
        # Raise error if distance columns not in table
        missing_cols = []
        for c in ["DISTMU", "DISTSIGMA", "DISTNORM"]:
            if c not in self.table.columns:
                missing_cols.append(c)
        if missing_cols:
            raise KeyError(f"{missing_cols} not found in HEALPix table.")

        # Generate distance grid using average distance (following ligo.skymap)
        mask = np.logical_and(np.isfinite(self.table["DISTMU"]), ~np.isnan(self.table["DISTMU"]))
        dl_mean = np.average(self.table["DISTMU"][mask], weights=self.table["PROB"][mask])
        dl_grid = np.linspace(0, dl_mean * 6, 1001)

        # Calculate distance pdf
        # NOTE: This excludes the dl**2 term in the ansatz, as we later multiply by the volume element
        pdf_dl = (
            self.table["DISTNORM"]
            / ((2 * np.pi) ** 0.5 * self.table["DISTSIGMA"])
            * np.exp(
                -0.5 * ((dl_grid[:, None] - self.table["DISTMU"][None, :]) / self.table["DISTSIGMA"]) ** 2
            )
        )  # n_dl x n_tiles
        # Take average to represent trapezoidal integral
        pdf_dl = (pdf_dl[:-1, :] + pdf_dl[1:, :]) / 2

        # Calculate voxel pdf (n_dl x n_tiles)
        pdf_voxel = self.table["PROBDENSITY"] * pdf_dl

        # Calculate voxel probs (n_dl x n_tiles) by multiplying by volume element
        voxel_volumes = (
            lsm_moc.uniq2pixarea(self.table["UNIQ"]) * np.array([dl_grid[1:] ** 3 - dl_grid[:-1] ** 3]).T / 3
        )
        prob_voxel = pdf_voxel * voxel_volumes

        # Get array of indices, ravel all
        voxel_indices_ravel = np.unravel_index(np.arange(prob_voxel.size), prob_voxel.shape)
        pdf_voxel_ravel = pdf_voxel.ravel()
        prob_voxel_ravel = prob_voxel.ravel()
        voxel_volumes_ravel = voxel_volumes.ravel()

        # Sort by pdf
        sort_indices = np.argsort(pdf_voxel_ravel)[::-1]
        pdf_voxel_ravel = pdf_voxel_ravel[sort_indices]
        prob_voxel_ravel = prob_voxel_ravel[sort_indices]
        voxel_volumes_ravel = voxel_volumes_ravel[sort_indices]
        voxel_indices_ravel = tuple(
            [vir[sort_indices] for vir in voxel_indices_ravel]
        )  # 2 x n_voxels ([row_indices], [column_indices])

        # Cumsum probability
        cumsum_probs = np.concatenate(([0], np.cumsum(prob_voxel_ravel)))
        cumsum_volumes = np.concatenate(([0], np.cumsum(voxel_volumes_ravel)))

        # Interpolate volume
        ci_vols = np.interp(ci, cumsum_probs, cumsum_volumes)

        # Cache volume if desired
        if cache:
            # Warn if multiple cis given
            if isinstance(ci, list | np.ndarray):
                print("Warning: caching only first CI volume for multiple CIs.")
                cache_ci = ci[0]
            else:
                cache_ci = ci
            # Get indices of voxels in CI
            ci_mask = cumsum_probs[1:] <= cache_ci
            # Add next index if not last (greedy volume)
            if not ci_mask[-1]:
                # If there are voxels in CI, add next one.
                if np.any(ci_mask):
                    ci_mask[np.where(ci_mask)[0][-1] + 1] = True
                # Else, add first one
                else:
                    ci_mask[0] = True
            ci_indices = tuple(vir[ci_mask] for vir in voxel_indices_ravel)

            # ci_indices is a tuple (row_indices, tile_indices) for voxels chosen in CI
            dl_indices = ci_indices[0]
            tile_indices = ci_indices[1]
            n_tiles = len(self.table)

            # If there are no voxels in CI, set mins/maxs to nan
            if dl_indices.size == 0:
                ci_dl_mins = np.full(n_tiles, np.nan, dtype=float)
                ci_dl_maxs = np.full(n_tiles, np.nan, dtype=float)
            else:
                # Prepare accumulators: large/small sentinels
                ci_dl_mins = np.full(n_tiles, np.inf, dtype=float)
                ci_dl_maxs = np.full(n_tiles, -np.inf, dtype=float)

                # Map voxel row index r -> interval [dl_grid[r], dl_grid[r+1]]
                dl_mins_vals = dl_grid[dl_indices]  # dl_grid[r]
                dl_maxs_vals = dl_grid[dl_indices + 1]  # dl_grid[r+1]

                # Reduce per-tile using numpy ufunc.at (C-level loops, memory-efficient)
                np.minimum.at(ci_dl_mins, tile_indices, dl_mins_vals)
                np.maximum.at(ci_dl_maxs, tile_indices, dl_maxs_vals)

                # Tiles with no voxels in CI remain +/-inf; set them to nan to match prior behavior
                mask_empty = ~np.isfinite(ci_dl_mins)
                ci_dl_mins[mask_empty] = np.nan
                ci_dl_maxs[mask_empty] = np.nan

            # Cache in table
            self.volume_ci = cache_ci
            self.table["CIDLMIN"] = ci_dl_mins
            self.table["CIDLMAX"] = ci_dl_maxs

        return ci_vols

    def sample_positions_2d(self, n_samples=1, rng=None):
        """Sample 2D positions in the skymap using tile probabilities.

        :param n_samples: Number of samples to draw, defaults to 1
        :type n_samples: int, optional
        :param rng: numpy random number generator, defaults to None
        :type rng: numpy.random.Generator, optional
        :return: Tuple of (selected tile indices, ras, decs) for the sampled positions
        :rtype: tuple
        """
        # Initialize random number generator if not provided
        if rng is None:
            rng = np.random.default_rng()
        # Sample tiles according to their probability
        selected_tiles = rng.choice(
            len(self.table),
            size=n_samples,
            p=self.table["PROB"] / np.sum(self.table["PROB"]),
        )
        # Get tile indices
        tile_uniqs = self.table["UNIQ"][selected_tiles]
        tile_levels, tile_ipixs = ah.uniq_to_level_ipix(tile_uniqs)
        tile_nsides = ah.level_to_nside(tile_levels)
        # Sample positions within tiles
        dx, dy = rng.uniform(0, 1, size=(2, n_samples))
        ras, decs = ah.healpix_to_lonlat(
            tile_ipixs,
            nside=tile_nsides,
            dx=dx,
            dy=dy,
            order="nested",
        )
        return selected_tiles, ras, decs

    def sample_positions_3d(self, n_samples=1, rng=None):
        """Sample 3D positions in the skymap using tile probabilities and distance distributions.

        :param n_samples: Number of samples to draw, defaults to 1
        :type n_samples: int, optional
        :param rng: numpy random number generator, defaults to None
        :type rng: numpy.random.Generator, optional
        :return: Tuple of (selected tile indices, ras, decs, distances) for the sampled positions
        :rtype: tuple
        """
        # Initialize random number generator if not provided
        if rng is None:
            rng = np.random.default_rng()
        # Sample 2D positions
        selected_tiles, ras, decs = self.sample_positions_2d(n_samples=n_samples, rng=rng)
        # Sample distances
        # Mask to tiles of interest
        mask = np.logical_and(np.isfinite(self.table["DISTMU"]), ~np.isnan(self.table["DISTMU"]))
        # Set grid
        dl_mean = np.average(self.table["DISTMU"][mask], weights=self.table["PROB"][mask])
        dl_grid = np.linspace(0, dl_mean * 6, 1001)
        # Calculate normalized distance pdfs for each tile (n_dl x n_tiles)
        pdf_dl = (
            self.table["DISTNORM"]
            / ((2 * np.pi) ** 0.5 * self.table["DISTSIGMA"])
            * np.exp(-0.5 * ((np.array([dl_grid]).T - self.table["DISTMU"]) / self.table["DISTSIGMA"]) ** 2)
            * dl_grid[:, np.newaxis] ** 2
        )  # n_dl x n_tiles
        pdf_dl = (pdf_dl[:-1, :] + pdf_dl[1:, :]) / 2
        pdf_dl /= np.sum(pdf_dl, axis=0)  # Normalize to get proper pdfs
        # Sample distances for each selected tile
        dls = np.zeros(n_samples)
        for i in range(n_samples):
            tile_idx = selected_tiles[i]
            cdf = np.concatenate(([0], np.cumsum(pdf_dl[:, tile_idx])))
            dls[i] = np.interp(
                rng.uniform(0, 1),
                cdf,
                dl_grid,
            )
        # Return
        return selected_tiles, ras, decs, dls
