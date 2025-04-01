import os
from typing import Union

import numpy as np
from numpy.typing import NDArray

from clfd.spike_finding import SpikeSubtractionPlan


class ArchiveHandler:
    """
    Simple wrapper for a psrchive.Archive object, which allows editing it.
    """

    def __init__(self, path: Union[str, os.PathLike]):
        import psrchive

        if hasattr(psrchive, "Archive_load"):
            loader = psrchive.Archive_load
        else:
            loader = psrchive.Archive.load

        self._archive = loader(str(path))

    def data_cube(self) -> NDArray:
        """
        Return the archive data as a 3-dimensional numpy array of shape
        (num_subints, num_chans, num_bins). Scrunch polarizations into
        total intensity. 
        """

        # Get the data cube
        data_cube = self._archive.get_data()

        # Scunch polarizations into total intensity
        if data_cube.shape[1] > 1: # if there are multiple polarizations
            data_cube = data_cube[:, (0, 1)].mean(axis=1) # average over polarizations 
        else: 
            data_cube = data_cube[:, 0, :, :] # only one polarization

        return data_cube

    def apply_profile_mask(self, mask: NDArray):
        """
        Apply profile mask to underlying archive, setting the weights of masked
        profiles to zero.
        """

        # Get the number of polarizations
        n_pols = self._archive.get_npol() 

        # Iterate through each polarization
        for ipol in range(n_pols):
            # Mask the profiles
            for isub, ichan in zip(*np.where(mask)):
                # NOTE: cast indices from numpy.int64 to int, otherwise
                # get_Profile() complains about argument type
                prof = self._archive.get_Profile(int(isub), ipol, int(ichan))
                prof.set_weight(0.0)

    def apply_spike_subtraction_plan(self, plan: SpikeSubtractionPlan):
        """
        Set the values of data inside bad time-phase bins to appropriate
        replacement values.
        """

        # Get the number of polarizations
        n_pols = self._archive.get_npol() 

        # Iterate through each polarization
        for ipol in range(n_pols):
            repvals = plan.replacement_values
            mapping = plan.subint_to_bad_phase_bins_mapping()

            for isub, bad_bins in mapping.items():
                for ichan in plan.valid_channels:
                    # NOTE: cast indices from numpy.int64 to int, otherwise
                    # get_Profile() complains about argument type
                    prof = self._archive.get_Profile(isub, ipol, int(ichan))
                    amps = prof.get_amps()
                    amps[bad_bins] = repvals[isub, ichan, bad_bins]

    def save(self, path: Union[str, os.PathLike]):
        """
        Save archive to given path.
        """
        self._archive.unload(str(path))
