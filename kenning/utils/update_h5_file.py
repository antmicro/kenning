# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Method for updating legacy H5 files.
"""

import shutil
from pathlib import Path
from typing import Optional

from kenning.utils.resource_manager import PathOrURI


def update_h5_file(
    h5_filepath: PathOrURI, output_path: Optional[Path] = None
) -> Path:
    """
    Update an H5 file to be compatible with the newest version of Tensorflow.

    Parameters
    ----------
    h5_filepath : PathOrURI
        Path to the H5 file to be updated.
    output_path: Optional[Path]
        Path to updated H5 file.

    Returns
    -------
    Path
        Path to updated file.
    """
    if h5_filepath.suffix not in (".h5", ".hdf5"):
        return h5_filepath

    if output_path is None:
        output_path = h5_filepath.with_name(
            f"{h5_filepath.stem}_updated{h5_filepath.suffix}"
        )

    shutil.copy2(h5_filepath, output_path)

    import h5py

    with h5py.File(str(output_path), mode="r+") as fd:
        model_configuration = fd.attrs.get("model_config")

        if model_configuration.find('"groups": 1,') != -1:
            model_configuration = model_configuration.replace(
                '"groups": 1,', ""
            )
            fd.attrs.modify("model_config", model_configuration)
            fd.flush()

            model_configuration = fd.attrs.get("model_config")
            assert model_configuration.find('"groups": 1,') == -1

        if model_configuration.find('"loss": "mae"') != -1:
            model_configuration = model_configuration.replace(
                '"loss": "mae"', '"loss": "mean_absolute_error"'
            )
            fd.attrs.modify("training_config", model_configuration)
            fd.flush()

            model_configuration = fd.attrs.get("training_config")
            assert model_configuration.find('"loss": "mae"') == -1
    return output_path
