"""Station (point-observation) dataset support, built on ``tabular-py``.

Everything here imports without ``tabular_py`` installed; the dependency is only
required once a dataset is actually loaded, which is what lets the rest of the
library import cleanly when the optional extra is absent.
"""

from .errors import translate_station_error
from .stations_client import StationsClient
from .wrap import WrappedStationDataset, wrap_station_dataset

__all__ = [
    "StationsClient",
    "WrappedStationDataset",
    "translate_station_error",
    "wrap_station_dataset",
]
