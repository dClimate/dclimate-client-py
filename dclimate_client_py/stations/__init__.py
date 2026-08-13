"""Station (point-observation) dataset support, built on ``tabular_py``.

``tabular_py`` is imported lazily, at load time rather than here. It is an
ordinary dependency, so it should always be present; deferring the import keeps
importing this library cheap, and keeps a broken or partial install from taking
down every other namespace along with station support.
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
