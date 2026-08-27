"""Entity (point-observation) dataset support, built on ``tabular_py``.

"Entity" is tabular's word for the thing a row belongs to -- a weather station in
GHCND, a buoy in NDBC -- and this namespace follows it rather than keeping an
older ``station`` vocabulary the layer beneath no longer uses.

``tabular_py`` is imported lazily, at load time rather than here. It is an
ordinary dependency, so it should always be present; deferring the import keeps
importing this library cheap, and keeps a broken or partial install from taking
down every other namespace along with entity support.
"""

from .entities_client import EntitiesClient
from .errors import translate_entity_error
from .wrap import WrappedEntityDataset, wrap_entity_dataset

__all__ = [
    "EntitiesClient",
    "WrappedEntityDataset",
    "translate_entity_error",
    "wrap_entity_dataset",
]
