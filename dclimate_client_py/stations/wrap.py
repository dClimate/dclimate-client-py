"""Keeping the error boundary closed across a whole selection chain.

Mirrors ``dclimate-client-js`` ``src/stations/wrap.ts``, which uses a ``Proxy``
for the same reason this uses ``__getattr__`` delegation.

Translating only at ``load`` leaves a hole the size of the actual API. ``open``
parses a manifest; the errors a caller is far likelier to hit come later, from
``select`` on an unknown station, ``where`` on a non-comparable column, or
``rows`` on a selection that matched nothing. Those are raised inside
``tabular_py`` and, untranslated, escape as ``DatasetReaderError`` -- so the one
``except ZarrClientError`` this library tells people to write misses precisely
the failures they were most likely to be catching for.

Delegation rather than a hand-written class of forwarding methods, for two
reasons. Selections return *new* ``StationDataset`` instances, so a class would
have to remember to re-wrap the result of every chainable -- and a method added
to ``tabular_py`` later would silently return an unwrapped dataset, reopening the
hole halfway down a chain. And ``plan``/``rows``/``to_arrow`` return plain data
that must pass through untouched; delegation distinguishes those by what comes
back rather than by a list someone has to keep current.

Three return shapes are handled, because the API has all three:

* sync chainable  (``select``, ``time_range``, ``where``) -> re-wrap
* async chainable (``nearest``)                           -> re-wrap on await
* async terminal  (``rows``, ``plan``, ``list_stations``) -> translate the raise
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

from .errors import translate_station_error

if TYPE_CHECKING:  # pragma: no cover
    from tabular_py import StationDataset


class WrappedStationDataset:
    """A :class:`~tabular_py.StationDataset` whose every failure is this library's.

    Attribute access is forwarded, so this stays correct as ``tabular_py`` grows
    methods. Anything callable is wrapped; anything else (a property such as
    ``reader`` or ``query_spec``) passes straight through.
    """

    __slots__ = ("_dataset",)

    def __init__(self, dataset: StationDataset) -> None:
        object.__setattr__(self, "_dataset", dataset)

    @property
    def unwrapped(self) -> StationDataset:
        """The underlying dataset, for callers who want tabular's own errors."""
        return self._dataset  # type: ignore[attr-defined,no-any-return]

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not found on this class, which is every method
        # of the wrapped dataset.
        attribute = getattr(object.__getattribute__(self, "_dataset"), name)
        if not callable(attribute):
            return attribute

        @functools.wraps(attribute)
        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                result = attribute(*args, **kwargs)
            except BaseException as cause:
                # A synchronous raise -- an unknown column rejected at selection
                # time, before any I/O.
                translate_station_error(cause)
            if inspect.isawaitable(result):
                return _awaited(result)
            return _rewrap(result)

        return call

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._dataset!r})"  # type: ignore[attr-defined]


async def _awaited(awaitable: Any) -> Any:
    try:
        resolved = await awaitable
    except BaseException as cause:
        translate_station_error(cause)
    # `nearest` resolves to a further dataset; keep the chain wrapped.
    return _rewrap(resolved)


def _rewrap(value: Any) -> Any:
    """Re-wrap a returned dataset, so translation does not stop at the first link."""
    from tabular_py import StationDataset

    if isinstance(value, StationDataset):
        return WrappedStationDataset(value)
    return value


def wrap_station_dataset(dataset: StationDataset) -> WrappedStationDataset:
    """Wrap a dataset so every method failure comes back as a ``ZarrClientError``."""
    return WrappedStationDataset(dataset)
