"""Translating ``tabular-py`` failures into this library's error types.

Mirrors ``dclimate-client-js`` ``src/stations/errors.ts``.

Station queries are answered by ``tabular_py``, whose errors descend from its own
base class rather than :class:`ZarrClientError`. A caller writing one
``except ZarrClientError`` around the client would therefore miss every station
failure -- so the boundary translates once, here, rather than each method
wrapping its own body.
"""

from __future__ import annotations

from typing import NoReturn

from ..dclimate_zarr_errors import (
    DatasetCorruptError,
    InvalidSelectionError,
    NoDataFoundError,
)


def translate_station_error(cause: BaseException) -> NoReturn:
    """Re-raise a station query failure as this library's own error type.

    The mapping follows the distinction the rest of this library draws, by who
    has to act: a malformed request is an :class:`InvalidSelectionError` (the
    caller's to fix), a well-formed request that matched nothing is a
    :class:`NoDataFoundError` (an empty answer, nobody's fault), and bytes that
    do not describe a readable dataset are a :class:`DatasetCorruptError` (the
    publisher's). ``tabular_py`` marks the first two on the error itself, so this
    reads a field rather than matching on message text, which would break the
    first time a message improved.

    Transport failures are deliberately left untranslated, because they are none
    of those three: a gateway timeout is retryable and says nothing about the
    dataset.

    The original message is preserved verbatim: it is the only part that names
    the station, column, or distance that actually failed.
    """
    # Imported here, not at module scope, so that importing this module -- which
    # the package's error exports do -- does not pull in tabular_py and, through
    # it, pyarrow, on every `import dclimate_client_py`.
    from tabular_py.errors import (
        DatasetIntegrityError,
        DatasetReaderError,
        DclimateTabularError,
        PredicateError,
        RangeSourceError,
        StationSelectionError,
    )

    if isinstance(cause, StationSelectionError):
        if cause.reason == "not-found":
            raise NoDataFoundError(str(cause)) from cause
        raise InvalidSelectionError(str(cause)) from cause

    # Corrupt data is not a bad question. Checked as its own branch rather than
    # before the reader case for emphasis: tabular deliberately does not descend
    # `DatasetIntegrityError` from `DatasetReaderError`, precisely so a boundary
    # like this one cannot sweep it into "you asked wrong".
    if isinstance(cause, DatasetIntegrityError):
        raise DatasetCorruptError(str(cause)) from cause

    # Other reader failures are malformed requests too -- an unknown column, a
    # predicate against a column that is not comparable.
    if isinstance(cause, (DatasetReaderError, PredicateError)):
        raise InvalidSelectionError(str(cause)) from cause

    # The gateway failing to hand over bytes is a transport fact, not a statement
    # about the dataset: it is retryable, and reporting it as corruption would
    # send a caller to the publisher over what is usually a network blip. Checked
    # before the base class below, which it descends from.
    if isinstance(cause, RangeSourceError):
        raise cause

    # Any remaining tabular error means bytes arrived and did not describe a
    # readable dataset -- a CID naming a UnixFS file rather than a root (a
    # `CodecError` from the dag-cbor decode), or a well-formed block whose fields
    # are not a dataset root (a `WireError`). Both are corruption in the same
    # sense as `DatasetIntegrityError`, and no rephrasing fixes either.
    #
    # Matched on the base class rather than by listing `CodecError` and
    # `WireError`: this is the branch that keeps the promise that everything
    # escaping the client is a `ZarrClientError`, and a list of leaf types
    # silently stops keeping it the day tabular adds one. The specific class
    # stays visible in the message, which is preserved verbatim.
    if isinstance(cause, DclimateTabularError):
        raise DatasetCorruptError(f"{type(cause).__name__}: {cause}") from cause

    # Anything else is not tabular's and not ours to reinterpret: a TypeError
    # from a bug in this client, or an HTTP failure that never reached the
    # reader, should surface as itself.
    raise cause
