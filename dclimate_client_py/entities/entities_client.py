"""Entity (point-observation) datasets, as opposed to the gridded Zarr datasets.

Mirrors ``dclimate-client-js`` ``src/entities/entities-client.ts``.

"Entity" is tabular's word for the thing a row belongs to -- a weather station in
GHCND, a buoy in NDBC -- and this namespace follows it rather than keeping an
older ``station`` vocabulary the layer beneath no longer uses.

The two are different enough underneath -- irregular entities with per-entity
time coverage, versus a regular lat/lon/time grid -- that sharing one loader
would help nobody. What they do share is how a caller wants to *ask*: degrees,
ISO timestamps, chained selections. ``EntityDataset`` provides that surface, so
this namespace stays thin: resolve a root, hand back the dataset.

Resolution is by CID only for now. There is no STAC equivalent for entity data
yet; when there is, ``load`` grows a ``collection``/``dataset`` form alongside
the CID and the rest of this file is unaffected.
"""

from __future__ import annotations

import typing

from ..dclimate_zarr_errors import (
    DatasetNotFoundError,
    TabularNotInstalledError,
)
from .errors import translate_entity_error
from .wrap import WrappedEntityDataset, wrap_entity_dataset

if typing.TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from py_hamt import KuboCAS
    from tabular_py import NearestEntity, TableField
    from tabular_py.entity_dataset import TimeInput

    ColumnKey = Callable[[TableField], str]


def _require_tabular() -> typing.Any:
    """Import ``tabular_py``, or say plainly that it is missing.

    An ordinary dependency, so this should never fire in a correctly installed
    environment. It exists because the import name and the distribution name
    differ -- ``tabular_py`` from ``dclimate-tabular-py`` -- which makes the
    stock ``ModuleNotFoundError`` name a package that ``pip install`` cannot
    find.
    """
    try:
        import tabular_py
    except ImportError as cause:
        raise TabularNotInstalledError(
            "Entity data requires the `tabular_py` module, which ships in the "
            "`dclimate-tabular-py` distribution. Reinstall this package, or "
            "install it directly with:\n"
            "  uv pip install dclimate-tabular-py"
        ) from cause
    return tabular_py


class EntitiesClient:
    """Loads entity datasets by CID.

    Reads go through whichever transport the parent client already holds -- its
    ``KuboCAS`` when there is one, so pinning, retries, and configured endpoints
    apply here too, and a plain HTTP gateway otherwise.
    """

    def __init__(
        self,
        *,
        gateway_url: str,
        cas: KuboCAS | None = None,
    ) -> None:
        self._gateway_url = gateway_url
        self._cas = cas
        self._owned_sources: list[typing.Any] = []

    def _adopt_sources_from(self, other: EntitiesClient) -> None:
        """Take over another instance's open sources, so they still get closed.

        Used when the parent client rebuilds this namespace onto a different
        transport. Building the replacement happens in a property, which cannot
        await, so the sources the old instance opened move here rather than
        being closed on the spot -- otherwise they would leak, since the
        replacement is the only instance the parent client still closes.
        """
        self._owned_sources.extend(other._owned_sources)
        other._owned_sources = []

    def _source(self, gateway_url: str | None) -> typing.Any:
        tabular = _require_tabular()
        # A per-request gateway override means a dedicated source; otherwise
        # prefer the client's own CAS, which is already configured and pooled.
        if gateway_url is None and self._cas is not None:
            return tabular.CasRangeSource(self._cas)
        source = tabular.GatewayRangeSource(gateway_url or self._gateway_url)
        # Gateway sources own an httpx client, so they have to be closed; CAS
        # sources borrow the client's and must not be.
        self._owned_sources.append(source)
        return source

    async def load(
        self,
        cid: str,
        *,
        gateway_url: str | None = None,
        column_key: ColumnKey | None = None,
    ) -> WrappedEntityDataset:
        """Open an entity dataset by root CID.

        Reads go through the IPFS HTTP gateway, so this needs no local daemon.

        ``column_key`` maps each schema field to the column name queries and
        results use. A dataset's published column names are a property of its
        profile, not of the stored blocks: GHCND stores a field named ``tmax``
        and publishes it as ``TMAX``, NDBC preserves mixed case like ``SwH``.
        The reader's default is the identity -- the schema's own field names --
        so without the dataset's mapping the published names are unreachable:
        ``elements("TMAX")`` is an unknown element even though every GHCND doc
        names it that way. For GHCND, pass ``lambda field: field.name.upper()``.
        """
        if not cid:
            raise DatasetNotFoundError(
                "An entity dataset CID is required. Catalog resolution is not "
                "available yet."
            )

        from multiformats import CID

        try:
            root = CID.decode(cid)
        except Exception as cause:
            raise DatasetNotFoundError(f"Not a valid CID: {cid}: {cause}") from cause

        tabular = _require_tabular()
        source = self._source(gateway_url)
        # Forwarded only when given, so tabular's own `default_column_key` stays
        # the default rather than being shadowed by a `None` this layer invented.
        mapping = {} if column_key is None else {"column_key": column_key}
        # Opening reads and parses the dataset's manifest, so a well-formed CID
        # that points at something else fails here rather than at query time.
        # Translated like any other reader failure: a caller catching
        # `ZarrClientError` should not have to also know tabular-py's hierarchy.
        #
        # The dataset is wrapped on the way out so that guarantee holds for the
        # whole chain, not just this call: `select`, `where`, and `rows` raise
        # tabular's errors too, and a caller has no reason to expect the boundary
        # to stop at `open`.
        try:
            return wrap_entity_dataset(
                await tabular.EntityDataset.open(source, root, **mapping)
            )
        except BaseException as cause:
            translate_entity_error(cause)

    async def nearest(
        self,
        cid: str,
        latitude: float,
        longitude: float,
        *,
        columns: Sequence[str] | None = None,
        max_km: float | None = None,
        within: tuple[TimeInput, TimeInput] | None = None,
        gateway_url: str | None = None,
        column_key: ColumnKey | None = None,
    ) -> NearestEntity:
        """The entity nearest a point that actually has the data you asked for.

        Resolves the dataset and the entity in one call, because "which station
        should I use for this location" is the question most callers open an
        entity dataset to ask, and answering it through :meth:`load` means
        knowing that ``nearest`` needs ``columns`` to avoid picking a station
        full of nulls.

        ``columns`` restricts the search to entities that actually report every
        one of those columns. Without it, "nearest" means nearest *entity*, not
        nearest *data* -- a station 40 km away that has never recorded TMAX wins
        over one 3,000 km away that has. Presence means the entity reported the
        column at some point, not that it reported it recently; pass ``within``
        to narrow that to a time range.

        ``columns`` names columns as this dataset publishes them, so a dataset
        needing a ``column_key`` needs it here too -- otherwise every name in
        ``columns`` is unknown and the search matches nothing.

        Returns the distance alongside the entity: a dataset with no coverage
        near the queried point still has a nearest entity, and the only way to
        tell that apart from a good match is how far away it is.
        """
        dataset = await self.load(cid, gateway_url=gateway_url, column_key=column_key)
        # Already wrapped, so failures here arrive as this library's errors.
        return typing.cast(
            "NearestEntity",
            await dataset.find_nearest_entity(
                latitude,
                longitude,
                max_km=max_km,
                require_columns=columns,
                within_range=within,
            ),
        )

    async def aclose(self) -> None:
        """Close gateway sources this client opened.

        A borrowed ``KuboCAS`` is the parent client's to close, so it is left
        alone; only sources created here are owned here.

        Every source is closed even if one of them fails, because they hold
        separate connection pools: stopping at the first failure would strand
        the rest, and the list is cleared up front, so nothing would ever retry
        them. The first failure is re-raised once the rest are shut.
        """
        sources, self._owned_sources = self._owned_sources, []
        failure: BaseException | None = None
        for source in sources:
            try:
                await source.aclose()
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
