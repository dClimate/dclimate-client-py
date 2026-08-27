"""Entity dataset support: the client namespace and the error boundary.

Mirrors ``dclimate-client-js`` ``tests/entities.test.ts``,
``tests/entities-wrap.test.ts``, and ``tests/entities-columnkey.test.ts``.

What is under test is the translation boundary, not ``tabular_py``'s reader:
these tests need methods that fail on demand, in each shape the real class can
fail in, so the datasets here are stand-ins rather than real ones opened over a
gateway.
"""

from __future__ import annotations

import pytest

from dclimate_client_py.dclimate_zarr_errors import (
    DatasetCorruptError,
    DatasetNotFoundError,
    InvalidSelectionError,
    NoDataFoundError,
    TabularNotInstalledError,
    ZarrClientError,
)
from dclimate_client_py.dclimate_client import dClimateClient
from dclimate_client_py.entities import EntitiesClient

tabular_py = pytest.importorskip(
    "tabular_py",
    reason="entity support needs the dclimate-tabular-py distribution",
)

from tabular_py.errors import (  # noqa: E402
    CodecError,
    DatasetIntegrityError,
    DatasetReaderError,
    GeoFilterError,
    PredicateError,
    RangeSourceError,
    EntitySelectionError,
    WireError,
)

from dclimate_client_py.entities.errors import translate_entity_error  # noqa: E402
from dclimate_client_py.entities.wrap import wrap_entity_dataset  # noqa: E402


def fake(**overrides: object) -> tabular_py.EntityDataset:
    """A stand-in for a real dataset.

    Built by hand rather than opened over a gateway: what is under test is the
    boundary, so these need methods that fail on demand. It is a real
    ``EntityDataset`` instance so the wrapper's ``isinstance`` re-wrapping check
    sees what it would see in production.
    """
    dataset = object.__new__(tabular_py.EntityDataset)
    for name, value in overrides.items():
        object.__setattr__(dataset, name, value)
    return dataset


class TestClientNamespace:
    def test_available_without_configuration(self) -> None:
        # Siren needs credentials and raises when unconfigured; entity reads
        # only need the transport the client already has, so requiring an option
        # would be ceremony with nothing behind it.
        client = dClimateClient()
        assert isinstance(client.entities, EntitiesClient)

    def test_reuses_one_instance_across_accesses(self) -> None:
        client = dClimateClient()
        assert client.entities is client.entities

    async def test_rejects_a_malformed_cid_before_touching_the_network(self) -> None:
        calls: list[str] = []

        class Recording:
            async def get_block(self, cid: object) -> bytes:
                calls.append("get_block")
                raise AssertionError("should not be reached")

            async def get_range(self, cid: object, offset: int, length: int) -> bytes:
                calls.append("get_range")
                raise AssertionError("should not be reached")

        entities = EntitiesClient(gateway_url="http://127.0.0.1:8080")
        entities._source = lambda _gateway: Recording()  # type: ignore[method-assign]

        with pytest.raises(DatasetNotFoundError):
            await entities.load("obviously-not-a-cid")
        # The point of the assertion: a typo'd CID should fail locally and
        # instantly, not after a gateway round trip.
        assert calls == []

    async def test_requires_a_cid_until_catalog_resolution_exists(self) -> None:
        client = dClimateClient()
        with pytest.raises(DatasetNotFoundError):
            await client.entities.load("")

    async def test_routes_reads_through_the_gateway_it_was_given(self) -> None:
        entities = EntitiesClient(gateway_url="https://gateway.example")
        source = entities._source(None)
        assert isinstance(source, tabular_py.GatewayRangeSource)
        # A 404 would end the load; the assertion is about where reads go.
        assert "gateway.example" in repr(source.__dict__)
        await entities.aclose()

    async def test_a_per_request_gateway_overrides_the_client_default(self) -> None:
        entities = EntitiesClient(gateway_url="https://default.example")
        source = entities._source("https://override.example")
        assert "override.example" in repr(source.__dict__)
        await entities.aclose()

    def test_prefers_the_clients_own_cas_when_open(self) -> None:
        sentinel = object()
        entities = EntitiesClient(gateway_url="https://unused.example", cas=sentinel)
        source = entities._source(None)
        # Reads go through the client's configured transport -- pinning, retries
        # and endpoints included -- rather than a second HTTP path.
        assert isinstance(source, tabular_py.CasRangeSource)

    def test_a_gateway_override_bypasses_the_cas(self) -> None:
        entities = EntitiesClient(gateway_url="https://default.example", cas=object())
        source = entities._source("https://override.example")
        assert isinstance(source, tabular_py.GatewayRangeSource)


class FakeCas:
    """Stands in for a ``KuboCAS`` the client owns and closes on exit."""

    async def __aexit__(self, *exc: object) -> None:
        return None


class TestNamespaceLifecycle:
    """Who owns which transport, and when it stops being usable."""

    async def test_a_closed_context_does_not_leave_a_dead_transport_cached(
        self,
    ) -> None:
        # A KuboCAS never reopens once closed. Keeping the instance built around
        # one would make entity reads after the context fail on a dead
        # transport, instead of falling back to the gateway as documented.
        client = dClimateClient()
        client._kubo_cas = FakeCas()  # type: ignore[assignment]
        during = client.entities
        assert during._cas is client._kubo_cas

        await client.__aexit__(None, None, None)

        assert client._entities_client is None
        after = client.entities
        assert after is not during
        assert after._cas is None

    async def test_rebuilding_onto_the_cas_does_not_strand_gateway_sources(
        self,
    ) -> None:
        # Entity data loaded before __aenter__ opens a gateway source. The
        # rebuilt instance is the only one __aexit__ closes, so the old one's
        # sources have to come with it or their pools leak.
        closed: list[object] = []

        class Source:
            async def aclose(self) -> None:
                closed.append(self)

        client = dClimateClient()
        before = client.entities
        source = Source()
        before._owned_sources.append(source)

        client._kubo_cas = FakeCas()  # type: ignore[assignment]
        after = client.entities
        assert after is not before
        assert before._owned_sources == []
        assert after._owned_sources == [source]

        await client.__aexit__(None, None, None)
        assert closed == [source]

    async def test_repeated_gateway_reads_reuse_one_source(self) -> None:
        # Each gateway source owns an httpx client, so opening one per call
        # would make a long-lived client accumulate a connection pool per query
        # until aclose. Reuse is keyed by URL: a per-request override is a
        # different gateway and gets its own.
        entities = EntitiesClient(gateway_url="https://gw.example")

        first = entities._source(None)
        assert entities._source(None) is first
        assert len(entities._owned_sources) == 1

        override = entities._source("https://other.example")
        assert override is not first
        assert len(entities._owned_sources) == 2

        await entities.aclose()
        # A closed source must not be handed out again.
        assert entities._owned_sources == []
        assert entities._source(None) is not first

    async def test_one_failing_source_does_not_strand_the_others(self) -> None:
        # The list is cleared up front, so a source skipped here is never
        # retried by anything.
        closed: list[str] = []

        class Source:
            def __init__(self, name: str, fails: bool = False) -> None:
                self.name = name
                self.fails = fails

            async def aclose(self) -> None:
                if self.fails:
                    raise RuntimeError(f"{self.name} refused to close")
                closed.append(self.name)

        entities = EntitiesClient(gateway_url="https://unused.example")
        entities._owned_sources.extend(
            [Source("first"), Source("second", fails=True), Source("third")]
        )

        with pytest.raises(RuntimeError, match="second refused to close"):
            await entities.aclose()

        # The failure is reported, but not at the cost of the healthy pools.
        assert closed == ["first", "third"]
        assert entities._owned_sources == []


class TestTranslateEntityError:
    """The mapping, by who has to act."""

    def test_not_found_selection_becomes_no_data_found(self) -> None:
        # A well-formed question with an empty answer stays an empty answer,
        # rather than being reported as the caller's mistake.
        with pytest.raises(NoDataFoundError):
            translate_entity_error(EntitySelectionError("no match", "not-found"))

    def test_invalid_selection_becomes_invalid_selection(self) -> None:
        with pytest.raises(InvalidSelectionError):
            translate_entity_error(EntitySelectionError("bad", "invalid"))

    def test_integrity_failure_becomes_corrupt_not_invalid(self) -> None:
        # tabular deliberately does not descend DatasetIntegrityError from
        # DatasetReaderError precisely so this boundary cannot sweep a corrupt
        # dataset into "you asked wrong".
        with pytest.raises(DatasetCorruptError):
            translate_entity_error(DatasetIntegrityError("bad digest"))

    @pytest.mark.parametrize(
        "cause",
        [
            DatasetReaderError("unknown column"),
            PredicateError("not comparable"),
            # A malformed geometry is the caller's to fix, exactly like a
            # malformed predicate. Covered explicitly because tabular-py
            # descends this one from the base class rather than from
            # `DatasetReaderError` (tabular-js descends it from the reader
            # error, so the two ports disagree) -- without its own branch it
            # reaches the corruption catch-all and blames the publisher for an
            # inverted bounding box.
            GeoFilterError("bbox latitude bounds are inverted"),
        ],
    )
    def test_reader_and_predicate_failures_are_invalid_selections(
        self, cause: Exception
    ) -> None:
        with pytest.raises(InvalidSelectionError):
            translate_entity_error(cause)

    def test_transport_failures_pass_through_untranslated(self) -> None:
        # A gateway timeout is retryable and says nothing about the dataset;
        # reporting it as corruption would send a caller to the publisher over
        # what is usually a network blip.
        original = RangeSourceError("gateway 503")
        with pytest.raises(RangeSourceError) as raised:
            translate_entity_error(original)
        assert raised.value is original

    @pytest.mark.parametrize(
        "cause", [CodecError("not dag-cbor"), WireError("not a root")]
    )
    def test_other_tabular_errors_are_corruption(self, cause: Exception) -> None:
        # Bytes arrived and did not describe a readable dataset. Matched on the
        # base class, so this keeps holding the day tabular adds a leaf type.
        with pytest.raises(DatasetCorruptError) as raised:
            translate_entity_error(cause)
        # The specific class stays visible even though the branch is generic.
        assert type(cause).__name__ in str(raised.value)

    def test_foreign_errors_surface_as_themselves(self) -> None:
        # A TypeError from a bug in this client is not ours to reinterpret.
        original = TypeError("bug")
        with pytest.raises(TypeError) as raised:
            translate_entity_error(original)
        assert raised.value is original

    def test_message_is_preserved_verbatim(self) -> None:
        # The message is the only part naming the entity, column, or distance
        # that actually failed.
        with pytest.raises(InvalidSelectionError, match="column 'TMPX' at row 12"):
            translate_entity_error(DatasetReaderError("column 'TMPX' at row 12"))


class TestWrapEntityDataset:
    """The boundary has to hold for the whole chain, not just ``load``."""

    async def test_translates_a_rejection_from_a_terminal_method(self) -> None:
        # The gap the wrapper exists to close: `rows()` is where a caller most
        # often meets a reader error, and it is nowhere near `load`'s try/except.
        async def rows() -> list[object]:
            raise DatasetReaderError("unknown column")

        dataset = wrap_entity_dataset(fake(rows=rows))
        with pytest.raises(InvalidSelectionError):
            await dataset.rows()

    async def test_keeps_the_not_found_distinction_across_the_boundary(self) -> None:
        async def rows() -> list[object]:
            raise EntitySelectionError("no match", "not-found")

        dataset = wrap_entity_dataset(fake(rows=rows))
        with pytest.raises(NoDataFoundError):
            await dataset.rows()

    def test_translates_a_synchronous_raise_from_a_selector(self) -> None:
        # An unknown column rejected at selection time, before any I/O.
        def elements(*names: str) -> object:
            raise DatasetReaderError("unknown element")

        dataset = wrap_entity_dataset(fake(elements=elements))
        with pytest.raises(InvalidSelectionError):
            dataset.elements("TMPX")

    def test_rewraps_a_sync_chainable_so_translation_survives_the_chain(self) -> None:
        async def rows() -> list[object]:
            raise DatasetReaderError("late failure")

        inner = fake(rows=rows)
        dataset = wrap_entity_dataset(fake(select=lambda *ids: inner))
        selected = dataset.select("USW00023174")
        # Without re-wrapping, translation would stop at the first link and this
        # would raise tabular's error instead of this library's.
        assert type(selected).__name__ == "WrappedEntityDataset"

    async def test_rewraps_an_async_chainable(self) -> None:
        async def rows() -> list[object]:
            raise DatasetReaderError("late failure")

        inner = fake(rows=rows)

        async def nearest(*args: object, **kwargs: object) -> object:
            return inner

        dataset = wrap_entity_dataset(fake(nearest=nearest))
        selected = await dataset.nearest(34.0, -118.0)
        assert type(selected).__name__ == "WrappedEntityDataset"
        with pytest.raises(InvalidSelectionError):
            await selected.rows()

    async def test_passes_plain_data_through_untouched(self) -> None:
        payload = [{"entity_id": "USW00023174", "value": 3}]

        async def rows() -> list[object]:
            return payload

        dataset = wrap_entity_dataset(fake(rows=rows))
        assert await dataset.rows() is payload

    def test_non_callable_attributes_pass_through(self) -> None:
        # `_query` backs the read-only `query_spec` property, so this covers the
        # real shape: a plain attribute forwarded without being called.
        dataset = wrap_entity_dataset(fake(_query="a-query"))
        assert dataset._query == "a-query"

    def test_unwrapped_exposes_the_underlying_dataset(self) -> None:
        inner = fake(_query="a-query")
        assert wrap_entity_dataset(inner).unwrapped is inner

    async def test_every_escaping_error_is_this_librarys(self) -> None:
        # The promise the boundary exists to keep: one `except ZarrClientError`
        # around the client catches entity failures too.
        for cause in (
            DatasetReaderError("bad"),
            EntitySelectionError("gone", "not-found"),
            DatasetIntegrityError("corrupt"),
            WireError("not a root"),
        ):

            async def rows(_cause: Exception = cause) -> list[object]:
                raise _cause

            dataset = wrap_entity_dataset(fake(rows=rows))
            with pytest.raises(ZarrClientError):
                await dataset.rows()


class TestMissingDependency:
    def test_absence_names_the_distribution_not_the_module(self, monkeypatch) -> None:
        import builtins

        from dclimate_client_py.entities import entities_client

        real_import = builtins.__import__

        def missing(name: str, *args: object, **kwargs: object) -> object:
            if name == "tabular_py":
                raise ImportError("No module named 'tabular_py'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing)
        # The whole point of catching this: the import name and the distribution
        # name differ, so the stock ModuleNotFoundError names `tabular_py`, which
        # `pip install` cannot find. The message has to name the installable one.
        with pytest.raises(TabularNotInstalledError, match="dclimate-tabular-py"):
            entities_client._require_tabular()


class TestColumnKey:
    """Naming, end to end through the real reader rather than a stand-in.

    Mirrors ``dclimate-client-js`` ``tests/entities-columnkey.test.ts``, with one
    deliberate difference in what is asserted. A dataset's published column names
    are a property of its profile, not of the stored blocks: every dClimate
    dataset published so far stores lowercase field names (``tmax``) and
    publishes the canonical uppercase element (``TMAX``).

    JS defaults to the identity, so *its* loader had to forward a `columnKey`
    before any documented `TMAX` query worked. ``tabular_py.default_column_key``
    already uppercases, so the documented names work here with no mapping at all
    -- verified against the published GHCND, NDBC, SCAN and SNOTEL roots, all
    four of which store lowercase.

    What `column_key` buys this client is therefore the *other* direction: a
    dataset whose published names are not simply the uppercase of its stored
    ones -- NDBC's mixed-case `SwH`, say -- is unreachable without it, and the
    parameter is what keeps such a dataset loadable at all.
    """

    @staticmethod
    def _client() -> EntitiesClient:
        return EntitiesClient(gateway_url="http://gateway.invalid")

    async def _load(self, **kwargs: object) -> object:
        """Open the in-memory fixture through the real client boundary."""
        from tests.helpers.entity_fixture import build_dataset

        source, root = build_dataset()
        entities = self._client()
        # The transport is the one thing that cannot be in-memory end to end:
        # substituting the source is what keeps this offline while every other
        # layer -- CID parse, open, wrap, translate -- is the real one.
        entities._source = lambda _gateway: source  # type: ignore[method-assign]
        return await entities.load(root, **kwargs)  # type: ignore[arg-type]

    async def test_published_names_resolve_without_a_mapping(self) -> None:
        # The fixture stores `tmax`, exactly as GHCND does. The reader's default
        # uppercases, so the documented name works with no mapping -- this is the
        # assertion that would fail if that default ever became the identity.
        dataset = await self._load()
        assert [(c.name, c.units) for c in dataset.columns()] == [
            ("TMAX", "degC_tenths")
        ]

        records = await dataset.select("USW00094728").to_records("TMAX")
        assert [record["value"] for record in records] == [250, 251, 252]

    async def test_a_mapping_is_forwarded_to_the_reader(self) -> None:
        # The regression this parameter exists for: a dataset whose published
        # names are not the uppercase of its stored ones is only reachable if the
        # client actually forwards the mapping rather than dropping it.
        dataset = await self._load(column_key=lambda field: field.name)
        assert [c.name for c in dataset.columns()] == ["tmax"]

        records = await dataset.select("USW00094728").to_records("tmax")
        assert [record["value"] for record in records] == [250, 251, 252]

    async def test_a_name_outside_the_mapping_is_an_invalid_selection(self) -> None:
        # Under an identity mapping `TMAX` names nothing. It has to come back as
        # this library's error, not tabular's, wherever it is raised -- eagerly at
        # selection or lazily at read.
        dataset = await self._load(column_key=lambda field: field.name)
        with pytest.raises(InvalidSelectionError):
            await dataset.elements("TMAX").rows()

    async def test_units_are_reported_but_never_applied(self) -> None:
        # `degC_tenths` documents that 250 means 25.0 degrees. Scaling it here is
        # how a reader ends up off by a power of ten, so the raw value stands.
        dataset = await self._load()
        assert dataset.columns()[0].units == "degC_tenths"
        records = await dataset.select("USW00094728").to_records("TMAX")
        assert records[0]["value"] == 250

    @pytest.mark.parametrize(("lat", "lon"), [(100.0, 0.0), (40.78, 250.0)])
    async def test_out_of_range_coordinates_are_invalid_selections(
        self, lat: float, lon: float
    ) -> None:
        # The validation lives in tabular; the client deliberately delegates it
        # and translates at the boundary rather than duplicating bounds checks.
        # This pins that contract from the caller's side: an impossible latitude
        # must surface as this library's error, not as a plausible-looking entity
        # for a point that does not exist.
        dataset = await self._load()
        with pytest.raises(InvalidSelectionError):
            await dataset.find_nearest_entity(lat, lon)

    async def test_an_in_range_point_still_resolves(self) -> None:
        dataset = await self._load()
        nearest = await dataset.find_nearest_entity(40.78, -73.97)
        assert nearest.entity_id == "USW00094728"
