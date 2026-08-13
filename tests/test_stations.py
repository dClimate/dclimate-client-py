"""Station dataset support: the client namespace and the error boundary.

Mirrors ``dclimate-client-js`` ``tests/stations.test.ts`` and
``tests/stations-wrap.test.ts``.

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
from dclimate_client_py.stations import StationsClient

tabular_py = pytest.importorskip(
    "tabular_py",
    reason="station support needs the dclimate-tabular-py distribution",
)

from tabular_py.errors import (  # noqa: E402
    CodecError,
    DatasetIntegrityError,
    DatasetReaderError,
    PredicateError,
    RangeSourceError,
    StationSelectionError,
    WireError,
)

from dclimate_client_py.stations.errors import translate_station_error  # noqa: E402
from dclimate_client_py.stations.wrap import wrap_station_dataset  # noqa: E402

# A real, published GHCND station dataset.
REAL_CID = "bafyr4if2wbttslbxpzmro427j4l4nvcrxqo4tufuffqqmqz7afj2pxyu4a"


def fake(**overrides: object) -> tabular_py.StationDataset:
    """A stand-in for a real dataset.

    Built by hand rather than opened over a gateway: what is under test is the
    boundary, so these need methods that fail on demand. It is a real
    ``StationDataset`` instance so the wrapper's ``isinstance`` re-wrapping check
    sees what it would see in production.
    """
    dataset = object.__new__(tabular_py.StationDataset)
    for name, value in overrides.items():
        object.__setattr__(dataset, name, value)
    return dataset


class TestClientNamespace:
    def test_available_without_configuration(self) -> None:
        # Siren needs credentials and raises when unconfigured; station reads
        # only need the transport the client already has, so requiring an option
        # would be ceremony with nothing behind it.
        client = dClimateClient()
        assert isinstance(client.stations, StationsClient)

    def test_reuses_one_instance_across_accesses(self) -> None:
        client = dClimateClient()
        assert client.stations is client.stations

    async def test_rejects_a_malformed_cid_before_touching_the_network(self) -> None:
        calls: list[str] = []

        class Recording:
            async def get_block(self, cid: object) -> bytes:
                calls.append("get_block")
                raise AssertionError("should not be reached")

            async def get_range(self, cid: object, offset: int, length: int) -> bytes:
                calls.append("get_range")
                raise AssertionError("should not be reached")

        stations = StationsClient(gateway_url="http://127.0.0.1:8080")
        stations._source = lambda _gateway: Recording()  # type: ignore[method-assign]

        with pytest.raises(DatasetNotFoundError):
            await stations.load("obviously-not-a-cid")
        # The point of the assertion: a typo'd CID should fail locally and
        # instantly, not after a gateway round trip.
        assert calls == []

    async def test_requires_a_cid_until_catalog_resolution_exists(self) -> None:
        client = dClimateClient()
        with pytest.raises(DatasetNotFoundError):
            await client.stations.load("")

    async def test_routes_reads_through_the_gateway_it_was_given(self) -> None:
        stations = StationsClient(gateway_url="https://gateway.example")
        source = stations._source(None)
        assert isinstance(source, tabular_py.GatewayRangeSource)
        # A 404 would end the load; the assertion is about where reads go.
        assert "gateway.example" in repr(source.__dict__)
        await stations.aclose()

    async def test_a_per_request_gateway_overrides_the_client_default(self) -> None:
        stations = StationsClient(gateway_url="https://default.example")
        source = stations._source("https://override.example")
        assert "override.example" in repr(source.__dict__)
        await stations.aclose()

    def test_prefers_the_clients_own_cas_when_open(self) -> None:
        sentinel = object()
        stations = StationsClient(gateway_url="https://unused.example", cas=sentinel)
        source = stations._source(None)
        # Reads go through the client's configured transport -- pinning, retries
        # and endpoints included -- rather than a second HTTP path.
        assert isinstance(source, tabular_py.CasRangeSource)

    def test_a_gateway_override_bypasses_the_cas(self) -> None:
        stations = StationsClient(gateway_url="https://default.example", cas=object())
        source = stations._source("https://override.example")
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
        # one would make station reads after the context fail on a dead
        # transport, instead of falling back to the gateway as documented.
        client = dClimateClient()
        client._kubo_cas = FakeCas()  # type: ignore[assignment]
        during = client.stations
        assert during._cas is client._kubo_cas

        await client.__aexit__(None, None, None)

        assert client._stations_client is None
        after = client.stations
        assert after is not during
        assert after._cas is None

    async def test_rebuilding_onto_the_cas_does_not_strand_gateway_sources(
        self,
    ) -> None:
        # Station data loaded before __aenter__ opens a gateway source. The
        # rebuilt instance is the only one __aexit__ closes, so the old one's
        # sources have to come with it or their pools leak.
        closed: list[object] = []

        class Source:
            async def aclose(self) -> None:
                closed.append(self)

        client = dClimateClient()
        before = client.stations
        source = Source()
        before._owned_sources.append(source)

        client._kubo_cas = FakeCas()  # type: ignore[assignment]
        after = client.stations
        assert after is not before
        assert before._owned_sources == []
        assert after._owned_sources == [source]

        await client.__aexit__(None, None, None)
        assert closed == [source]

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

        stations = StationsClient(gateway_url="https://unused.example")
        stations._owned_sources.extend(
            [Source("first"), Source("second", fails=True), Source("third")]
        )

        with pytest.raises(RuntimeError, match="second refused to close"):
            await stations.aclose()

        # The failure is reported, but not at the cost of the healthy pools.
        assert closed == ["first", "third"]
        assert stations._owned_sources == []


class TestTranslateStationError:
    """The mapping, by who has to act."""

    def test_not_found_selection_becomes_no_data_found(self) -> None:
        # A well-formed question with an empty answer stays an empty answer,
        # rather than being reported as the caller's mistake.
        with pytest.raises(NoDataFoundError):
            translate_station_error(StationSelectionError("no match", "not-found"))

    def test_invalid_selection_becomes_invalid_selection(self) -> None:
        with pytest.raises(InvalidSelectionError):
            translate_station_error(StationSelectionError("bad", "invalid"))

    def test_integrity_failure_becomes_corrupt_not_invalid(self) -> None:
        # tabular deliberately does not descend DatasetIntegrityError from
        # DatasetReaderError precisely so this boundary cannot sweep a corrupt
        # dataset into "you asked wrong".
        with pytest.raises(DatasetCorruptError):
            translate_station_error(DatasetIntegrityError("bad digest"))

    @pytest.mark.parametrize(
        "cause",
        [DatasetReaderError("unknown column"), PredicateError("not comparable")],
    )
    def test_reader_and_predicate_failures_are_invalid_selections(
        self, cause: Exception
    ) -> None:
        with pytest.raises(InvalidSelectionError):
            translate_station_error(cause)

    def test_transport_failures_pass_through_untranslated(self) -> None:
        # A gateway timeout is retryable and says nothing about the dataset;
        # reporting it as corruption would send a caller to the publisher over
        # what is usually a network blip.
        original = RangeSourceError("gateway 503")
        with pytest.raises(RangeSourceError) as raised:
            translate_station_error(original)
        assert raised.value is original

    @pytest.mark.parametrize(
        "cause", [CodecError("not dag-cbor"), WireError("not a root")]
    )
    def test_other_tabular_errors_are_corruption(self, cause: Exception) -> None:
        # Bytes arrived and did not describe a readable dataset. Matched on the
        # base class, so this keeps holding the day tabular adds a leaf type.
        with pytest.raises(DatasetCorruptError) as raised:
            translate_station_error(cause)
        # The specific class stays visible even though the branch is generic.
        assert type(cause).__name__ in str(raised.value)

    def test_foreign_errors_surface_as_themselves(self) -> None:
        # A TypeError from a bug in this client is not ours to reinterpret.
        original = TypeError("bug")
        with pytest.raises(TypeError) as raised:
            translate_station_error(original)
        assert raised.value is original

    def test_message_is_preserved_verbatim(self) -> None:
        # The message is the only part naming the station, column, or distance
        # that actually failed.
        with pytest.raises(InvalidSelectionError, match="column 'TMPX' at row 12"):
            translate_station_error(DatasetReaderError("column 'TMPX' at row 12"))


class TestWrapStationDataset:
    """The boundary has to hold for the whole chain, not just ``load``."""

    async def test_translates_a_rejection_from_a_terminal_method(self) -> None:
        # The gap the wrapper exists to close: `rows()` is where a caller most
        # often meets a reader error, and it is nowhere near `load`'s try/except.
        async def rows() -> list[object]:
            raise DatasetReaderError("unknown column")

        dataset = wrap_station_dataset(fake(rows=rows))
        with pytest.raises(InvalidSelectionError):
            await dataset.rows()

    async def test_keeps_the_not_found_distinction_across_the_boundary(self) -> None:
        async def rows() -> list[object]:
            raise StationSelectionError("no match", "not-found")

        dataset = wrap_station_dataset(fake(rows=rows))
        with pytest.raises(NoDataFoundError):
            await dataset.rows()

    def test_translates_a_synchronous_raise_from_a_selector(self) -> None:
        # An unknown column rejected at selection time, before any I/O.
        def elements(*names: str) -> object:
            raise DatasetReaderError("unknown element")

        dataset = wrap_station_dataset(fake(elements=elements))
        with pytest.raises(InvalidSelectionError):
            dataset.elements("TMPX")

    def test_rewraps_a_sync_chainable_so_translation_survives_the_chain(self) -> None:
        async def rows() -> list[object]:
            raise DatasetReaderError("late failure")

        inner = fake(rows=rows)
        dataset = wrap_station_dataset(fake(select=lambda *ids: inner))
        selected = dataset.select("USW00023174")
        # Without re-wrapping, translation would stop at the first link and this
        # would raise tabular's error instead of this library's.
        assert type(selected).__name__ == "WrappedStationDataset"

    async def test_rewraps_an_async_chainable(self) -> None:
        async def rows() -> list[object]:
            raise DatasetReaderError("late failure")

        inner = fake(rows=rows)

        async def nearest(*args: object, **kwargs: object) -> object:
            return inner

        dataset = wrap_station_dataset(fake(nearest=nearest))
        selected = await dataset.nearest(34.0, -118.0)
        assert type(selected).__name__ == "WrappedStationDataset"
        with pytest.raises(InvalidSelectionError):
            await selected.rows()

    async def test_passes_plain_data_through_untouched(self) -> None:
        payload = [{"station_id": "USW00023174", "value": 3}]

        async def rows() -> list[object]:
            return payload

        dataset = wrap_station_dataset(fake(rows=rows))
        assert await dataset.rows() is payload

    def test_non_callable_attributes_pass_through(self) -> None:
        # `_query` backs the read-only `query_spec` property, so this covers the
        # real shape: a plain attribute forwarded without being called.
        dataset = wrap_station_dataset(fake(_query="a-query"))
        assert dataset._query == "a-query"

    def test_unwrapped_exposes_the_underlying_dataset(self) -> None:
        inner = fake(_query="a-query")
        assert wrap_station_dataset(inner).unwrapped is inner

    async def test_every_escaping_error_is_this_librarys(self) -> None:
        # The promise the boundary exists to keep: one `except ZarrClientError`
        # around the client catches station failures too.
        for cause in (
            DatasetReaderError("bad"),
            StationSelectionError("gone", "not-found"),
            DatasetIntegrityError("corrupt"),
            WireError("not a root"),
        ):

            async def rows(_cause: Exception = cause) -> list[object]:
                raise _cause

            dataset = wrap_station_dataset(fake(rows=rows))
            with pytest.raises(ZarrClientError):
                await dataset.rows()


class TestMissingDependency:
    def test_absence_names_the_distribution_not_the_module(self, monkeypatch) -> None:
        import builtins

        from dclimate_client_py.stations import stations_client

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
            stations_client._require_tabular()
