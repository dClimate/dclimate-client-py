"""A tiny real `dclimate-tabular/1` dataset, built in memory.

Mirrors ``dclimate-client-js`` ``tests/entities-columnkey.test.ts``, which builds
its fixture with tabular's ``DatasetWriter``. ``tabular_py`` is deliberately
reader-only and ships no writer, so this composes the same public encoders the
reader exposes -- ``*_to_wire``, ``make_block``, ``MemoryRangeSource`` -- into
the smallest dataset that can answer a query.

Deliberately not a writer: one entity, one fragment, no bucketing, compaction,
rollup, or commit sequencing. What it exists to support is the ``column_key``
contract, which needs a dataset whose *stored* field name differs from its
*published* column name -- GHCND's ``tmax``/``TMAX`` -- read end to end through
the real reader rather than a stand-in.

Every figure in the manifest is derived by reading the Parquet bytes back, so
the manifest cannot describe a payload it does not have.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tabular_py import (
    MemoryRangeSource,
    blake3_digest,
    entity_entry_to_wire,
    make_block,
    root_to_wire,
    table_schema_to_wire,
)

# The structural types live in `tabular_py.model`; only part of the model is
# re-exported from the barrel, and reaching for the module directly keeps this
# working whichever names the barrel happens to carry.
from tabular_py.model import (
    FLAG_SORTED,
    SPEC_VERSION,
    ColumnChunk,
    ColumnStat,
    DatasetRoot,
    DatasetStats,
    EntityEntry,
    EntityIndexInline,
    EntitySummary,
    FragmentBucket,
    FragmentEntry,
    GeoPoint,
    Latest,
    Partitioning,
    RowGroup,
    TableField,
    TableSchema,
    WriteConfig,
)

DAY_US = 24 * 60 * 60 * 1_000_000
# 2024-01-01T00:00:00Z, in microseconds.
EPOCH_2024 = 1_704_067_200 * 1_000_000

ENTITY_ID = "USW00094728"
ROW_COUNT = 3
LAT_MICRO = 40_779_000
LON_MICRO = -73_969_000

# Stored lowercase, exactly as GHCND stores it -- so a caller asking for the
# published `TMAX` needs the mapping, which is the whole point of the fixture.
SCHEMA = TableSchema(
    schema_id=0,
    fields=(
        TableField(field_id=1, name="entity_id", type="string", nullable=False),
        TableField(field_id=2, name="ts", type="int64", nullable=False),
        TableField(
            field_id=10, name="tmax", type="int32", nullable=True, units="degC_tenths"
        ),
    ),
)

_TIMESTAMPS = [EPOCH_2024 + day * DAY_US for day in range(ROW_COUNT)]
_TMAX = [250 + day for day in range(ROW_COUNT)]


def _parquet_bytes() -> bytes:
    table = pa.table(
        {
            "entity_id": [ENTITY_ID] * ROW_COUNT,
            "ts": _TIMESTAMPS,
            "tmax": _TMAX,
        },
        schema=pa.schema(
            [
                pa.field("entity_id", pa.string(), nullable=False),
                pa.field("ts", pa.int64(), nullable=False),
                pa.field("tmax", pa.int32(), nullable=True),
            ]
        ),
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd", version="2.6")
    return buffer.getvalue()


def _fragment_entry(data: bytes, cid: Any) -> FragmentEntry:
    """Describe the fragment by reading it back, never by asserting its shape."""
    metadata = pq.ParquetFile(io.BytesIO(data)).metadata
    # A Parquet file ends with the footer, its 4-byte length, and `PAR1`.
    footer_length = int.from_bytes(data[-8:-4], "little")
    footer_offset = len(data) - 8 - footer_length
    footer = data[footer_offset : footer_offset + footer_length]

    field_ids = [field.field_id for field in SCHEMA.fields]
    row_groups: list[RowGroup] = []
    stats: dict[int, tuple[int, Any, Any]] = {}
    first_row = 0

    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        chunks: list[ColumnChunk] = []
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            field_id = field_ids[column_index]
            start = column.dictionary_page_offset or column.data_page_offset
            chunks.append(
                ColumnChunk(
                    field_id=field_id,
                    offset=int(start),
                    length=int(column.total_compressed_size),
                )
            )
            null_count, minimum, maximum = stats.get(field_id, (0, None, None))
            column_stats = column.statistics
            if column_stats is not None:
                null_count += int(column_stats.null_count)
                low, high = column_stats.min, column_stats.max
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
            stats[field_id] = (null_count, minimum, maximum)

        row_groups.append(
            RowGroup(
                first_row=first_row,
                row_count=group.num_rows,
                ts_min=min(_TIMESTAMPS),
                ts_max=max(_TIMESTAMPS),
                chunks=tuple(chunks),
            )
        )
        first_row += group.num_rows

    return FragmentEntry(
        cid=cid,
        byte_length=len(data),
        schema_id=SCHEMA.schema_id,
        tier=0,
        flags=FLAG_SORTED,
        row_count=ROW_COUNT,
        ts_min=min(_TIMESTAMPS),
        ts_max=max(_TIMESTAMPS),
        footer_offset=footer_offset,
        footer_length=footer_length,
        footer_digest=blake3_digest(footer),
        col_stats=tuple(
            ColumnStat(field_id=field_id, null_count=null, min=low, max=high)
            for field_id, (null, low, high) in sorted(stats.items())
        ),
        row_groups=tuple(row_groups),
        commit_seq=0,
    )


def build_dataset() -> tuple[MemoryRangeSource, str]:
    """A one-entity dataset, and the root CID to open it by."""
    blocks: dict[str, bytes] = {}

    def put(value: Any) -> Any:
        cid, encoded = make_block(value)
        blocks[str(cid)] = encoded
        return cid

    data = _parquet_bytes()
    fragment_cid, _ = make_block(data)
    # Parquet bytes are stored raw, not dag-cbor wrapped: the reader fetches
    # byte ranges out of them directly.
    blocks[str(fragment_cid)] = data

    schema_cid = put(table_schema_to_wire(SCHEMA))
    fragment = _fragment_entry(data, fragment_cid)

    entry = EntityEntry(
        entity_id=ENTITY_ID,
        geo=GeoPoint(lat_micro=LAT_MICRO, lon_micro=LON_MICRO),
        elev_cm=0,
        ts_min=min(_TIMESTAMPS),
        ts_max=max(_TIMESTAMPS),
        row_count=ROW_COUNT,
        columns=("tmax",),
        latest=Latest(ts=max(_TIMESTAMPS), fragment_idx=0),
        fragments=(fragment,),
    )
    entry_cid = put(entity_entry_to_wire(entry))

    summary = EntitySummary(
        entity_id=ENTITY_ID,
        ts_min=min(_TIMESTAMPS),
        ts_max=max(_TIMESTAMPS),
        lat_micro=LAT_MICRO,
        lon_micro=LON_MICRO,
        entry=entry_cid,
        gaps=(),
    )

    root = DatasetRoot(
        spec=SPEC_VERSION,
        dataset_id="columnkey-e2e",
        sequence=0,
        created_us=EPOCH_2024,
        message="fixture",
        watermark_us=max(_TIMESTAMPS),
        parent=None,
        ancestry=(),
        schemas=(schema_cid,),
        current_schema=SCHEMA.schema_id,
        partitioning=Partitioning(key=("entity_id",), sort=("ts",)),
        write_config=WriteConfig(
            fragment_bucket=FragmentBucket(unit="year", count=1), delta_ts=False
        ),
        entities=EntityIndexInline(entries=(summary,)),
        projections={},
        compat=(),
        stats=DatasetStats(
            entities=1,
            rows=ROW_COUNT,
            ts_min=min(_TIMESTAMPS),
            ts_max=max(_TIMESTAMPS),
        ),
        prev_hash_compat=None,
    )
    root_cid = put(root_to_wire(root))
    return MemoryRangeSource(blocks), str(root_cid)
