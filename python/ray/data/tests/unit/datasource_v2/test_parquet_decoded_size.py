import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ray.data._internal.datasource_v2.formats.parquet.parquet_file_reader import (
    _estimate_batch_size_from_chunk_stats,
)
from ray.data._internal.datasource_v2.formats.parquet.parquet_size_estimator import (
    PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT,
)
from ray.data._internal.datasource_v2.formats.parquet.parquet_utils import (
    DECODED_SIZE_ESTIMATE_ENV,
    _row_group_budget_size,
    _row_group_decoded_size_estimate,
    _row_group_uncompressed_size,
)


@pytest.fixture(autouse=True)
def _default_estimate_env(monkeypatch):
    monkeypatch.delenv(DECODED_SIZE_ESTIMATE_ENV, raising=False)


def _metadata(table: pa.Table, **write_kwargs) -> pq.FileMetaData:
    buf = io.BytesIO()
    pq.write_table(table, buf, **write_kwargs)
    buf.seek(0)
    return pq.ParquetFile(buf).metadata


def _estimate(md: pq.FileMetaData, column_indices=None) -> int:
    return _row_group_decoded_size_estimate(md.row_group(0), md.schema, column_indices)


def test_fixed_width_columns_match_arrow_size() -> None:
    n = 10_000
    table = pa.table(
        {
            "i32": pa.array(range(n), pa.int32()),
            "i64": pa.array(range(n), pa.int64()),
            "f64": pa.array([float(i % 7) for i in range(n)], pa.float64()),
        }
    )
    # High-cardinality dictionary pages encode larger than the values; the
    # estimate must still be the exact Arrow size.
    for kwargs in ({}, {"use_dictionary": False}):
        assert _estimate(_metadata(table, **kwargs)) == table.nbytes


N = 4_000
# Arrow type -> a column of N values. Each case asserts the estimate tracks the
# Arrow size of what pyarrow reads back.
_TYPE_CASES = {
    "int8": pa.array([i % 100 for i in range(N)], pa.int8()),
    "uint16": pa.array(range(N), pa.uint16()),
    "int32": pa.array(range(N), pa.int32()),
    "int64_low_cardinality": pa.array([i % 3 for i in range(N)], pa.int64()),
    "float32": pa.array([i / 3 for i in range(N)], pa.float32()),
    "float64": pa.array([i / 3 for i in range(N)], pa.float64()),
    "float16": pa.array([1.0, 2.0] * (N // 2), pa.float16()),
    "bool": pa.array([i % 3 == 0 for i in range(N)]),
    "date32": pa.array(range(N), pa.date32()),
    "time32": pa.array(range(N), pa.time32("ms")),
    "timestamp": pa.array(range(N), pa.timestamp("us")),
    "decimal_9": pa.array([1] * N, pa.decimal128(9, 2)),
    "decimal_18": pa.array([1] * N, pa.decimal128(18, 2)),
    "decimal_30": pa.array([1] * N, pa.decimal128(30, 2)),
    "decimal_50": pa.array([1] * N, pa.decimal256(50, 2)),
    "fixed_size_binary": pa.array([b"abcd"] * N, pa.binary(4)),
    "nullable_int64": pa.array([None if i % 4 else i for i in range(N)], pa.int64()),
    "plain_string": pa.array([f"value-{i}" for i in range(N)]),
    "list_int64": pa.array([[i % 5] * 8 for i in range(N)], pa.list_(pa.int64())),
    "list_string": pa.array([[f"s{i}", "t"] for i in range(N)]),
    "struct": pa.array([{"x": i, "y": i / 2} for i in range(N)]),
    "map": pa.array(
        [[(f"k{i % 7}", i)] for i in range(N)], pa.map_(pa.string(), pa.int32())
    ),
}


@pytest.mark.parametrize("name", sorted(_TYPE_CASES))
def test_estimate_tracks_arrow_size_across_types(name) -> None:
    table = pa.table({"c": _TYPE_CASES[name]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    md = pq.ParquetFile(io.BytesIO(buf.getvalue())).metadata
    actual = pq.read_table(io.BytesIO(buf.getvalue())).nbytes
    estimate = _estimate(md)
    assert 0.8 * actual <= estimate <= 1.5 * actual, (estimate, actual)


def test_dictionary_encoded_strings_never_below_encoded_size() -> None:
    # Known gap: the footer has no decoded byte count for dictionary-encoded
    # strings (pyarrow does not expose Parquet SizeStatistics), so the estimate
    # under-states them. It must still never fall below the old budget, and it
    # does count the offsets, which already exceed the encoded size here.
    table = pa.table({"s": pa.array(["a fairly long repeated value"] * N)})
    md = _metadata(table)
    assert _estimate(md) >= _row_group_uncompressed_size(md.row_group(0))
    assert _estimate(md) >= (N + 1) * 4


def test_dictionary_encoded_list_column_sizes_decoded_not_encoded() -> None:
    # A low-cardinality list<int64> column (like token ids) dictionary-encodes
    # far below its Arrow size; the estimate must track the decoded size.
    rows, width = 2_000, 256
    table = pa.table(
        {
            "tokens": pa.array(
                [[(r + j) % 50 for j in range(width)] for r in range(rows)],
                pa.list_(pa.int64()),
            )
        }
    )
    md = _metadata(table)
    actual = table.nbytes
    estimate = _estimate(md)
    assert abs(estimate - actual) / actual < 0.01
    # The encoded size this replaces under-states the decoded size.
    assert _row_group_uncompressed_size(md.row_group(0)) < actual / 2


def test_nested_list_adds_offsets_per_level() -> None:
    table = pa.table(
        {"x": pa.array([[[1, 2], [3]], [[4]]] * 500, pa.list_(pa.list_(pa.int32())))}
    )
    md = _metadata(table)
    leaf_values = md.row_group(0).column(0).num_values
    estimate = _estimate(md)
    # Values plus at least the outer level's per-row offsets.
    assert estimate >= leaf_values * 4 + (table.num_rows + 1) * 4
    assert estimate >= table.nbytes * 0.9


def test_projection_sums_only_listed_leaves() -> None:
    n = 1_000
    table = pa.table(
        {"a": pa.array(range(n), pa.int64()), "b": pa.array(range(n), pa.int32())}
    )
    md = _metadata(table)
    assert _estimate(md, [0]) == n * 8
    assert _estimate(md, [1]) == n * 4
    assert _estimate(md, []) == 0


def test_byte_array_falls_back_to_uncompressed_with_offset_floor() -> None:
    table = pa.table({"s": pa.array(["abc", "de", None, "fghij"] * 1_000)})
    md = _metadata(table, use_dictionary=False)
    chunk = md.row_group(0).column(0)
    assert _estimate(md) == max(
        chunk.total_uncompressed_size, (chunk.num_values + 1) * 4
    )
    assert _estimate(md) >= table.nbytes * 0.5


def test_env_flag_restores_encoded_budget(monkeypatch) -> None:
    table = pa.table({"tokens": pa.array([[1, 2, 3] * 50] * 500, pa.list_(pa.int64()))})
    md = _metadata(table)
    rg = md.row_group(0)
    assert _row_group_budget_size(rg, md.schema) == _estimate(md)

    monkeypatch.setenv(DECODED_SIZE_ESTIMATE_ENV, "0")
    assert _row_group_budget_size(rg, md.schema) == _row_group_uncompressed_size(rg)
    # Batches budgeted on the encoded size apply the legacy fixed ratio again.
    assert _estimate_batch_size_from_chunk_stats(1000, 100, 1000) == 100 // (
        PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT
    )


@pytest.mark.parametrize(
    "size, rows, target, expected",
    [
        (1000, 10, 500, 5),  # 100 bytes/row -> 5 rows fit 500 bytes
        (1000, 10, 10_000, 10),  # never more than the chunk's rows
        (0, 10, 500, None),
        (1000, 0, 500, None),
    ],
)
def test_chunk_stats_batch_size_uses_size_as_decoded(
    size, rows, target, expected
) -> None:
    assert _estimate_batch_size_from_chunk_stats(size, rows, target) == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
