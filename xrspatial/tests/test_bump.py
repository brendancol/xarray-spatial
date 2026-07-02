try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial import bump
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)
from xrspatial.utils import has_cuda_and_cupy


def test_bump():
    bumps = bump(20, 20)
    assert bumps is not None


def test_bump_agg_numpy():
    agg = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    np.random.seed(42)
    result = bump(agg=agg)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (20, 30)
    assert isinstance(result.data, np.ndarray)

    # determinism: same seed → same output
    np.random.seed(42)
    result2 = bump(agg=agg)
    np.testing.assert_array_equal(result.values, result2.values)


@cuda_and_cupy_available
def test_bump_agg_cupy():
    import cupy

    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_cp = agg_np.copy()
    agg_cp.data = cupy.asarray(agg_cp.data)

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_cp = bump(agg=agg_cp)

    assert isinstance(result_cp.data, cupy.ndarray)
    np.testing.assert_array_equal(result_np.values, result_cp.data.get())


@dask_array_available
def test_bump_agg_dask():
    """Single chunk (no internal boundaries) → bitwise match with numpy."""
    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(20, 30))

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_dask = bump(agg=agg_dask)

    assert isinstance(result_dask.data, da.Array)
    np.testing.assert_array_equal(result_np.values, result_dask.values)


@dask_array_available
def test_bump_agg_dask_chunked():
    """Multiple chunks: verify laziness, shape, and approximate values."""
    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(10, 15))

    np.random.seed(42)
    result_np = bump(agg=agg_np, spread=1)

    np.random.seed(42)
    result_dask = bump(agg=agg_dask, spread=1)

    assert isinstance(result_dask.data, da.Array)
    assert result_dask.shape == (20, 30)
    computed = result_dask.values
    # With spread=1, edge effects are minimal; total energy should be close
    np.testing.assert_allclose(computed.sum(), result_np.values.sum(), rtol=0.1)


@dask_array_available
@cuda_and_cupy_available
def test_bump_agg_dask_cupy():
    """Single chunk (no internal boundaries) → bitwise match with numpy."""
    import cupy

    agg_np = xr.DataArray(np.zeros((20, 30)), dims=['y', 'x'])
    agg_dc = agg_np.copy()
    agg_dc.data = da.from_array(
        cupy.asarray(agg_dc.data), chunks=(20, 30)
    )

    np.random.seed(42)
    result_np = bump(agg=agg_np)

    np.random.seed(42)
    result_dc = bump(agg=agg_dc)

    assert isinstance(result_dc.data, da.Array)
    computed = result_dc.data.compute()
    np.testing.assert_array_equal(result_np.values, computed.get())


def test_bump_preserves_coords():
    ys = np.linspace(0, 1, 10)
    xs = np.linspace(0, 2, 20)
    agg = xr.DataArray(
        np.zeros((10, 20)),
        dims=['lat', 'lon'],
        coords={'lat': ys, 'lon': xs},
    )
    result = bump(agg=agg)
    assert list(result.dims) == ['lat', 'lon']
    np.testing.assert_array_equal(result.coords['lat'].values, ys)
    np.testing.assert_array_equal(result.coords['lon'].values, xs)


def test_bump_agg_infers_shape():
    """When agg is given, width/height are inferred — no need to pass them."""
    agg = xr.DataArray(np.zeros((15, 25)), dims=['y', 'x'])
    np.random.seed(42)
    result = bump(agg=agg)
    assert result.shape == (15, 25)

    # Equivalent to explicit width/height
    np.random.seed(42)
    result2 = bump(width=25, height=15)
    np.testing.assert_array_equal(result.values, result2.values)


def test_bump_decay_strongest_at_center_1102():
    """Pixels adjacent to center should be taller than pixels far from center.

    Regression test for #1102: decay formula was inverted (d2/s instead
    of (s-d2)/s), giving more height to farther pixels.
    """
    from xrspatial.bump import _finish_bump

    locs = np.array([[5, 5]], dtype=np.uint16)
    heights = np.array([10.0])
    out = _finish_bump(11, 11, locs, heights, spread=3)

    center = out[5, 5]
    adjacent = out[5, 6]  # 1 pixel away
    far = out[5, 8]       # 3 pixels away (edge of spread)

    assert center > adjacent > 0, f"center={center}, adjacent={adjacent}"
    assert adjacent > far, f"adjacent={adjacent}, far={far}"


def test_bump_spread_reaches_both_sides_1102():
    """Spread should reach pixels on both sides of center.

    Regression test for #1102: range upper bound excluded x+spread pixel,
    making the bump one pixel short on the positive side.
    """
    from xrspatial.bump import _finish_bump

    locs = np.array([[5, 5]], dtype=np.uint16)
    heights = np.array([10.0])
    out = _finish_bump(11, 11, locs, heights, spread=3)

    # Pixels 1 step away on BOTH sides should be reached
    assert out[5, 4] > 0, "left-1 should be > 0"
    assert out[5, 6] > 0, "right-1 should be > 0"
    assert out[4, 5] > 0, "up-1 should be > 0"
    assert out[6, 5] > 0, "down-1 should be > 0"

    # Pixels well outside spread should be 0
    assert out[5, 9] == 0, "well outside spread should be 0"
    assert out[0, 0] == 0, "corner should be 0"


# --- Issue #1206 regression tests ---

def test_bump_locs_use_int32_not_uint16():
    """Coordinates > 65535 must not wrap around (#1206)."""
    from xrspatial.bump import _finish_bump

    # Place a bump at x=70000. With uint16 this wraps to 4464.
    locs = np.array([[70000, 50]], dtype=np.int32)
    heights = np.array([10.0])
    out = _finish_bump(80000, 100, locs, heights, spread=0)

    assert out[50, 70000] > 0, "bump should land at the actual coordinate"
    assert out[50, 4464] == 0, "uint16 wrap-around location should be empty"


def test_bump_default_count_capped(monkeypatch):
    """Default count should not exceed _MAX_DEFAULT_COUNT (#1206)."""
    import sys

    from xrspatial.bump import _MAX_DEFAULT_COUNT

    # The 20000 x 20000 raster is 3.2 GB, which exceeds the memory guard on
    # small CI runners (#1234). Stub available memory so the guard does not
    # interfere with this count-cap assertion. xrspatial.__init__ rebinds
    # the `bump` name to the function, so reach the module via sys.modules.
    bump_mod = sys.modules["xrspatial.bump"]
    monkeypatch.setattr(bump_mod, "_available_memory_bytes", lambda: 64 * 1024**3)

    # 20000 x 20000 → w*h//10 = 40M, should be capped to 10M
    result = bump(width=20000, height=20000, spread=0)
    assert result.shape == (20000, 20000)
    # The number of non-zero pixels can't exceed count (no spread means
    # only center pixels get values), so verify the cap took effect
    nonzero = np.count_nonzero(result.values)
    assert nonzero <= _MAX_DEFAULT_COUNT


def test_bump_memory_guard_raises():
    """Memory guard should reject huge explicit count (#1206)."""
    import pytest

    # Request more bumps than memory can hold (50 billion)
    with pytest.raises(MemoryError, match="bump.*count"):
        bump(width=100, height=100, count=50_000_000_000)


@dask_array_available
def test_bump_dask_closure_size():
    """Dask task graph should not serialize full locs into every chunk (#1206).

    With per-chunk partitioning, the total serialized data in the graph
    should be proportional to the total bump count, not count * n_chunks.
    """
    import pickle

    agg = xr.DataArray(
        da.zeros((100, 100), chunks=(25, 25), dtype=np.float64),
        dims=['y', 'x'],
    )
    np.random.seed(42)
    result = bump(agg=agg, count=200, spread=1)
    graph_bytes = len(pickle.dumps(result.data.__dask_graph__()))

    # With the old closure approach, 200 bumps * 16 chunks * 16 bytes/bump
    # = ~51 kB minimum from duplicated arrays.  With partitioning, the total
    # is ~200 * 16 = 3.2 kB plus overhead.  Allow generous headroom but
    # ensure it's well below the duplication threshold.
    max_expected = 200 * 16 * 4  # 4x overhead for pickle framing
    assert graph_bytes < max_expected, (
        f"graph size {graph_bytes} bytes suggests locs are duplicated per chunk"
    )


@dask_array_available
def test_bump_dask_partitioned_matches_numpy():
    """Partitioned dask path should produce the same result as numpy
    when all bumps fit in a single chunk (#1206)."""
    agg_np = xr.DataArray(np.zeros((30, 40)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(30, 40))

    np.random.seed(123)
    result_np = bump(agg=agg_np, count=50, spread=2)

    np.random.seed(123)
    result_dask = bump(agg=agg_dask, count=50, spread=2)

    np.testing.assert_array_equal(result_np.values, result_dask.values)


# --- Issue #1231 regression tests ---

def test_bump_raster_memory_guard_rejects_pathological_size():
    """Memory guard should reject huge width*height even with a small
    count (#1231).

    Pre-fix: ``bump(width=1_000_000, height=1_000_000)`` would pass the
    bump-count guard (default count capped at 10M ~ 160 MB) and then
    allocate an 8 TB float64 output raster inside ``_finish_bump``.
    """
    import pytest

    with pytest.raises(MemoryError, match=r"bump.*width=1,000,000"):
        bump(width=1_000_000, height=1_000_000)


def test_bump_raster_memory_guard_mentions_raster_bytes():
    """Error message should name the raster allocation as the culprit
    when the raster dominates the budget (#1231)."""
    import pytest

    with pytest.raises(MemoryError, match="output raster"):
        bump(width=500_000, height=500_000, count=1)


@dask_array_available
def test_bump_dask_count_guard_reports_location_arrays():
    """On a dask agg the raster is never materialized, so a runaway
    ``count`` trips the guard on the location/height arrays alone and the
    message must name those (not the output raster) as the culprit (#1231)."""
    import pytest

    agg = xr.DataArray(
        da.zeros((1_000, 1_000), chunks=(500, 500), dtype=np.float64),
        dims=['y', 'x'],
    )
    with pytest.raises(MemoryError, match="location/height arrays"):
        bump(agg=agg, count=50_000_000_000)


@dask_array_available
def test_bump_dask_bypasses_raster_guard():
    """Dask paths build the output lazily, so the raster-size guard
    must not reject a huge dask-backed agg (#1231)."""
    # 100k x 100k float64 would be 80 GB if materialized, but the dask
    # backend never holds the full array in memory.
    agg = xr.DataArray(
        da.zeros((100_000, 100_000), chunks=(1_000, 1_000), dtype=np.float64),
        dims=['y', 'x'],
    )
    result = bump(agg=agg, count=10, spread=0)
    assert result.shape == (100_000, 100_000)
    assert isinstance(result.data, da.Array)


# --- Parameter coverage: custom height_func ---

def test_bump_custom_height_func():
    """The public ``height_func`` argument is the main customization point
    but the default-``None`` path is all that other tests exercise.  A
    custom function must actually drive the output magnitudes."""
    def constant_height(locs):
        return np.full(len(locs), 7.0)

    np.random.seed(7)
    # spread=0 keeps only the centre pixels, so every non-zero cell is the
    # height the custom function returned (up to per-pixel accumulation).
    result = bump(width=40, height=40, count=3, spread=0,
                  height_func=constant_height)
    nz = result.values[result.values != 0]
    assert nz.size > 0
    # Values are sums of the constant 7.0 over coincident bumps.
    assert np.all(nz % 7.0 == 0)
    assert result.values.max() >= 7.0


def test_bump_custom_height_func_through_agg():
    """height_func must also flow through the agg (backend-dispatch) path."""
    def constant_height(locs):
        return np.full(len(locs), 3.0)

    agg = xr.DataArray(np.zeros((30, 30)), dims=['y', 'x'])
    np.random.seed(11)
    result = bump(agg=agg, count=2, spread=0, height_func=constant_height)
    nz = result.values[result.values != 0]
    assert nz.size > 0
    assert np.all(nz % 3.0 == 0)


# --- Geometric edge cases ---

def test_bump_single_pixel_raster():
    """A 1x1 raster must not raise on the spread-clamping loops."""
    result = bump(width=1, height=1, count=1, spread=1)
    assert result.shape == (1, 1)
    assert result.values[0, 0] > 0


def test_bump_single_column_strip():
    """A 1-wide (Nx1) strip exercises kernel-boundary clamping in x."""
    result = bump(width=1, height=10, count=3, spread=2)
    assert result.shape == (10, 1)
    assert np.count_nonzero(result.values) > 0


def test_bump_single_row_strip():
    """A 1-tall (1xN) strip exercises kernel-boundary clamping in y."""
    result = bump(width=10, height=1, count=3, spread=2)
    assert result.shape == (1, 10)
    assert np.count_nonzero(result.values) > 0


# --- Backend coverage: empty (bump-free) chunks in the dask paths ---

@dask_array_available
def test_bump_dask_numpy_sparse_chunks_match_numpy():
    """Multi-chunk dask+numpy with few bumps leaves some chunks empty
    (the ``part is None`` branch), which must still match numpy."""
    agg_np = xr.DataArray(np.zeros((40, 40)), dims=['y', 'x'])
    agg_dask = agg_np.copy()
    agg_dask.data = da.from_array(agg_dask.data, chunks=(20, 20))

    np.random.seed(1)
    result_np = bump(agg=agg_np, count=3, spread=1)

    np.random.seed(1)
    result_dask = bump(agg=agg_dask, count=3, spread=1)

    assert isinstance(result_dask.data, da.Array)
    np.testing.assert_array_equal(result_np.values, result_dask.values)


@dask_array_available
@cuda_and_cupy_available
@pytest.mark.xfail(
    reason="dask+cupy empty-chunk path appends numpy da.zeros; da.block "
           "then fails to concatenate numpy and cupy chunks (bump.py:146). "
           "Known backend bug surfaced by the test-coverage sweep 2026-07-02.",
    strict=False,
)
def test_bump_dask_cupy_sparse_chunks_match_numpy():
    """Multi-chunk dask+cupy with bump-free chunks must match numpy.

    Currently raises ``TypeError`` on compute because empty chunks are
    materialized as numpy ``da.zeros`` and cannot be concatenated with the
    cupy bump chunks.  Dense inputs (every chunk has a bump) work.
    """
    import cupy

    agg_dc = xr.DataArray(
        da.from_array(cupy.zeros((40, 40)), chunks=(20, 20)),
        dims=['y', 'x'],
    )
    agg_np = xr.DataArray(np.zeros((40, 40)), dims=['y', 'x'])

    np.random.seed(1)
    result_np = bump(agg=agg_np, count=3, spread=1)

    np.random.seed(1)
    result_dc = bump(agg=agg_dc, count=3, spread=1)

    computed = result_dc.data.compute()
    np.testing.assert_array_equal(result_np.values, computed.get())
