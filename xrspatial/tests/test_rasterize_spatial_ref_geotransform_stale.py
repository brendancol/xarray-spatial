"""rasterize() must not hand back a stale ``GeoTransform`` when it
reshapes the grid of a rioxarray template.

rioxarray stores the affine transform twice: once as ``attrs['transform']``
on the array and again inside the ``spatial_ref`` grid-mapping coord as
``attrs['GeoTransform']`` (a space-separated GDAL 6-tuple).  ``rio.transform()``
and ``rio.resolution()`` prefer the cached ``GeoTransform`` over recomputing
from the x/y coords.

``rasterize(like=template, ...)`` carries the ``spatial_ref`` coord through
verbatim.  When the caller overrides ``resolution`` / ``bounds`` /
``width`` + ``height`` so the output grid no longer matches the template,
the freshly built x/y coords are correct but the inherited ``GeoTransform``
still describes the template's grid -- so a downstream
``result.rio.reproject()`` / ``.rio.to_raster()`` is silently
misgeoreferenced.  This is the same stale-grid metadata lie the existing
``res`` / ``transform`` attr strip guards against, reaching downstream
through the coord channel instead.

The fix drops ``GeoTransform`` from the carried-through ``spatial_ref``
coord whenever the grid is reshaped (so rioxarray recomputes it from the
correct coords), while preserving it when the output reuses the template's
coords bit-identically.
"""
import numpy as np
import pytest
import xarray as xr

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

try:
    import rioxarray  # noqa: F401
    has_rioxarray = True
except ImportError:
    has_rioxarray = False

try:
    import dask.array as da  # noqa: F401
    has_dask = True
except ImportError:
    has_dask = False

try:
    import cupy  # noqa: F401
    from numba import cuda
    has_cuda = cuda.is_available()
except Exception:
    has_cuda = False

pytestmark = [
    pytest.mark.skipif(not has_shapely, reason="shapely not installed"),
    pytest.mark.skipif(not has_rioxarray, reason="rioxarray not installed"),
]

skip_no_dask = pytest.mark.skipif(not has_dask, reason="dask not installed")
skip_no_cuda = pytest.mark.skipif(not has_cuda, reason="CUDA not available")


def _template(width=10, height=10):
    """A rioxarray-style EPSG:3857 template carrying a cached GeoTransform."""
    x = np.linspace(0.5, width - 0.5, width)
    y = np.linspace(height - 0.5, 0.5, height)
    t = xr.DataArray(
        np.zeros((height, width)), dims=['y', 'x'],
        coords={'y': y, 'x': x},
    )
    return t.rio.write_crs("EPSG:3857").rio.write_transform()


def _geoms():
    return [(box(0, 0, 10, 10), 1.0)]


def _backend_kwargs():
    cases = [("numpy", {}), ("dask+numpy", {"chunks": 8})]
    if has_cuda:
        cases += [("cupy", {"gpu": True}),
                  ("dask+cupy", {"gpu": True, "chunks": 8})]
    return cases


@pytest.mark.parametrize(
    "label,kw",
    _backend_kwargs(),
    ids=[c[0] for c in _backend_kwargs()],
)
def test_resolution_reshape_drops_stale_geotransform(label, kw):
    t = _template()
    out = rasterize(_geoms(), like=t, resolution=0.5, **kw)

    assert out.shape == (20, 20)
    # The stale template GeoTransform must not survive a reshape.
    assert 'GeoTransform' not in out.spatial_ref.attrs
    # rioxarray's cached-vs-recomputed transform now agree (no stale lie).
    assert out.rio.resolution() == (0.5, -0.5)
    assert out.rio.resolution() == out.rio.resolution(recalc=True)
    # CRS still propagates (only the transform cache was dropped).
    assert out.rio.crs.to_epsg() == 3857


def test_bounds_override_drops_stale_geotransform():
    t = _template()
    out = rasterize(_geoms(), like=t, bounds=(0, 0, 5, 5))
    assert 'GeoTransform' not in out.spatial_ref.attrs


def test_width_height_override_drops_stale_geotransform():
    t = _template()
    out = rasterize(_geoms(), like=t, width=5, height=5, bounds=(0, 0, 10, 10))
    assert out.shape == (5, 5)
    assert 'GeoTransform' not in out.spatial_ref.attrs


def test_reuse_coords_preserves_geotransform():
    """No reshape: the template's GeoTransform is correct and must stay."""
    t = _template()
    gt = t.spatial_ref.attrs['GeoTransform']
    out = rasterize(_geoms(), like=t)
    assert out.shape == (10, 10)
    assert out.spatial_ref.attrs.get('GeoTransform') == gt
    assert out.rio.resolution() == out.rio.resolution(recalc=True) == (1.0, -1.0)


def test_reshape_does_not_mutate_template_coord():
    """Stripping GeoTransform must copy the coord, not edit the template."""
    t = _template()
    gt_before = t.spatial_ref.attrs['GeoTransform']
    rasterize(_geoms(), like=t, resolution=0.5)
    assert t.spatial_ref.attrs['GeoTransform'] == gt_before
