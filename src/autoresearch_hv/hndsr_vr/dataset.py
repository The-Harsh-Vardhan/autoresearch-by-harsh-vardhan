"""Backwards-compatible re-exports. Prefer autoresearch_hv.domains.hndsr_vr.dataset."""

from autoresearch_hv.domains.hndsr_vr.dataset import (  # noqa: F401
    DatasetBundle,
    SatellitePairDataset,
    SyntheticSatellitePairDataset,
    build_loaders,
    build_split_manifest,
)
