"""Backwards-compatible re-exports. Prefer autoresearch_hv.core.utils + autoresearch_hv.domains.hndsr_vr.utils."""

# Re-export core utilities at the old location
from autoresearch_hv.core.utils import (  # noqa: F401
    REPO_ROOT,
    repo_path,
    ensure_dir,
    prepare_workspace_temp,
    load_config,
    write_json,
    read_json,
    write_text,
    set_seed,
    get_device,
    flatten_config,
    describe_run_dirs,
)

# Re-export HNDSR-specific utilities
from autoresearch_hv.domains.hndsr_vr.utils import (  # noqa: F401
    VERSION_PATTERN,
    version_stem,
    version_slug,
)
