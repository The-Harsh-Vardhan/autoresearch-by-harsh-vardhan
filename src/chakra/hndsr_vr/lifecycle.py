"""Backwards-compatible re-exports. Prefer chakra.domains.hndsr_vr.lifecycle."""

from chakra.domains.hndsr_vr.lifecycle import LifecycleHooks  # noqa: F401

# Re-export the old function-based API by wrapping the hooks class
_hooks = LifecycleHooks()

from chakra.core.lifecycle import (  # noqa: F401,E402
    scaffold_version as _scaffold,
    validate_version as _validate,
    push_kaggle as _push,
    kaggle_status as _status,
    pull_kaggle as _pull,
    sync_run as _sync,
    review_run as _review,
    mirror_obsidian as _mirror,
    next_ablation as _next,
)
from functools import partial  # noqa: E402

scaffold_version = partial(_scaffold, "hndsr_vr")
validate_version = partial(_validate, "hndsr_vr")
push_kaggle = partial(_push, "hndsr_vr")
kaggle_status = partial(_status, "hndsr_vr")
pull_kaggle = partial(_pull, "hndsr_vr")
sync_run = partial(_sync, "hndsr_vr")
review_run = partial(_review, "hndsr_vr")
mirror_obsidian = partial(_mirror, "hndsr_vr")
next_ablation = partial(_next, "hndsr_vr")

# Re-export resolve_version_paths at the old location
resolve_version_paths = _hooks.resolve_version_paths
render_notebook = _hooks.render_notebook
