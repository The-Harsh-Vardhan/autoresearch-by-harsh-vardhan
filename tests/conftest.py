import os
import tempfile
from pathlib import Path


def pytest_configure() -> None:
    temp_root = Path("artifacts/.tmp/temp")
    temp_root.mkdir(parents=True, exist_ok=True)
    resolved = str(temp_root.resolve())
    os.environ["TMP"] = resolved
    os.environ["TEMP"] = resolved
    os.environ["TMPDIR"] = resolved
    tempfile.tempdir = resolved
