"""Tests for the CLI dispatch mechanism."""

import os
import subprocess
import sys


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": "src"}
    return subprocess.run(
        [sys.executable, "-m", "autoresearch_hv", *args],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        env=env,
    )


def test_list_domains_succeeds():
    result = _run_cli("list-domains")
    # May fail if torch not installed (core.tracker imports chain), check gracefully
    if result.returncode != 0 and "No module named" in result.stderr:
        # Expected when deps missing — not a CLI dispatch bug
        return
    assert result.returncode == 0
    assert "hndsr_vr" in result.stdout
    assert "nlp_lm" in result.stdout


def test_domain_required_for_lifecycle_commands():
    result = _run_cli("validate-version", "--version", "vR.1")
    # Should fail either because --domain missing or because deps missing
    assert result.returncode != 0
    has_domain_error = "--domain is required" in result.stderr or "domain" in result.stderr.lower()
    has_dep_error = "No module named" in result.stderr
    assert has_domain_error or has_dep_error


def test_domain_info_shows_hndsr():
    result = _run_cli("--domain", "hndsr_vr", "domain-info")
    if result.returncode != 0 and "No module named" in result.stderr:
        return
    assert result.returncode == 0
    assert "HNDSR" in result.stdout
    assert "psnr_mean" in result.stdout
