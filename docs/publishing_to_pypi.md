# Publishing Chakra to PyPI

Step-by-step guide to make Chakra installable via `pip install chakra_auto_research`.

---

## Prerequisites

- A GitHub repo with all changes committed and pushed
- Python 3.10+ installed
- Your `pyproject.toml` is already configured (done ✅)

---

## Step 1: Create a PyPI Account

1. Go to **[https://pypi.org/account/register/](https://pypi.org/account/register/)**
2. Create your account and **verify your email**
3. Enable **2FA** (PyPI requires it for uploads)

---

## Step 2: Create an API Token

1. Go to **[https://pypi.org/manage/account/](https://pypi.org/manage/account/)**
2. Scroll to **"API tokens"** → click **"Add API token"**
3. Name: `chakra-upload`
4. Scope: **"Entire account"** (you can scope to project after the first upload)
5. Click **"Add token"**
6. **Copy the token** — it starts with `pypi-` and is shown only once

> ⚠️ Save this token somewhere safe (password manager). You cannot view it again.

---

## Step 3: (Optional) Create a TestPyPI Account

TestPyPI is a separate instance for testing. **Strongly recommended for first-time publishing.**

1. Go to **[https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)**
2. Create account, verify email, enable 2FA
3. Create an API token (same process as Step 2)

---

## Step 4: Install Build Tools

```powershell
pip install build twine
```

| Tool | Purpose |
|------|---------|
| `build` | Creates distributable `.tar.gz` and `.whl` files |
| `twine` | Uploads packages to PyPI securely |

---

## Step 5: Build the Package

```powershell
cd "c:\D Drive\Projects\6th Sem\Chakra - Autonomous Research System"
python -m build
```

This creates two files in `dist/`:

```
dist/
├── chakra_auto_research-0.3.0.tar.gz            # Source distribution
└── chakra_auto_research-0.3.0-py3-none-any.whl  # Wheel (pre-built, fast install)
```

### Verify the build

```powershell
# Check the package contents look correct
python -m twine check dist/*
```

You should see: `PASSED` for both files.

---

## Step 6: Upload to TestPyPI (Dry Run)

```powershell
python -m twine upload --repository testpypi dist/*
```

When prompted:
- **Username:** `__token__`
- **Password:** paste your **TestPyPI** API token (the `pypi-...` string)

### Verify on TestPyPI

```powershell
# Install from TestPyPI (may fail for dependencies — that's normal)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chakra_auto_research
```

> The `--extra-index-url` flag tells pip to fetch dependencies (torch, wandb, etc.) from real PyPI, since TestPyPI doesn't host them.

Check it works:

```powershell
chakra list-domains
```

If that succeeds, you're ready for the real thing.

### Clean up test install

```powershell
pip uninstall chakra_auto_research
```

---

## Step 7: Upload to Real PyPI

```powershell
python -m twine upload dist/*
```

When prompted:
- **Username:** `__token__`
- **Password:** paste your **real PyPI** API token

### Alternative: Store credentials so you don't re-enter them

Create a file at `%USERPROFILE%\.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-REAL-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

Then uploads just work without prompts:

```powershell
python -m twine upload dist/*               # → real PyPI
python -m twine upload --repository testpypi dist/*  # → TestPyPI
```

---

## Step 8: Verify the Published Package

```powershell
# Install from PyPI
pip install chakra_auto_research

# Verify the CLI works
chakra list-domains

# Verify the Python import works
python -c "import chakra; print('Chakra imported successfully')"
```

Your package is now live at:

**https://pypi.org/project/chakra_auto_research/**

---

## Step 9: Update the README Badge

Add a PyPI badge to your `README.md` after the first successful publish:

```markdown
[![PyPI](https://img.shields.io/pypi/v/chakra_auto_research.svg)](https://pypi.org/project/chakra_auto_research/)
```

---

## Publishing Future Versions

When you make changes and want to release a new version:

### 1. Bump the version in `pyproject.toml`

```toml
version = "0.4.0"  # was 0.3.0
```

### 2. Clean old builds

```powershell
Remove-Item -Recurse -Force dist
```

### 3. Build and upload

```powershell
python -m build
python -m twine upload dist/*
```

### Version numbering convention

| Version | When to Use |
|---------|-------------|
| `0.3.1` | Bug fixes, minor improvements |
| `0.4.0` | New features (e.g., new domain, new CLI command) |
| `1.0.0` | First stable release (Aavart works end-to-end, all domains verified) |

---

## Troubleshooting

### "File already exists" error

PyPI doesn't allow overwriting a version. You must bump the version number.

### "Invalid or non-existent authentication"

- Make sure you're using `__token__` as the username (literally the string `__token__`)
- Make sure the password is the full token including the `pypi-` prefix
- Make sure you're using the right token for the right server (TestPyPI ≠ PyPI)

### Package installs but `chakra` command not found

The `chakra` CLI is registered via `[project.scripts]` in `pyproject.toml`. After installing:

```powershell
# Check where it's installed
where chakra

# If not found, make sure your venv/Scripts is on PATH
# Or run directly:
python -m chakra.chakra_cli
```

### Build fails with "module not found"

Make sure you're building from the project root:

```powershell
cd "c:\D Drive\Projects\6th Sem\Chakra - Autonomous Research System"
python -m build
```

---

## Quick Reference

```powershell
# One-time setup
pip install build twine

# Build
python -m build

# Check
python -m twine check dist/*

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to real PyPI
python -m twine upload dist/*

# Verify
pip install chakra-research
chakra list-domains
```

---

## What Users Will Do

After you publish, anyone in the world can run:

```bash
pip install chakra-research
chakra aavart --domain tabular_cls --version v1.0 --device cpu --force
```

That's it. 🎉
