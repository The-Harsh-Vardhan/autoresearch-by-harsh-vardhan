# Troubleshooting

Common issues when using the Chakra CLI usually fall into a small number of buckets.

## `--domain` is required

Some commands need a domain because the lifecycle depends on a registered manifest.

Fix:

```bash
python -m chakra list-domains
```

Then rerun the command with one of the discovered domain names.

## Domain not found

If a domain is missing, check that the domain package exists under `src/chakra/domains/` and that its `domain.yaml` is present.

## Version contract failures

If `validate-version` fails, one of the required assets is missing or malformed.

Check:

- notebook path
- docs path
- review path
- config files

## Kaggle or execution failures

If `run-execution` fails during the remote path, try:

- `--strategy local` to isolate the problem
- `--dry-run` to inspect the decision flow
- `python -m chakra validate-version --domain <name> --version <version>` to confirm the version is intact

## Clean reinstall

If the CLI itself seems out of sync with the repo, reinstall the editable package:

```bash
pip install -e ".[dev]"
```

## Still stuck

Start with the [How to Use](../how_to_use.md) guide. It includes the full lifecycle walkthrough and domain-specific notes.