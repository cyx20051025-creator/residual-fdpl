# Contributing

The public project focuses on the canonical Residual FDPL v3.1 pipeline.
Historical model generations are not part of the initial maintenance scope.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

## Change Rules

- Keep raw data, checkpoints, logs, and paper files out of Git.
- Preserve the locked experimental constants unless a change is explicitly
  reviewed as a new protocol version.
- Add a focused test for every model, loss, metric, or data-loader change.
- Do not weaken causal, seed, subset, or cross-architecture limitations in the
  documentation.

