# Contributing

The public project maintains the Residual FDPL v3.1 pipeline and the
risk-controlled residual RGB-correction component. Historical model
generations and end-to-end third-party experiment assets remain outside the
initial maintenance scope.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

## Change Rules

- Keep raw data, checkpoints, feature caches, model weights, credentials,
  logs, and manuscript files out of Git.
- Preserve the locked experimental constants unless a change is explicitly
  reviewed as a new protocol version.
- Add a focused test for every model, loss, metric, or data-loader change.
- Add a focused CPU test for changes to RGB geometry, calibration, split
  validation, or selection behaviour.
- Do not weaken causal, seed, subset, or cross-architecture limitations in the
  documentation.
- Describe pre-clipping diagnostics and rendered post-clipping metrics as
  different quantities.
