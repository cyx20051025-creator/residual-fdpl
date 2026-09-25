# Security Policy

## Reporting

Do not open a public issue for credentials, private datasets, or unpublished
paper material. Report security concerns directly to the repository owner.

## Repository Rules

The following must never be committed:

- remote host passwords or private keys
- cloud access tokens or `.env` files
- upload logs containing host or authentication context
- SIDD, Urban100, BSDS500, or other dataset copies
- training HDF5 caches
- model checkpoints unless explicitly approved as a Release asset
- unpublished manuscripts, reviewer material, or paper source

Before the first public push, run a full-history credential scan and verify that
the repository contains only intentionally public source and documentation.

