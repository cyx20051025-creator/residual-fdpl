# Open-Source Decisions

The following decisions remain with the authors:

1. Resolved: the project code is licensed under Apache-2.0.
2. Confirm whether datasets and derived HDF5 caches may be redistributed.
3. Resolved for the review release: the six core final checkpoints may be
   distributed as Release assets. External baseline weights remain out of
   scope.
4. Confirm whether the first release contains only the v3.1 canonical pipeline.
5. Confirm the GitHub owner and repository visibility.
6. Confirm the supported environment matrix.
7. Confirm the final paper title, authors, DOI, and BibTeX metadata.
8. Resolved for the first release: NAFNet, RIDNet, SwinIR, and other external
   baselines are not included; upstream terms remain in
   `THIRD_PARTY_NOTICES.md`.
9. Resolved: the author email remains public in `pyproject.toml`.
10. Resolved for the first release: the public repository is source-only.
    Paper result JSON files stay in the private asset repository.

Until these are resolved, the repository remains a preparation tree and must
not be published as an open-source release.
