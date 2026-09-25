# Assets

This directory is reserved for the public download manifest. The checkpoint
download links will point to the `v3.1-review` GitHub Release after the public
repository owner is assigned.

No dataset, checkpoint, HDF5 cache, or generated figure is bundled in the
public source repository. The six final checkpoints are approved for the review
Release. Dataset, quick-evaluation data, and per-image result redistribution
remain pending upstream and dataset-terms confirmation.

The prepared private asset bundle contains:

```text
checkpoints/
|-- rcab3_seed42_fdpl_final.pth
|-- rcab3_seed42_nofdpl_final.pth
|-- rcab3_seed43_fdpl_final.pth
|-- rcab3_seed43_nofdpl_final.pth
|-- rcab3_seed44_fdpl_final.pth
`-- rcab3_seed44_nofdpl_final.pth

results/raw/
|-- selected paper, mechanism, and paired-significance JSON files
`-- table-to-source mapping in results/paper/README.md

release_bundles/
`-- sidd_quick_eval_256.zip
```

The quick-evaluation bundle contains 1,024 paired SIDD validation images and
expands to approximately 185 MB. It is kept outside the normal public Git
history. The private asset repository also contains a verified `SHA256SUMS`
manifest for every staged file.
