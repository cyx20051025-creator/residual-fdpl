# Assets

This directory is reserved for the public download manifest after the asset
repository has an approved URL.

No dataset, checkpoint, HDF5 cache, or generated figure is bundled in this
repository. Model and result redistribution will be decided after the upstream
license review.

The prepared private asset bundle contains:

```text
weights/
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
