# RTX 3060 RGB offset pilot

The archive includes scripts, five fixed evaluation HDF5 files, source
feature caches, the three frozen denoisers, fitted regressors, 15 external
result files, and the original analysis outputs. See PROTOCOL.md for the
design and interpretation. No cloud credentials are required.

## Reproduce

From the extracted directory with Python 3.11:

1. Check the archived inputs with `shasum -a 256 -c SHA256SUMS.txt`.
2. Run `bash run.sh summarize` to regenerate the table from saved results
   with NumPy, without image inference.
3. Optionally run `bash run.sh fit` to refit the predictors on the included
   CPU feature caches. This rewrites the fitted arrays and source results
   in the extracted directory. The archived original summaries are under
   reference/. For exact-weight evaluation, use a fresh copy of the package.
4. Install PyTorch 2.4.0 CUDA 12.1 with torchvision 0.19.0 and install
   requirements-eval.txt. The original 3060 evaluation reported PyTorch
   2.4.0+cu121 and NumPy 2.1.2. The dependency list does not assert exact
   original versions for the other packages.
5. Run `bash run.sh evaluate` on a CUDA GPU. The included PIQ DISTS
   coefficients are installed in the local Torch cache. AlexNet and VGG16
   weights are fetched from the PyTorch model host when missing.

The printed PSNR values are differences from each uncorrected frozen A
output. In particular, +1.024 dB is the Ridge result, and +0.671 dB is the
small MLP result on the same CC-noise images.
