# HOLISMOKES XXI — lensed SN Ia detection from LSST-like time series

Code for:

> **HOLISMOKES XXI: Detecting strongly lensed type Ia supernovae from time series
> of multi-band LSST-like imaging data**
> [arXiv:2605.05318](https://arxiv.org/abs/2605.05318)

A ConvLSTM network that scores each epoch of a multi-band image time series as
lensed SN Ia or not. Inputs are the image cube and the observation times.

## Data

The paper trains on ~128 000 systems (~200 GB). `demo_data/` ships a small subset
so the notebook runs end to end. Per split (`Train/`, `Val/`, `Test/`):

| file | shape | contents |
|------|-------|----------|
| `All_bands_arranged_<split>_sam_images_pad0.npz` | `(N, 14, 59, 59, 4)` | image cube: 14 epochs × 59×59 px × 4 bands (*griz*), `float16` |
| `All_bands_arranged_<split>_sam_epochs.npy` | `(N, 14)` | normalised observation time per epoch |
| `All_<split>_sam_labels.npy` | `(N, 14)` | binary label per epoch |
| `label_and_details_<split>.txt` | `(N, 2)` | class code, catalogue ID |
| `idxs_<split>.npy` | `(N,)` | indices into the full dataset |

Images are zlib-compressed to stay under GitHub's file limit; the loader reads
`.npz` or memory-mapped `.npy`. Class codes `2` and `4` are positive, `0`, `-21`,
`-22`, `-31` negative.

In the published version the validation and test sets share the same composition,
with the negative class boosted. Here the demo validation and test sets are kept
class-balanced instead (`DEMO_BALANCED` in CONFIG), so that the improvement during
training is still visible on sets this small.

## Running it

```bash
jupyter notebook FIMN_JT01.ipynb
```

Everything you need to change is in the **CONFIG** cell: set `DATA_DIR` and
`IMG_SHAPE` for your data, then

```python
DEMO_DATA = True     # small demo set
DEMO_DATA = False    # production: full dataset
```

`False` runs exactly as the published job did. `True` adds the two pipeline
settings a few-hundred-sample set needs, `STEPS_PER_EPOCH` and `RECALIBRATE_BN`.
The network, optimiser and learning-rate schedule are identical either way.

Training writes to `Results_run<RUN_ID>_.../`, keeps the best model by `val_loss`
and rolls back after `NHOLD` bad epochs. The last cell loads the best epoch and
plots the test-set score distribution.

**This is a smoke test, not a reproduction.**
