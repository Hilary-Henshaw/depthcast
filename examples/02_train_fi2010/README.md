# 02 - Train on FI-2010

This example trains DepthCast on the real FI-2010 benchmark instead of
synthetic data. It requires the dataset and benefits from a GPU.

## Get the data

FI-2010 is published by Ntakaris et al. and is free for research. Download
the `NoAuction` decimal-precision files, for example:

- `Train_Dst_NoAuction_DecPre_CF_7.txt`
- `Test_Dst_NoAuction_DecPre_CF_7.txt`

Place them somewhere accessible and note the paths. The files are large
(several gigabytes uncompressed).

## Train

Using the CLI with the provided config:

```bash
cd examples/02_train_fi2010
depthcast train --config config.yaml \
  --train-file /path/to/Train_Dst_NoAuction_DecPre_CF_7.txt \
  --val-fraction 0.2
```

Or with the Python script, which accepts the path as an argument:

```bash
python train.py /path/to/Train_Dst_NoAuction_DecPre_CF_7.txt
```

## Evaluate

```bash
depthcast evaluate --checkpoint checkpoints/best.pt \
  --test-file /path/to/Test_Dst_NoAuction_DecPre_CF_7.txt
```

## Notes

- `config.yaml` uses the full-size architecture (32 conv channels). On a
  CPU this is slow; prefer a GPU, or reduce the channel counts for a quick
  trial.
- The default horizon is `horizon_index: 4` (predicting 100 events
  ahead). Edit the config to target a different horizon.
