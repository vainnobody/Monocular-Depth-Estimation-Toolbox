# Vaihingen RingMoE Protocol

This repository now includes a dedicated Vaihingen protocol aligned with the setup reported by RingMoE and later remote-sensing height-estimation papers.

## What changed

- The new dataset base config is [`configs/_base_/datasets/vaihingen_ringmoe.py`](/Users/lanjie/Proj/SSL/Monocular-Depth-Estimation-Toolbox/configs/_base_/datasets/vaihingen_ringmoe.py).
- It follows the official 16-train / 17-test tile split and normalizes raw DSM heights from `240.70-360.00 m` into `[1e-3, 1.0]`.
- The new model configs are:
  - [`configs/adabins/adabins_efnetb5ap_vaihingen_ringmoe_24e.py`](/Users/lanjie/Proj/SSL/Monocular-Depth-Estimation-Toolbox/configs/adabins/adabins_efnetb5ap_vaihingen_ringmoe_24e.py)
  - [`configs/binsformer/binsformer_resnet50_vaihingen_ringmoe_24e.py`](/Users/lanjie/Proj/SSL/Monocular-Depth-Estimation-Toolbox/configs/binsformer/binsformer_resnet50_vaihingen_ringmoe_24e.py)

## Expected data layout

The default split files assume a flat layout:

```text
data/vaihingen/
  image/top_mosaic_09cm_area1.tif
  dsm/top_mosaic_09cm_area1.tif
  ...
```

If your data is nested, for example under `data/vaihingen/train/image` and `data/vaihingen/train/dsm`, regenerate the split files:

```bash
python3 tools/misc/generate_vaihingen_splits.py --data-root data/vaihingen
```

The generator scans recursively and writes split entries that match your actual on-disk paths.

## Re-evaluate an existing checkpoint output

If you already have `tools/test.py --out pred.pkl` from a model trained on raw DSM values, compare the two protocols with:

```bash
python3 tools/misc/eval_vaihingen_predictions.py pred.pkl \
  --data-root data/vaihingen \
  --split splits/vaihingen/ringmoe/test.txt \
  --pred-domain raw \
  --compare-raw-domain
```

This lets you verify whether `a1/a2/a3` collapse toward `1.0` only in the raw DSM domain.
