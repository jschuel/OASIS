# OASIS Overlap-Aware Segmentation of ImageS

OASIS is a Pytorch-based pixel-level segmentation-regression framework designed to
separate overlapping objects in scientific image data. OASIS employs custom loss
function with region-specific weights that can be tuned to prioritize overlapping
pixels during training. A paper detailing the framework and its performance
on data from the MIGDAL experiment is in progress and will be linked soon.

![OASISGif](figures/OASIS.gif)

## Installation

1. Navigate to the parent `OASIS` directory and install the package with

```python
pip install -e .
```

**Note**: Installing OASIS doesn't install any other python packages
so you'll need to install all dependencies. The main ones are
-numpy
-pandas
-pytorch
-matplotlib
-tqdm

## Package Layout
```
OASIS/
  config.py           # OASISConfig (single dataclass)
  datasets.py         # OASISDataset in pytorch
  model.py            # UNetSmall (GroupNorm+SiLU, Softplus head)
  losses.py           # SegRegLoss + masked_tv (smoothness)
  train.py            # train(cfg)
  eval.py	      # evaluate(cfg), view_and_process()
  generate_tensors.py # Script to generate pytorch tensors from dataframe contents
```
