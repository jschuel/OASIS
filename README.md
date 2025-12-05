# OASIS Overlap-Aware Segmentation of ImageS

OASIS is a Pytorch-based pixel-level segmentation-regression framework designed to
separate overlapping objects in scientific image data. OASIS employs a custom loss
function with region-specific weights that can be tuned to prioritize overlapping
pixels during training. See [our paper](https://arxiv.org/abs/2510.06194) for more
details about the framework. Check back here for updates -- detailed documentation,
code updates, and sample data are coming soon!

![OASISGif](figures/OASIS.gif)

## Installation

0. **(Recommended)**: Create and activate a dedicated anaconda environment for OASIS:

```
conda create -n OASIS python=3.10
```

then

```
conda activate OASIS
```

1. Clone the OASIS repository

```
git clone https://github.com/jschuel/OASIS.git
```

or

```
git clone git@github.com:jschuel/OASIS.git
```

2. Navigate to the parent `OASIS` directory and install the package with

```
pip install -e .
```

Running this will install all required dependencies **except** pytorch.

3. Follow the instructions on the installation guide of the front page of the [PyTorch website](https://pytorch.org/) to install pytorch. The “compute platform” row allows you to select the relevant platform for your GPU or “CPU only” if you do not have a compatible GPU

4. Navigate to OASIS/OASIS and run

```
source setup_environment.sh
```

This script will download test data and three trained OASIS models that you can use to test the software

5. That's it! Now you can try out OASIS by running the Jupyter Notebook tutorial at `OASIS/OASIS/tutorial.ipynb`

## Package Layout
```
OASIS/
  config.py            # OASISConfig (single dataclass)
  datasets.py          # OASISDataset in pytorch
  model.py             # UNetSmall (GroupNorm+SiLU, Softplus head)
  losses.py            # SegRegLoss + masked_tv (smoothness)
  train.py             # train(cfg)
  eval.py	       # evaluate(cfg), view_and_process()
  generate_tensors.py  # Script to generate pytorch tensors from dataframe contents
  setup_environment.sh # Script to download data and place it in target directories
```
