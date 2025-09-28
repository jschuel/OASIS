
"""
Generate .pt tensors

The following two cases are supported
  - NR_only = False (new hybrid data; needs `unique_NRs` lookup)
  - NR_only = True (old hybrid data; uses row/col bounds per-row)

Assumptions about dataframe columns:
 - Sparse coordinates 'col', 'row', 'intensity', and bounding box coords ('colmin/max', 'rowmin/max') 
 - ER and NR-specific counterparts of these columns are present as well for creating truth labels

Column names need to be adjusted below if your dataframe format differs
"""
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from OASIS.utils import scale_image_log10p

H, W = 288, 512  # default; override via args

def make_im(cols, rows, weights, bins=(512,288), ranges=((0,512),(0,288))):
    h = np.histogram2d(cols, rows, weights=weights, bins=bins, range=ranges)[0].T.astype('float32')
    return h

def _rect_mask(rowmin, rowmax, colmin, colmax):
    v = np.zeros((H,W), dtype=np.float32)
    v[rowmin:rowmax+1, colmin:colmax+1] = 1.0
    return v

def _save_tuple(outdir, i, hyb_scaled, t_er_s, t_nr_s, s, ymax10, tmax, valid):
    """Deprecated, now save as a dictionary"""
    rec = (
        torch.tensor(hyb_scaled, dtype=torch.float32),
        torch.tensor(np.stack([t_er_s, t_nr_s], axis=0), dtype=torch.float32),
        float(s), float(ymax10), float(tmax),
        torch.tensor(valid, dtype=torch.float32)
    )
    torch.save(rec, os.path.join(outdir, f"{i}.pt"))


def _save_dict(outdir, i, hyb_scaled, t_er_s, t_nr_s, s, ymax10, tmax, valid):
    """
    Saves a single example in dict schema expected by OASISDataset:
      {
        "input":  [H,W] float32,
        "target": [2,H,W] float32 (ER, NR),
        "valid":  [H,W] float32 in {0,1},
        "s": float, "ymax10": float, "tmax": float,
      }
    """
    rec = {
        "input":  torch.tensor(hyb_scaled, dtype=torch.float32),                # [H,W]
        "target": torch.tensor(np.stack([t_er_s, t_nr_s], axis=0), dtype=torch.float32),  # [2,H,W]
        "valid":  torch.tensor(valid, dtype=torch.float32),                     # [H,W]
        "s": float(s),
        "ymax10": float(ymax10),
        "tmax": float(tmax)
    }
    torch.save(rec, os.path.join(outdir, f"{i}.pt"))

def generate_from_df(
    pickle_path: str,
    out_root: str,
    split: str = "test",
    kind: str = "hybrid",
    percentile_scale: float = 99.0,
    height: int = 288,
    width: int = 512
):
    global H, W
    H, W = height, width

    df = pd.read_pickle(pickle_path)

    outdir = os.path.join(out_root, split, kind)
    os.makedirs(outdir, exist_ok=True)

    print("Generating %s %s tensors to %s"%(len(df),kind,outdir))
    for i in tqdm(range(len(df))):
        tmp = df.iloc[i]
        try:
            #Try to make ER image
            NRim = tmp['NRim']
            ERim = make_im(tmp["ER_col"], tmp["ER_row"], tmp["ER_intensity"], bins=(W,H), ranges=((0,W),(0,H)))
            valid = torch.zeros([288,512]).float()
            valid[tmp['row'].min():tmp['row'].max()+1,tmp['col'].min():tmp['col'].max()+1]=1
        except:
            #if ER columns don't exist then the data is NR-only
            NRim = tmp['im']
            ERim = np.zeros([H,W])
            valid = torch.zeros([288,512]).float()
            valid[tmp['rowmin']:tmp['rowmax']+1,tmp['colmin']:tmp['colmax']+1]=1
        NRim[NRim <= 1] = 0 #signal to noise cut
        ERim[ERim <= 1] = 0
        hybridim = NRim + ERim
        # log scale hybrid
        hyb_scaled, s, ymax10 = scale_image_log10p(hybridim, pclip=percentile_scale, thresh=0.0)

        # log scale labels per shared s; normalize by tmax
        t_er = np.log10(1 + ERim / max(s,1e-6))
        t_nr = np.log10(1 + NRim / max(s,1e-6))
        tmax = float(max(t_er.max(), t_nr.max(), 1e-6))
        t_er_s = (t_er / tmax).astype(np.float32)
        t_nr_s = (t_nr / tmax).astype(np.float32)

        _save_dict(outdir, i, hyb_scaled, t_er_s, t_nr_s, s, ymax10, tmax, valid)

    print(f"Wrote {len(df)} tensors to {outdir}")

if __name__ == "__main__":
    from OASIS import OASISConfig
    """
    We're going to make it so we only have to generate tensors once.
    In this sample package we are only generating tensors for the
    test set so we can evaluate pre-trained models.
    """

    #Let's generate both for the hybrid and NR-only testsets
    #You can manually change the paths here for your use-case

    cfg = OASISConfig()
    split = 'test' # Can be train or test
    outpath = cfg.data_root
    dataframe_dir = cfg.dataframe_root
    dataframepaths = {'hybrid': os.path.join(dataframe_dir,'hybrid_test.pkl'),
                      'NR': os.path.join(dataframe_dir,'NR_test.pkl')}

    for (key,path) in zip(dataframepaths.keys(),dataframepaths.values()):
        generate_from_df(pickle_path = path,
                         out_root = outpath,
                         split = split,
                         kind = key)
