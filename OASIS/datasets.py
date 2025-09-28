import os, glob, torch
from typing import Sequence, Tuple, List
from torch.utils.data import Dataset
import re

def _numeric_key(p: str):
    """Sort by the first integer found in the basename; fallback to name."""
    s = os.path.basename(p)
    m = re.search(r'\d+', s)
    return (int(m.group()), s) if m else (float('inf'), s)

class OASISDataset(Dataset):
    """
    Directory layout:
      tensors/
        train | val | test/
          hybrid/
            0.pt, 1.pt, ...
          NR/
            0.pt, 1.pt, ...

    Structure of each .pt file is:
      (dict)          {"input":[H,W], "target":[2,H,W], "valid":[H,W],
                       "s":float, "ymax10":float, "tmax":float}

    Returns per item:
      I_in  : [1,H,W]  float32
      t     : [2,H,W]  float32
      valid : [1,H,W]  float32
      aux   : {"s":..., "ymax10":..., "tmax":..., "meta":{...}, "kind": "hybrid"|"NR"}
    """
    def __init__(self, root_dir: str, split: str = "train", kinds: Sequence[str] = ("hybrid","NR"),
                 file_pattern: str = "*.pt", hw: Tuple[int,int] = (288,512)):
        self.root_dir = root_dir
        self.split = split
        self.kinds = tuple(kinds)
        self.H, self.W = hw

        self.paths: List[str] = []
        self.labels: List[str] = []
        self.ids:    List[int] = []
        for k in self.kinds:
            kdir = os.path.join(self.root_dir, self.split, k)
            matches = glob.glob(os.path.join(kdir, file_pattern))
            matches = sorted(matches, key=_numeric_key)  # <-- numeric sort

            self.paths.extend(matches)
            self.labels.extend([k] * len(matches))

            # track numeric ids (useful for verification)
            for p in matches:
                m = re.search(r'\d+', os.path.basename(p))
                self.ids.append(int(m.group()) if m else -1)

        if not self.paths:
            raise ValueError(f"No files under {self.root_dir}/{self.split}/{{{','.join(self.kinds)}}}/{file_pattern}")

    def __len__(self):
        return len(self.paths)

    def _load_record(self, p: str):
        d = torch.load(p, map_location="cpu")
        # Dict schema
        I   = d["input"]
        tgt = d["target"]
        val = d["valid"]
        s      = float(d.get("s", 1.0))
        ymax10 = float(d.get("ymax10", 1.0))
        tmax   = float(d.get("tmax", 1.0))
        
        if not torch.is_tensor(I):
            I = torch.tensor(I)
        if not torch.is_tensor(tgt):
            tgt = torch.tensor(tgt)
        if not torch.is_tensor(val):
            val = torch.tensor(val)

        I   = I.float()
        tgt = tgt.float()
        val = val.float()

        if I.dim() == 2:
            I = I.unsqueeze(0)     # [1,H,W]
        if val.dim() == 2:
            val = val.unsqueeze(0) # [1,H,W]

        if I.shape[-2:] != (self.H, self.W):
            raise ValueError(f"{p}: input has shape {tuple(I.shape)}, expected [1,{self.H},{self.W}]")
        if tgt.dim() != 3 or tgt.shape[0] != 2 or tgt.shape[-2:] != (self.H, self.W):
            raise ValueError(f"{p}: target has shape {tuple(tgt.shape)}, expected [2,{self.H},{self.W}]")
        if val.shape != (1, self.H, self.W):
            raise ValueError(f"{p}: valid has shape {tuple(val.shape)}, expected [1,{self.H},{self.W}]")

        aux = {"s": s, "ymax10": ymax10, "tmax": tmax}
        return I, tgt, val, aux

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        kind = self.labels[idx]
        I, tgt, val, aux = self._load_record(p)
        aux["kind"] = kind
        return I, tgt, val, aux
