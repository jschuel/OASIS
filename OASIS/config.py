
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional
import os

@dataclass
class OASISConfig:
    """
    Single config dataclass used for training and evaluation.
    Parameters set here are passed into train() and evaluate()
    """
    # Data
    data_root: str = "../data/tensors" # path containing {train,val,test}/{hybrid,NR}/*.pt
    #data_root: str = "/media/jeef/ssd_data/Migdal/hybrids/segTensorsNew/"
    kinds: Sequence[str] = ("hybrid", "NR") # species subfolders for data_root
    split: str = "train" # used by dataset helpers
    hw: Tuple[int,int] = (288, 512) # (H,W) image dimensions
    file_pattern: str = "*.pt"
    dataframe_root: str = "../data/dataframes"

    # U--Net
    base_channels: int = 32

    # Training
    model_dir: str = "../data/models"
    best_training_weights: str = "best.pt" #name of weights file you want training to save
    last_training_weights: str = "last.pt"
    epochs: int = 50
    batch_size: int = 16
    lr: float = 2e-4
    patience: int = 10
    seed: int = 42
    device: str = "cuda" #Choose "cpu" or "cuda" depending on your system
    n_train_per_kind: int = 18000
    n_val_per_kind: int = 2000

    # Loss weights
    w_ER: float = 5.0
    w_NR: float = 1.0
    W_ER: float = 3.0
    W_NR: float = 1.0
    W_overlap: float = 6.0
    alpha_reg: float = 1.0
    alpha_tv: float = 0.01
    
    # Evaluation
    model_file: str = os.path.join(model_dir,"nominal_weights.pt") #name of weights file you want to use for evaluation
    eval_split: str = "test"
    eval_batch_size: int = 16
