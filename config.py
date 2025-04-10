import os
from constants import KAGGLE_DATA_PATH, KAGGLE_CIF_PATH, RIBNET_MODULES_PATH, RIBNET_WEIGHTS_PATH

# Config dictionary for RNA structure prediction
config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "model_config_path": os.path.join(RIBNET_MODULES_PATH, 'configs','pairwise.yaml'),
    "epochs": 600,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "min_len_filter": 10, 
    "structural_violation_epoch": 50,
    "balance_weight": False,
}

# Training specific configuration
train_config = {
    "epochs": 50,
    "cos_epoch": 35,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
} 