import json
from omegaconf import OmegaConf
import torch
import numpy as np


def convert_result_to_json(result: str):
    result = result.replace('```json', '').replace('```', '')
    return json.loads(result)


def load_json_from_file(path: str):
    return json.load(open(path, 'r'))

def save_json(data, path: str):
    return json.dump(data, open(path, 'w'), indent=4)


def load_config(path):
    config = OmegaConf.load(path)
    return config


def reproducibility_setting(seed):
    """
    set the random seed to make sure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    
    
def count_parameters(model: torch.nn.Module, dtype):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 2 if dtype == torch.float16 else 4  # 2 字节用于 FP16，4 字节用于 FP32
    total_memory_MB = total_params * bytes_per_param / (1024 ** 2)
    return total_params, total_memory_MB