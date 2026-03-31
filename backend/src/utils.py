import torch
import random
import numpy as np
import json

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_select():
    """Selects the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def save_json(data, path):
    """Saves a dictionary to a JSON file."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save JSON to {path}: {e}")

def load_json(path):
    """Loads a JSON file from a path."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found at {path}")
        return {}
    except Exception as e:
        print(f"Failed to load JSON from {path}: {e}")
        return {}