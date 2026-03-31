# backend/src/train.py
# Robust imports + GPU-safe training entrypoint.
# Works when run as:
#   python -m backend.src.train
# or
#   python backend/src/train.py

import sys
from pathlib import Path

# ---------------- determine project root and ensure importability ----------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = None
cand = HERE
for _ in range(8):
    if (cand / "backend").exists():
        PROJECT_ROOT = cand
        break
    cand = cand.parent
if PROJECT_ROOT is None:
    PROJECT_ROOT = HERE.parents[2]  # fallback

proj_root_str = str(PROJECT_ROOT)
if proj_root_str not in sys.path:
    sys.path.insert(0, proj_root_str)

# debug info
print("=== DEBUG: train.py startup ===")
print("This file:", HERE)
print("Detected PROJECT_ROOT:", PROJECT_ROOT)
print("sys.path[0]:", sys.path[0])
print("PROJECT_ROOT/backend exists?:", (PROJECT_ROOT / "backend").exists())
print("PROJECT_ROOT/backend/src exists?:", (PROJECT_ROOT / "backend" / "src").exists())
try:
    print("Listing backend/src:")
    for p in sorted((PROJECT_ROOT / "backend" / "src").iterdir()):
        print("   ", p.name)
except Exception as e:
    print("   (could not list):", e)
print("=== end debug ===\n")

# ---------------- try imports (relative preferred) ----------------
_import_err = None
try:
    # relative imports (works when running as package module)
    # --- FIX: Removed CLASS_NAMES ---
    from .cnn_model import build_model
    from .optimizer import DAHGA_PSO
    from .utils import set_seed, device_select, save_json
    print("Imported local modules with relative imports.")
except Exception as e_rel:
    _import_err = e_rel
    try:
        # fallback to top-level src.*
        # --- FIX: Removed CLASS_NAMES ---
        from src.cnn_model import build_model
        from src.optimizer import DAHGA_PSO
        from src.utils import set_seed, device_select, save_json
        print("Imported local modules with top-level 'src' imports.")
    except Exception as e_top:
        print("Relative import error:", e_rel)
        print("Top-level import error:", e_top)
        raise RuntimeError("Failed to import local modules. Check PROJECT_ROOT and backend/src.") from e_top

# ---------------- standard imports ----------------
import os
import copy
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

# ---------------- config & device ----------------
set_seed(42)
# Prefer the utility device_select if available, else fallback to torch detection
try:
    device = device_select()
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

DATA_ROOT = PROJECT_ROOT / "data" / "vision" / "pomegranate"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- dataset helpers ----------------
def build_loaders(batch_size=32, num_workers=2):
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(
            f"Missing train/val folders under {DATA_ROOT!s}.\n"
            f"Train exists: {train_dir.exists()}  Val exists: {val_dir.exists()}\n"
            f"Checked PROJECT_ROOT: {PROJECT_ROOT}"
        )

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=tf)

    # --- START FIX 1 ---
    # Get class info DIRECTLY from the dataset
    dataset_class_names = train_ds.classes
    num_classes = len(dataset_class_names)
    print(f"Found {num_classes} classes: {dataset_class_names}")
    # --- END FIX 1 ---

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Return the new class info
    return train_loader, val_loader, num_classes, dataset_class_names

# ---------------- training / eval ----------------
def train_one_epoch(model, dl, opt):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for x, y in tqdm(dl, leave=False, desc="Train"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = ce(out, y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(1, len(dl))

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    for x, y in tqdm(dl, leave=False, desc="Eval"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += ce(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / max(1, total)
    return acc, (loss / max(1, len(dl)))

# ---------------- HPO fitness & runner ----------------
def fitness_fn(hp):
    lr, drop = float(hp[0]), float(hp[1])

    # --- START FIX 2 ---
    # Get num_classes from the loader, ignore the class_names list for now
    train_loader, val_loader, num_classes, _ = build_loaders(batch_size=16)

    model = build_model(num_classes=num_classes, dropout=drop).to(device)
    # --- END FIX 2 ---

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # quick one epoch per trial
    train_one_epoch(model, train_loader, optimizer)
    val_acc, _ = evaluate(model, val_loader)
    return val_acc

def run_hpo():
    low = [1e-4, 0.1]
    high = [5e-3, 0.6]
    opt = DAHGA_PSO(pop_size=5, dim=2, low=low, high=high, max_iter=6)
    history = opt.step(lambda pos: fitness_fn(pos))
    save_json({"history": history, "best": float(opt.gbest_score), "gbest": opt.gbest.tolist()},
              str(MODEL_DIR / "hgs_pso_history.json"))
    return opt.gbest

# ---------------- main ----------------
if __name__ == "__main__":
    print("Running training with device:", device)
    best_hp = run_hpo()
    lr, drop = float(best_hp[0]), float(best_hp[1])
    print("HPO best:", best_hp)

    # --- START FIX 3 ---
    # Get num_classes AND the actual class names from the loader
    train_loader, val_loader, num_classes, found_class_names = build_loaders(batch_size=32)
    model = build_model(num_classes=num_classes, dropout=drop).to(device)
    # --- END FIX 3 ---

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_state = None
    epochs = 5
    for e in range(epochs):
        train_one_epoch(model, train_loader, optimizer)
        acc, _ = evaluate(model, val_loader)
        print(f"Epoch {e+1}/{epochs} - val_acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        torch.save(best_state, str(MODEL_DIR / "best_model.pth"))
        
        # --- START FIX 4 ---
        # Save the CORRECT class names discovered from the folder
        save_json({"classes": found_class_names}, str(MODEL_DIR / "metadata.json"))
        # --- END FIX 4 ---
        
        print("Saved best model to", MODEL_DIR / "best_model.pth")
    print("✅ Training done. Best val accuracy:", best_acc)