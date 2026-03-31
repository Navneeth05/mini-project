import torch
import torchvision.transforms as T
from PIL import Image

# --- Imports from your other files ---
from .utils import load_json, device_select
from .cnn_model import build_model
# -------------------------------------

def load_model(model_path, meta_path):
    """
    Loads the trained model, metadata, and inference transforms.
    """
    device = device_select()
    
    # Load metadata
    meta = load_json(meta_path)
    if not meta:
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    
    num_classes = len(meta.get("classes", []))
    if num_classes == 0:
        raise ValueError("Could not read 'classes' from metadata.json")

    # Build the exact same model structure as in train.py
    # Note: dropout is active during eval() by default, which is fine
    model = build_model(num_classes=num_classes, dropout=0.3) 
    
    # Load the saved weights
    try:
        # Load weights onto the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights from {model_path}")
        raise e

    model.to(device)
    model.eval() # Set model to evaluation mode

    # Define the *exact* same transforms as used in training/validation
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    print(f"Model loaded successfully. Classes: {meta['classes']}")
    return model, tf, meta, device

@torch.no_grad()
def predict_pil_image(model, tf, device, img: Image.Image):
    """
    Runs inference on a single PIL image.
    """
    # Ensure image is RGB, in case of RGBA or single-channel
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Apply transforms and add batch dimension
    img_t = tf(img).unsqueeze(0).to(device)

    # Get model output
    output = model(img_t)
    
    # Convert to probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    
    # Get the top confidence and class index
    confidence, index = torch.max(probs, 1)
    
    return index.item(), confidence.item(), probs.cpu().numpy()