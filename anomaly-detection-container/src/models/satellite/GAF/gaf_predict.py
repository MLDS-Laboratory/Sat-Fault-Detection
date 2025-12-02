import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GAF.gaf_main import load_model
from GAF.gaf_transform import compute_gaf

def predict_gaf_image(model, ts, device='cpu'):
    """
    Make a prediction on a time series by converting it to GAF and running through model.
    
    Parameters:
    - model: Loaded model
    - ts: Time series data (numpy array)
    - device: Device to run inference on
    
    Returns:
    - Predicted class (0 or 1)
    - Class probabilities
    """
    # Apply GAF transformation
    gaf_img = compute_gaf(ts)
    gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-8)
    gaf_img = np.uint8(255 * gaf_img)
    img = Image.fromarray(gaf_img).resize((224, 224)).convert("RGB")
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probs.cpu().numpy()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions with a saved model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--model_type', type=str, default='pretrained', choices=['pretrained', 'scratch'],
                       help='Type of model architecture')
    parser.add_argument('--segment_path', type=str, required=True, 
                       help='Path to a pickled segment or numpy time series')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path, args.model_type)
    device = next(model.parameters()).device
    
    # Load time series
    if args.segment_path.endswith('.npy'):
        ts = np.load(args.segment_path)
    else:
        import pickle
        with open(args.segment_path, 'rb') as f:
            segment = pickle.load(f)
            ts = segment['ts']
    
    # Make prediction
    pred_class, probs = predict_gaf_image(model, ts, device)
    print(f"Prediction: {'Anomaly' if pred_class == 1 else 'Normal'}")
    print(f"Probabilities: Normal: {probs[0]:.4f}, Anomaly: {probs[1]:.4f}")