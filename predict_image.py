"""
Defect Detection - Image Prediction
Test with static images instead of webcam
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import os

# ============================================
# Configuration
# ============================================
MODEL_PATH = 'defect_model.pth'
TEST_IMAGE_DIR = 'test_images'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# Load Model
# ============================================
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# ============================================
# Image Transform
# ============================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# Prediction Function
# ============================================
def predict_image(image_path, model):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transform
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
    
    return predicted.item(), confidence, img

# ============================================
# Create Sample Test Images
# ============================================
def create_sample_images():
    """Create sample images for testing"""
    os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
    
    # Create defective image (with red square)
    img_defective = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_defective[50:150, 50:150] = [255, 0, 0]  # Red square = defect
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'defective_1.jpg'), img_defective)
    
    # Create normal image (clean)
    img_normal = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'normal_1.jpg'), img_normal)
    
    print(f"✅ Created sample images in {TEST_IMAGE_DIR}/")

# ============================================
# Main Function
# ============================================
def main():
    print("🔍 Loading model...")
    model = load_model()
    print("✅ Model loaded!")
    
    # Create sample images if not exist
    if not os.path.exists(os.path.join(TEST_IMAGE_DIR, 'defective_1.jpg')):
        create_sample_images()
    
    # Test all images in folder
    print(f"\n🖼️ Testing images from {TEST_IMAGE_DIR}/")
    print("-" * 50)
    
    for filename in os.listdir(TEST_IMAGE_DIR):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(TEST_IMAGE_DIR, filename)
            result = predict_image(image_path, model)
            
            if result:
                pred, conf, img = result
                label = "🔴 DEFECTIVE" if pred == 0 else "🟢 NORMAL"
                print(f"{filename}: {label} ({conf*100:.1f}%)")
                
                # Draw label on image
                color = (0, 0, 255) if pred == 0 else (0, 255, 0)
                cv2.putText(img, f"{label}: {conf*100:.1f}%", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Save result
                cv2.imwrite(f"results/{filename}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("-" * 50)
    print("✅ Results saved in results/ folder!")

if __name__ == "__main__":
    main()
