"""
Real-time Defect Detection
Uses webcam to detect defects in products
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# ============================================
# Configuration
# ============================================
MODEL_PATH = 'defect_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# Load Model
# ============================================
def load_model():
    model = models.resnet18(pretrained=False)
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
def predict(frame, model):
    # Transform frame
    img = transform(frame)
    img = img.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
    
    return predicted.item(), confidence

# ============================================
# Main Loop
# ============================================
def main():
    print("Loading model...")
    model = load_model()
    
    print("Starting camera...")
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("✅ Camera started! Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Make prediction
        pred, conf = predict(frame, model)
        
        # Set label and color
        if pred == 0:  # defective
            label = "DEFECTIVE"
            color = (0, 0, 255)  # Red
        else:
            label = "NORMAL"
            color = (0, 255, 0)  # Green
        
        # Draw on frame
        cv2.putText(frame, f"{label}: {conf*100:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw bounding box
        h, w, _ = frame.shape
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), color, 3)
        
        # Show frame
        cv2.imshow('Defect Detection', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera closed!")

if __name__ == "__main__":
    main()

    
