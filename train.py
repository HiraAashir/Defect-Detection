"""
Defect Detection Model Training
Uses Transfer Learning with ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# Configuration
# ============================================
DATA_DIR = 'data/defects'
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ============================================
# Data Transforms
# ============================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# Create Dummy Dataset (For Demo)
# ============================================
def create_dummy_data():
    """Create sample images for demonstration"""
    
    # Create folders
    for split in ['train', 'test']:
        for category in ['defective', 'normal']:
            path = os.path.join(DATA_DIR, split, category)
            os.makedirs(path, exist_ok=True)
    
    # Generate sample images
    print("Creating dummy dataset...")
    
    # Create 50 defective images (with random noise)
    for i in range(50):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some "defect" patterns
        img[50:100, 50:100] = [255, 0, 0]  # Red square as defect
        img = Image.fromarray(img)
        img.save(os.path.join(DATA_DIR, 'train', 'defective', f'defect_{i}.jpg'))
    
    # Create 50 normal images (clean)
    for i in range(50):
        img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(DATA_DIR, 'train', 'normal', f'normal_{i}.jpg'))
    
    # Create test images
    for i in range(20):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img[50:100, 50:100] = [255, 0, 0]
        img = Image.fromarray(img)
        img.save(os.path.join(DATA_DIR, 'test', 'defective', f'defect_test_{i}.jpg'))
        
        img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(DATA_DIR, 'test', 'normal', f'normal_test_{i}.jpg'))
    
    print("Dummy dataset created!")

# Create data if not exists
if not os.path.exists(os.path.join(DATA_DIR, 'train', 'defective')):
    create_dummy_data()

# ============================================
# Load Data
# ============================================
train_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
test_dataset = ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes: {train_dataset.classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================
# Build Model (Transfer Learning)
# ============================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: defective, normal
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)

# ============================================
# Training Loop
# ============================================
print("\n🚀 Starting Training...")
train_losses = []
train_accs = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)
    
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

# ============================================
# Save Model
# ============================================
torch.save(model.state_dict(), 'defect_model.pth')
print("\n✅ Model saved as 'defect_model.pth'")

# ============================================
# Plot Results
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Training Accuracy', color='green')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print("\n🎉 Training Complete!")
