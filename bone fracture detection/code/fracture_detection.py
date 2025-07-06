# To install required packages, run:
# pip install torch torchvision scikit-learn matplotlib

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Custom ImageFolder to skip corrupted images
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Skipping corrupted image: {self.samples[index][0]}")
            return None

# Load datasets
train_dataset = SafeImageFolder(root="data set/train", transform=transform)
test_dataset = SafeImageFolder(root="data set/test", transform=transform)

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

# Load pre-trained ResNet18
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/batch_count:.4f}")

# Function to generate saliency maps
def generate_saliency_map(model, images, labels):
    model.eval()
    images.requires_grad_()
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get the gradients
    saliency, _ = torch.max(images.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    return saliency

# Modified evaluation function to include saliency maps
def evaluate_model_with_saliency(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Generate and visualize saliency maps
            saliency_maps = generate_saliency_map(model, images, labels)
            
            # Visualize the first saliency map
            plt.imshow(saliency_maps[0], cmap='hot')
            plt.axis('off')
            plt.title("Saliency Map")
            plt.show()
    
    if len(all_labels) == 0:
        print("No valid images in test set")
        return
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Train, save, and evaluate
train_model(model, train_loader, criterion, optimizer, num_epochs=18)
torch.save(model.state_dict(), "fracture_model.pth")
evaluate_model_with_saliency(model, test_loader)