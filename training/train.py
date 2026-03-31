import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.fusion_model import MultiModalDiagnosisModel
from training.custom_dataset import XrayMetadataDataset

def train(data_dir, batch_size=8, epochs=5, lr=1e-4, device='cpu', limit_batches=None):
    """
    Main training loop for the multi-modal diagnosis model.
    """
    # Define paths to metadata files (per class as requested)
    csv_files = {
        'COVID': os.path.join(data_dir, 'COVID.metadata.xlsx'),
        'Normal': os.path.join(data_dir, 'Normal.metadata.xlsx'),
        'Lung Opacity': os.path.join(data_dir, 'Lung_Opacity.metadata.xlsx'),
        'Viral Pneumonia': os.path.join(data_dir, 'Viral Pneumonia.metadata.xlsx')
    }
    
    # Initialize Dataset and DataLoader
    print("Loading dataset...")
    try:
        dataset = XrayMetadataDataset(data_dir=data_dir, csv_files=csv_files, img_size=64)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dataset = [] 
        
    if len(dataset) == 0:
        print("Dataset is empty. Check your data paths.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MultiModalDiagnosisModel(
        img_size=64, 
        patch_size=8, 
        in_channels=1, 
        img_embed_dim=128, 
        metadata_dim=5, 
        metadata_embed_dim=64, 
        n_classes=4
    ).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    print(f"Starting training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, (imgs, metas, labels) in enumerate(dataloader):
            # Batch limiting for fast prototyping
            if limit_batches and batch_idx >= limit_batches:
                break
                
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / min(len(dataloader), limit_batches if limit_batches else len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    DATA_PATH = os.path.join('data', 'COVID-19_Radiography_Dataset')
    train(DATA_PATH, batch_size=4, epochs=2, limit_batches=10)
