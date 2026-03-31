import os
import torch
from torch.utils.data import DataLoader
from models.fusion_model import MultiModalDiagnosisModel
from training.custom_dataset import XrayMetadataDataset

def evaluate(data_dir, model_path='model.pth', device='cpu'):
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return

    csv_files = {
        'COVID': os.path.join(data_dir, 'COVID.metadata.xlsx'),
        'Normal': os.path.join(data_dir, 'Normal.metadata.xlsx'),
        'Lung Opacity': os.path.join(data_dir, 'Lung_Opacity.metadata.xlsx'),
        'Viral Pneumonia': os.path.join(data_dir, 'Viral Pneumonia.metadata.xlsx')
    }
    
    # Initialize Dataset and DataLoader
    print("Loading test dataset...")
    dataset = XrayMetadataDataset(data_dir=data_dir, csv_files=csv_files, img_size=64)
    if len(dataset) == 0:
        print("Dataset is empty. Evaluation aborted.")
        return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Initialize Model & Load Weights
    model = MultiModalDiagnosisModel(
        img_size=64, patch_size=8, in_channels=1, img_embed_dim=128, metadata_dim=5, n_classes=4
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating Model performance...")
    with torch.no_grad():
        for batch_idx, (imgs, metas, labels) in enumerate(dataloader):
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
            
            outputs = model(imgs, metas)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Batch [{batch_idx+1}/{len(dataloader)}] current accuracy: {100 * correct / total:.2f}%")
                
    accuracy = 100 * correct / total
    print(f"\nFinal Accuracy on {total} samples: {accuracy:.2f}%")

if __name__ == "__main__":
    DATA_PATH = os.path.join('data', 'COVID-19_Radiography_Dataset')
    evaluate(DATA_PATH, model_path='model.pth')
